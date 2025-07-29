import os
import json
import pickle
import logging
import numpy as np
from typing import List, Dict, Any, Optional, Set, Tuple
from pathlib import Path
import time
import google.generativeai as genai
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_core.documents import Document
from pydantic import Field, PrivateAttr
from pyvis.network import Network
import networkx as nx
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from config import Config
from graph_interfaces import (
    GraphNode, GraphRelationship, GraphData, GraphSearchResult,
    EmbeddingService, GraphTransformer, GraphStorage, GraphVisualizer, 
    GraphSearcher, GraphService
)

logger = logging.getLogger(__name__)


class SentenceTransformerEmbedding(EmbeddingService):
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        self.model = SentenceTransformer(model_name)
    
    def encode(self, texts: List[str]) -> List[List[float]]:
        return self.model.encode(texts).tolist()


class GeminiChat(BaseChatModel):
    model_name: str = Field(default="gemini-pro")
    api_key: Optional[str] = Field(default=None)
    _model: Any = PrivateAttr(default=None)
    
    def __init__(self, model_name: str = "gemini-pro", api_key: Optional[str] = None, **kwargs):
        super().__init__(model_name=model_name, api_key=api_key, **kwargs)
        
        actual_api_key = api_key or os.environ.get("GEMINI_API_KEY")
        if not actual_api_key:
            raise ValueError("GEMINI_API_KEY must be provided")
            
        genai.configure(api_key=actual_api_key)
        self._model = genai.GenerativeModel(model_name)
    
    @property
    def model(self):
        return self._model

    def invoke(self, input_data, config=None, **kwargs):
        prompt = self._extract_prompt(input_data)
        response = self.model.generate_content(prompt)
        return AIMessage(content=response.text)

    def _generate(self, messages: List[BaseMessage], stop: Optional[List[str]] = None, 
                  run_manager: Optional[Any] = None, **kwargs: Any) -> ChatResult:
        prompt = self._extract_prompt_from_messages(messages)
        
        try:
            response = self.model.generate_content(prompt)
            message = AIMessage(content=response.text)
            generation = ChatGeneration(message=message)
            return ChatResult(generations=[generation])
        except Exception as e:
            error_message = AIMessage(content=f"Error generating response: {str(e)}")
            generation = ChatGeneration(message=error_message)
            return ChatResult(generations=[generation])

    def _extract_prompt(self, input_data) -> str:
        if isinstance(input_data, dict):
            return input_data.get("input", str(input_data))
        elif isinstance(input_data, str):
            return input_data
        elif isinstance(input_data, list):
            return self._extract_prompt_from_messages(input_data)
        return str(input_data)
    
    def _extract_prompt_from_messages(self, messages: List[BaseMessage]) -> str:
        return "\n".join([
            m.content for m in messages 
            if isinstance(m, (HumanMessage, AIMessage)) and hasattr(m, 'content')
        ])

    @property
    def _llm_type(self) -> str:
        return "gemini-chat"

    @property
    def _identifying_params(self) -> dict:
        return {"model_name": self.model_name}

    class Config:
        arbitrary_types_allowed = True


class LangChainGraphTransformer(GraphTransformer):
    def __init__(self, llm: BaseChatModel):
        self.transformer = LLMGraphTransformer(llm=llm)
    
    def transform_documents(self, documents: List[Document]) -> GraphData:
        combined_content = self._combine_document_contents(documents)
        combined_doc = Document(page_content=combined_content)
        
        graph_documents = self.transformer.convert_to_graph_documents([combined_doc])
        
        if not graph_documents or not graph_documents[0].nodes:
            return GraphData()
        
        return self._convert_to_graph_data(graph_documents[0])
    
    def _combine_document_contents(self, documents: List[Document]) -> str:
        combined_parts = []
        for i, doc in enumerate(documents):
            source = doc.metadata.get('source', f'Document_{i+1}')
            combined_parts.extend([f"=== {source} ===", doc.page_content, ""])
        return "\n".join(combined_parts)
    
    def _convert_to_graph_data(self, graph_doc) -> GraphData:
        graph_data = GraphData()
        
        node_map = {}
        for node in graph_doc.nodes:
            graph_node = GraphNode(node.id, node.type, getattr(node, 'properties', {}))
            graph_data.nodes.append(graph_node)
            node_map[node.id] = graph_node
        
        for rel in graph_doc.relationships:
            if rel.source.id in node_map and rel.target.id in node_map:
                graph_rel = GraphRelationship(
                    node_map[rel.source.id],
                    node_map[rel.target.id],
                    rel.type,
                    getattr(rel, 'properties', {})
                )
                graph_data.relationships.append(graph_rel)
        
        return graph_data


class FileGraphStorage(GraphStorage):
    def __init__(self, storage_directory: str):
        self.storage_directory = Path(storage_directory)
        self.storage_directory.mkdir(parents=True, exist_ok=True)
        self.graph_file = self.storage_directory / "graph_data.pkl"
        self.embeddings_file = self.storage_directory / "node_embeddings.pkl"
    
    def save(self, graph_data: GraphData, embeddings: Dict[str, List[float]]) -> None:
        try:
            serializable_data = self._serialize_graph_data(graph_data)
            
            with open(self.graph_file, 'wb') as f:
                pickle.dump(serializable_data, f)
            
            if embeddings:
                with open(self.embeddings_file, 'wb') as f:
                    pickle.dump(embeddings, f)
            
            logger.info(f"Graph data saved: {len(graph_data.nodes)} nodes, {len(graph_data.relationships)} relationships")
        except Exception as e:
            logger.error(f"Failed to save graph data: {e}")
            raise
    
    def load(self) -> Tuple[Optional[GraphData], Dict[str, List[float]]]:
        if not self.graph_file.exists():
            return None, {}
        
        try:
            with open(self.graph_file, 'rb') as f:
                serialized_data = pickle.load(f)
            
            graph_data = self._deserialize_graph_data(serialized_data)
            
            embeddings = {}
            if self.embeddings_file.exists():
                with open(self.embeddings_file, 'rb') as f:
                    embeddings = pickle.load(f)
            
            logger.info(f"Graph data loaded: {len(graph_data.nodes)} nodes, {len(graph_data.relationships)} relationships")
            return graph_data, embeddings
        except Exception as e:
            logger.error(f"Failed to load graph data: {e}")
            return None, {}
    
    def _serialize_graph_data(self, graph_data: GraphData) -> Dict:
        return {
            'nodes': [
                {
                    'id': node.id,
                    'type': node.type,
                    'properties': node.properties
                }
                for node in graph_data.nodes
            ],
            'relationships': [
                {
                    'source_id': rel.source.id,
                    'target_id': rel.target.id,
                    'type': rel.type,
                    'properties': rel.properties
                }
                for rel in graph_data.relationships
            ]
        }
    
    def _deserialize_graph_data(self, serialized_data: Dict) -> GraphData:
        graph_data = GraphData()
        node_map = {}
        
        for node_data in serialized_data['nodes']:
            node = GraphNode(
                node_data['id'],
                node_data['type'],
                node_data.get('properties', {})
            )
            graph_data.nodes.append(node)
            node_map[node.id] = node
        
        for rel_data in serialized_data['relationships']:
            source_id = rel_data['source_id']
            target_id = rel_data['target_id']
            
            if source_id in node_map and target_id in node_map:
                rel = GraphRelationship(
                    node_map[source_id],
                    node_map[target_id],
                    rel_data['type'],
                    rel_data.get('properties', {})
                )
                graph_data.relationships.append(rel)
        
        return graph_data


class PyvisGraphVisualizer(GraphVisualizer):
    def __init__(self, storage_directory: str):
        self.storage_directory = Path(storage_directory)
        self.storage_directory.mkdir(parents=True, exist_ok=True)
    
    def create_visualization(self, graph_data: GraphData, filename: str) -> str:
        try:
            net = self._create_pyvis_network()
            self._add_nodes_to_network(net, graph_data.nodes)
            self._add_edges_to_network(net, graph_data.relationships)
            
            output_path = self.storage_directory / filename
            return self._safe_save_visualization(net, output_path)
        except Exception as e:
            logger.error(f"Graph visualization failed: {e}")
            return ""
    
    def _create_pyvis_network(self) -> Network:
        net = Network(height="800px", width="100%", bgcolor="#222222", font_color="white")
        net.barnes_hut(
            gravity=-80000,
            central_gravity=0.01,
            spring_length=200,
            spring_strength=0.05,
            damping=0.09
        )
        return net
    
    def _add_nodes_to_network(self, net: Network, nodes: List[GraphNode]) -> None:
        node_colors = {
            'Person': '#ff6b6b', 'Organization': '#4ecdc4', 'Location': '#45b7d1',
            'Event': '#96ceb4', 'Concept': '#feca57', 'Document': '#ff9ff3',
            'Entity': '#a8e6cf', 'Default': '#c7ecee'
        }
        
        for node in nodes:
            try:
                color = node_colors.get(node.type, node_colors['Default'])
                title = self._create_node_title(node)
                label = self._truncate_text(node.id, 50)
                
                net.add_node(node.id, label=label, title=title, color=color, size=25)
            except Exception as e:
                logger.warning(f"Failed to add node {node.id}: {e}")
    
    def _add_edges_to_network(self, net: Network, relationships: List[GraphRelationship]) -> None:
        for rel in relationships:
            try:
                edge_title = f"Relationship: {rel.type}"
                if rel.properties:
                    props_str = self._truncate_text(str(rel.properties), 100)
                    edge_title += f"\nProperties: {props_str}"
                
                net.add_edge(
                    rel.source.id,
                    rel.target.id,
                    label=self._truncate_text(rel.type, 20),
                    title=edge_title,
                    color={'color': '#848484'},
                    width=2
                )
            except Exception as e:
                logger.warning(f"Failed to add relationship {rel.source.id} -> {rel.target.id}: {e}")
    
    def _create_node_title(self, node: GraphNode) -> str:
        title = f"ID: {node.id}\nType: {node.type}"
        if node.properties:
            props_str = self._truncate_text(str(node.properties), 200)
            title += f"\nProperties: {props_str}"
        return title
    
    def _truncate_text(self, text: str, max_length: int) -> str:
        return text[:max_length] + "..." if len(text) > max_length else text
    
    def _safe_save_visualization(self, net: Network, output_path: Path) -> str:
        try:
            net.save_graph(str(output_path))
            if output_path.exists() and output_path.stat().st_size > 0:
                logger.info(f"Graph visualization saved to {output_path}")
                return str(output_path)
        except Exception as e:
            logger.error(f"Failed to save to {output_path}: {e}")
        
        return self._fallback_save(net)
    
    def _fallback_save(self, net: Network) -> str:
        import tempfile
        try:
            temp_dir = Path(tempfile.gettempdir())
            fallback_path = temp_dir / f"graph_fallback_{int(time.time())}.html"
            
            html_content = net.generate_html()
            if html_content:
                fallback_path.write_text(html_content, encoding='utf-8')
                logger.info(f"Graph saved using fallback method: {fallback_path}")
                return str(fallback_path)
        except Exception as e:
            logger.error(f"Fallback save method failed: {e}")
        
        return ""


class SemanticGraphSearcher(GraphSearcher):
    def __init__(self, embedding_service: EmbeddingService):
        self.embedding_service = embedding_service
        self.nx_graph = None
    
    def search(self, query: str, graph_data: GraphData, embeddings: Dict[str, List[float]], max_nodes: int) -> GraphSearchResult:
        try:
            self._build_networkx_graph(graph_data)
            entry_points = self._find_semantic_entry_points(query, embeddings)
            
            if not entry_points:
                return GraphSearchResult("No relevant entry points found", [], [], [])
            
            subgraph_nodes, subgraph_rels = self._build_subgraph(entry_points, max_nodes)
            context = self._create_context(query, subgraph_nodes, subgraph_rels)
            
            return GraphSearchResult(context, subgraph_nodes, subgraph_rels, entry_points)
        except Exception as e:
            logger.error(f"Graph search failed: {e}")
            return GraphSearchResult("Error searching graph", [], [], [])
    
    def _build_networkx_graph(self, graph_data: GraphData) -> None:
        self.nx_graph = nx.Graph()
        
        for node in graph_data.nodes:
            self.nx_graph.add_node(node.id, type=node.type, properties=node.properties)
        
        for rel in graph_data.relationships:
            self.nx_graph.add_edge(rel.source.id, rel.target.id, type=rel.type, properties=rel.properties)
    
    def _find_semantic_entry_points(self, query: str, embeddings: Dict[str, List[float]]) -> List[Tuple[str, float]]:
        if not embeddings:
            return []
        
        try:
            query_embedding = self.embedding_service.encode([query])[0]
            similarities = []
            
            for node_id, node_embedding in embeddings.items():
                similarity = cosine_similarity([query_embedding], [node_embedding])[0][0]
                if similarity > 0.3:
                    similarities.append((node_id, float(similarity)))
            
            similarities.sort(key=lambda x: x[1], reverse=True)
            return similarities[:5]
        except Exception as e:
            logger.error(f"Failed to find semantic entry points: {e}")
            return []
    
    def _build_subgraph(self, entry_points: List[Tuple[str, float]], max_nodes: int) -> Tuple[List[Dict], List[Dict]]:
        if not self.nx_graph:
            return [], []
        
        visited = set()
        nodes = []
        relationships = []
        
        from heapq import heappush, heappop
        queue = []
        
        for node_id, score in entry_points:
            heappush(queue, (-score, node_id, 0))
        
        while queue and len(visited) < max_nodes:
            neg_score, current_node, depth = heappop(queue)
            
            if current_node in visited:
                continue
            
            visited.add(current_node)
            node_details = self._get_node_details(current_node)
            if node_details:
                nodes.append(node_details)
            
            if depth < 2:
                for neighbor in self.nx_graph.neighbors(current_node):
                    if neighbor not in visited:
                        neighbor_score = -neg_score * 0.7
                        heappush(queue, (-neighbor_score, neighbor, depth + 1))
                    
                    rel_details = self._get_relationship_details(current_node, neighbor)
                    if rel_details and rel_details not in relationships:
                        relationships.append(rel_details)
        
        return nodes, relationships
    
    def _get_node_details(self, node_id: str) -> Optional[Dict]:
        if not self.nx_graph.has_node(node_id):
            return None
        
        node_data = self.nx_graph.nodes[node_id]
        return {
            "id": node_id,
            "type": node_data.get("type", "Unknown"),
            "properties": node_data.get("properties", {})
        }
    
    def _get_relationship_details(self, source_id: str, target_id: str) -> Optional[Dict]:
        if not self.nx_graph.has_edge(source_id, target_id):
            return None
        
        edge_data = self.nx_graph.edges[source_id, target_id]
        return {
            "source": source_id,
            "target": target_id,
            "type": edge_data.get("type", "RELATED"),
            "properties": edge_data.get("properties", {})
        }
    
    def _create_context(self, query: str, nodes: List[Dict], relationships: List[Dict]) -> str:
        if not nodes:
            return "Tidak ditemukan informasi graph yang relevan dengan pertanyaan."
        
        context_parts = ["Entitas yang relevan dengan pertanyaan:"]
        
        nodes_by_type = {}
        for node in nodes:
            node_type = node.get('type', 'Unknown')
            nodes_by_type.setdefault(node_type, []).append(node)
        
        for node_type, type_nodes in nodes_by_type.items():
            context_parts.append(f"\n{node_type}:")
            for node in type_nodes[:5]:
                properties_str = self._format_properties(node.get('properties', {}))
                context_parts.append(f"  - {node['id']}{properties_str}")
        
        if relationships:
            context_parts.append("\nHubungan antar entitas:")
            rels_by_type = {}
            for rel in relationships:
                rel_type = rel.get('type', 'RELATED')
                rels_by_type.setdefault(rel_type, []).append(rel)
            
            for rel_type, type_rels in rels_by_type.items():
                context_parts.append(f"\n{rel_type}:")
                for rel in type_rels[:5]:
                    context_parts.append(f"  - {rel['source']} → {rel['target']}")
        
        return "\n".join(context_parts)
    
    def _format_properties(self, properties: Dict) -> str:
        if not properties:
            return ""
        
        key_props = list(properties.keys())[:2]
        if key_props:
            prop_items = [f"{k}: {properties[k]}" for k in key_props]
            return f" ({', '.join(prop_items)})"
        return ""


class SemanticGraphService(GraphService):
    def __init__(self, config: Config):
        self.config = config
        self.graph_data = None
        self.embeddings = {}
        
        self.llm = GeminiChat(config.GRAPH_LLM_MODEL, config.GEMINI_API_KEY)
        self.transformer = LangChainGraphTransformer(self.llm)
        self.embedding_service = SentenceTransformerEmbedding()
        self.storage = FileGraphStorage(config.GRAPH_STORE_DIRECTORY)
        self.visualizer = PyvisGraphVisualizer(config.GRAPH_STORE_DIRECTORY)
        self.searcher = SemanticGraphSearcher(self.embedding_service)
        
        self._load_existing_data()
    
    def process_documents(self, documents: List[Document]) -> bool:
        if not self.config.ENABLE_GRAPH_PROCESSING:
            return False
        
        try:
            logger.info(f"Processing {len(documents)} documents to graph...")
            
            self.graph_data = self.transformer.transform_documents(documents)
            
            if not self.graph_data.nodes:
                logger.warning("No graph data generated from documents")
                return False
            
            self._generate_embeddings()
            self.storage.save(self.graph_data, self.embeddings)
            
            logger.info(f"Graph processing complete: {len(self.graph_data.nodes)} nodes, {len(self.graph_data.relationships)} relationships")
            return True
        except Exception as e:
            logger.error(f"Graph processing failed: {e}")
            return False
    
    def update_with_documents(self, documents: List[Document]) -> bool:
        if not self.config.ENABLE_GRAPH_PROCESSING:
            return False
        
        try:
            new_graph_data = self.transformer.transform_documents(documents)
            
            if not new_graph_data.nodes:
                return False
            
            if self.graph_data is None:
                self.graph_data = new_graph_data
            else:
                self._merge_graph_data(new_graph_data)
            
            self._generate_embeddings()
            self.storage.save(self.graph_data, self.embeddings)
            
            logger.info(f"Graph updated: {len(self.graph_data.nodes)} nodes, {len(self.graph_data.relationships)} relationships")
            return True
        except Exception as e:
            logger.error(f"Graph update failed: {e}")
            return False
    
    def search(self, query: str, max_nodes: int = 20) -> GraphSearchResult:
        if not self.has_data():
            return GraphSearchResult("No graph data available", [], [], [])
        
        return self.searcher.search(query, self.graph_data, self.embeddings, max_nodes)
    
    def visualize(self, filename: str) -> str:
        if not self.has_data():
            return ""
        
        return self.visualizer.create_visualization(self.graph_data, filename)
    
    def has_data(self) -> bool:
        return self.graph_data is not None and len(self.graph_data.nodes) > 0
    
    def get_stats(self) -> Dict[str, Any]:
        if not self.has_data():
            return {"nodes": 0, "relationships": 0, "node_types": [], "relationship_types": []}
        
        node_types = list(set(node.type for node in self.graph_data.nodes))
        rel_types = list(set(rel.type for rel in self.graph_data.relationships))
        
        return {
            "nodes": len(self.graph_data.nodes),
            "relationships": len(self.graph_data.relationships),
            "node_types": node_types,
            "relationship_types": rel_types,
            "has_embeddings": len(self.embeddings) > 0,
            "embedding_coverage": len(self.embeddings) / len(self.graph_data.nodes) if self.graph_data.nodes else 0
        }
    
    def _load_existing_data(self) -> None:
        self.graph_data, self.embeddings = self.storage.load()
        if self.graph_data:
            logger.info("Loaded existing graph data")
    
    def _generate_embeddings(self) -> None:
        if not self.graph_data:
            return
        
        try:
            texts = []
            node_ids = []
            
            for node in self.graph_data.nodes:
                text_parts = [node.id, node.type]
                if node.properties:
                    text_parts.extend([f"{k}: {str(v)}" for k, v in node.properties.items()])
                
                texts.append(" ".join(text_parts))
                node_ids.append(node.id)
            
            if texts:
                embeddings_list = self.embedding_service.encode(texts)
                self.embeddings = dict(zip(node_ids, embeddings_list))
                logger.info(f"Generated embeddings for {len(self.embeddings)} nodes")
        except Exception as e:
            logger.error(f"Failed to generate embeddings: {e}")
    
    def _merge_graph_data(self, new_graph_data: GraphData) -> None:
        if not self.graph_data:
            self.graph_data = new_graph_data
            return
        
        existing_node_keys = {f"{node.id}_{node.type}".lower() for node in self.graph_data.nodes}
        existing_rel_keys = {f"{rel.source.id}_{rel.type}_{rel.target.id}".lower() for rel in self.graph_data.relationships}
        
        nodes_added = 0
        rels_added = 0
        
        for node in new_graph_data.nodes:
            node_key = f"{node.id}_{node.type}".lower()
            if node_key not in existing_node_keys:
                self.graph_data.nodes.append(node)
                existing_node_keys.add(node_key)
                nodes_added += 1
        
        for rel in new_graph_data.relationships:
            rel_key = f"{rel.source.id}_{rel.type}_{rel.target.id}".lower()
            if rel_key not in existing_rel_keys:
                self.graph_data.relationships.append(rel)
                existing_rel_keys.add(rel_key)
                rels_added += 1
        
        logger.info(f"Merged graph data: {nodes_added} new nodes, {rels_added} new relationships")