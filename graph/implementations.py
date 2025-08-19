import pickle
import json
import logging
import networkx as nx
import numpy as np
from typing import List, Dict, Any, Optional, Set, Tuple
from pathlib import Path
from langchain_community.chat_models import ChatOllama
from langchain_community.llms import Ollama 
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_core.documents import Document
from pydantic import Field, PrivateAttr
from pyvis.network import Network
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from config.settings import settings
from graph.interfaces import (
    GraphNode, GraphRelationship, GraphData, GraphSearchResult,
    EmbeddingService, GraphTransformer, GraphStorage, GraphVisualizer, 
    GraphSearcher, GraphService, GraphException
)

logger = logging.getLogger(__name__)


class SentenceTransformerEmbedding(EmbeddingService):
    """Local sentence transformer embedding service"""
    
    def __init__(self, model_path: str):
        try:
            self.model = SentenceTransformer(model_path, trust_remote_code=True)
            logger.info(f"SentenceTransformer loaded from: {model_path}")
        except Exception as e:
            logger.warning(f"Failed to load local model, using fallback: {e}")
            self.model = SentenceTransformer('all-MiniLM-L6-v2')
    
    def encode(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for list of texts"""
        try:
            embeddings = self.model.encode(texts, normalize_embeddings=True)
            return embeddings.tolist()
        except Exception as e:
            logger.error(f"Encoding failed: {e}")
            return [[0.0] * 384 for _ in texts]  # Fallback embeddings


class OllamaChat(BaseChatModel):
    """Ollama chat model wrapper for graph processing"""
    
    model_name: str = Field(default="qwen3:4b")
    base_url: str = Field(default="http://localhost:11434")
    _model: Any = PrivateAttr(default=None)
    
    def __init__(self, model_name: str = None, base_url: str = None, **kwargs):
        super().__init__(
            model_name=model_name or settings.model.graph_llm_model,
            base_url=base_url or settings.model.ollama_base_url,
            **kwargs
        )
        self._model = ChatOllama(
            model=self.model_name,
            base_url=self.base_url,
            temperature=0.0,
            num_ctx=8192,
            top_p=0.9,
            top_k=40
        )
            
    @property
    def model(self):
        return self._model

    def invoke(self, input_data, config=None, **kwargs):
        """Invoke the model with input data"""
        prompt = self._extract_prompt(input_data)
        try:
            response = self.model.invoke(prompt)
            return response
        except Exception as e:
            logger.error(f"Ollama invoke failed: {e}")
            # Return a mock response for fallback
            return AIMessage(content="Unable to process graph extraction at this time.")

    def _generate(self, messages: List[BaseMessage], stop: Optional[List[str]] = None, 
                  run_manager: Optional[Any] = None, **kwargs: Any) -> ChatResult:
        """Generate response from messages"""
        prompt = self._extract_prompt_from_messages(messages)
        
        try:
            response = self.model.invoke(prompt)
            message = AIMessage(content=response.content if hasattr(response, 'content') else str(response))
            generation = ChatGeneration(message=message)
            return ChatResult(generations=[generation])
        except Exception as e:
            logger.error(f"Generation failed: {e}")
            message = AIMessage(content="Graph processing unavailable")
            generation = ChatGeneration(message=message)
            return ChatResult(generations=[generation])

    def _extract_prompt(self, input_data) -> str:
        """Extract prompt from various input types"""
        if isinstance(input_data, dict):
            return input_data.get('text', input_data.get('prompt', str(input_data)))
        elif isinstance(input_data, str):
            return input_data
        elif isinstance(input_data, list):
            return "\n".join(str(item) for item in input_data)
        return str(input_data)
    
    def _extract_prompt_from_messages(self, messages: List[BaseMessage]) -> str:
        """Extract prompt from message list"""
        return "\n".join([
            m.content for m in messages 
            if isinstance(m, (HumanMessage, AIMessage)) and hasattr(m, 'content')
        ])

    @property
    def _llm_type(self) -> str:
        return "ollama_chat"

    @property
    def _identifying_params(self) -> dict:
        return {"model_name": self.model_name, "base_url": self.base_url}

    class Config:
        arbitrary_types_allowed = True


class LangChainGraphTransformer(GraphTransformer):
    """Transform documents to graph using LangChain"""
    
    def __init__(self, llm: BaseChatModel):
        self.transformer = LLMGraphTransformer(llm=llm)
    
    def transform_documents(self, documents: List[Document]) -> GraphData:
        """Transform documents into graph data"""
        if not documents:
            return GraphData()

        logger.info(f"Starting graph transformation for {len(documents)} documents")
        
        try:
            # Limit documents to prevent context overflow
            max_docs = min(settings.document.max_docs_for_graph, len(documents))
            combined_content = self._combine_document_contents(documents[:max_docs])

            logger.info(f"Combined content length: {len(combined_content)} characters")
            
            # Create enhanced prompt for graph extraction
            graph_prompt = f"""
            Extract entities and relationships from the following legal documents.
            Focus on:
            - Legal entities (organizations, laws, regulations, articles, sections)
            - Key concepts and definitions
            - Relationships between them (REGULATES, APPLIES_TO, REFERENCES, DEFINES, etc.)
            
            Document content:
            {combined_content}
            
            Extract clear, specific entities and their meaningful relationships.
            """
            
            combined_doc = Document(page_content=graph_prompt)
            
            # Transform using LangChain
            graph_documents = self.transformer.convert_to_graph_documents([combined_doc])
            
            if not graph_documents:
                logger.warning("No graph documents generated")
                return GraphData()
            
            # Convert to our GraphData format
            graph_data = self._convert_to_graph_data(graph_documents[0])
            logger.info(f"Created graph with {len(graph_data.nodes)} nodes and {len(graph_data.relationships)} relationships")
            
            return graph_data
            
        except Exception as e:
            logger.error(f"Graph transformation failed: {e}")
            return GraphData()
    
    def _combine_document_contents(self, documents: List[Document]) -> str:
        """Combine document contents with length limits"""
        combined_parts = []
        total_length = 0
        max_combined_length = settings.document.max_combined_content_length
        
        for i, doc in enumerate(documents):
            content = doc.page_content[:settings.document.doc_truncate_length]
            
            if total_length + len(content) > max_combined_length:
                break
            
            # Add document separator and metadata
            source = doc.metadata.get('source', f'Document {i+1}')
            doc_header = f"\n--- {source} ---\n"
            combined_parts.append(doc_header + content)
            total_length += len(doc_header) + len(content)
        
        return "\n".join(combined_parts)
    
    def _convert_to_graph_data(self, graph_doc) -> GraphData:
        """Convert LangChain graph document to our GraphData format"""
        graph_data = GraphData()
        
        # Create node map for relationship linking
        node_map = {}
        
        # Add nodes
        for node in graph_doc.nodes:
            graph_node = GraphNode(
                id=node.id,
                type=node.type if hasattr(node, 'type') else 'Entity',
                properties={
                    'name': getattr(node, 'id', ''),
                    'description': getattr(node, 'properties', {}).get('description', ''),
                    **getattr(node, 'properties', {})
                }
            )
            graph_data.add_node(graph_node)
            node_map[node.id] = graph_node
        
        # Add relationships
        for rel in graph_doc.relationships:
            if rel.source.id in node_map and rel.target.id in node_map:
                graph_rel = GraphRelationship(
                    source=node_map[rel.source.id],
                    target=node_map[rel.target.id],
                    type=rel.type if hasattr(rel, 'type') else 'RELATED_TO',
                    properties=getattr(rel, 'properties', {})
                )
                graph_data.add_relationship(graph_rel)
        
        return graph_data


class FileGraphStorage(GraphStorage):
    """File-based graph storage"""
    
    def __init__(self, storage_directory: str):
        self.storage_directory = Path(storage_directory)
        self.storage_directory.mkdir(parents=True, exist_ok=True)
        self.graph_file = self.storage_directory / "graph_data.json"
        self.embeddings_file = self.storage_directory / "node_embeddings.pkl"
    
    def save(self, graph_data: GraphData, embeddings: Dict[str, List[float]]) -> None:
        """Save graph data and embeddings to files"""
        try:
            # Save graph data as JSON
            serialized_data = self._serialize_graph_data(graph_data)
            with open(self.graph_file, 'w', encoding='utf-8') as f:
                json.dump(serialized_data, f, indent=2, ensure_ascii=False)
            
            # Save embeddings as pickle
            with open(self.embeddings_file, 'wb') as f:
                pickle.dump(embeddings, f)
            
            logger.info(f"Saved graph data: {len(graph_data.nodes)} nodes, {len(graph_data.relationships)} relationships")
            
        except Exception as e:
            logger.error(f"Failed to save graph data: {e}")
            raise GraphException(f"Graph save failed: {e}")
    
    def load(self) -> Tuple[Optional[GraphData], Dict[str, List[float]]]:
        """Load graph data and embeddings from files"""
        if not self.graph_file.exists():
            return None, {}
        
        try:
            # Load graph data
            with open(self.graph_file, 'r', encoding='utf-8') as f:
                serialized_data = json.load(f)
            
            graph_data = self._deserialize_graph_data(serialized_data)
            
            # Load embeddings
            embeddings = {}
            if self.embeddings_file.exists():
                with open(self.embeddings_file, 'rb') as f:
                    embeddings = pickle.load(f)
            
            logger.info(f"Loaded graph data: {len(graph_data.nodes)} nodes, {len(graph_data.relationships)} relationships")
            return graph_data, embeddings
            
        except Exception as e:
            logger.error(f"Failed to load graph data: {e}")
            return None, {}
    
    def _serialize_graph_data(self, graph_data: GraphData) -> Dict:
        """Serialize graph data to JSON-compatible format"""
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
        """Deserialize JSON data back to GraphData"""
        graph_data = GraphData()
        node_map = {}
        
        # Recreate nodes
        for node_data in serialized_data['nodes']:
            node = GraphNode(
                id=node_data['id'],
                type=node_data['type'],
                properties=node_data['properties']
            )
            graph_data.add_node(node)
            node_map[node.id] = node
        
        # Recreate relationships
        for rel_data in serialized_data['relationships']:
            if rel_data['source_id'] in node_map and rel_data['target_id'] in node_map:
                rel = GraphRelationship(
                    source=node_map[rel_data['source_id']],
                    target=node_map[rel_data['target_id']],
                    type=rel_data['type'],
                    properties=rel_data['properties']
                )
                graph_data.add_relationship(rel)
        
        return graph_data


class PyvisGraphVisualizer(GraphVisualizer):
    """Graph visualization using Pyvis"""
    
    def __init__(self, storage_directory: str):
        self.storage_directory = Path(storage_directory)
        self.storage_directory.mkdir(parents=True, exist_ok=True)
    
    def create_visualization(self, graph_data: GraphData, filename: str) -> str:
        """Create interactive graph visualization"""
        try:
            output_path = self.storage_directory / filename
            
            # Create pyvis network
            net = self._create_pyvis_network()
            
            # Add nodes and edges
            self._add_nodes_to_network(net, graph_data.nodes)
            self._add_edges_to_network(net, graph_data.relationships)
            
            # Save visualization
            visualization_path = self._safe_save_visualization(net, output_path)
            logger.info(f"Graph visualization saved to: {visualization_path}")
            
            return str(visualization_path)
            
        except Exception as e:
            logger.error(f"Graph visualization failed: {e}")
            return self._fallback_save(net if 'net' in locals() else None)
    
    def _create_pyvis_network(self) -> Network:
        """Create configured pyvis network"""
        net = Network(
            height=settings.graph.graph_visualization_height,
            width=settings.graph.graph_visualization_width,
            bgcolor="#222222",
            font_color="white"
        )
        
        # Configure physics
        net.barnes_hut(
            gravity=-80000,
            central_gravity=0.01,
            spring_length=200,
            spring_strength=0.05,
            damping=0.09
        )
        
        return net
    
    def _add_nodes_to_network(self, net: Network, nodes: List[GraphNode]) -> None:
        """Add nodes to pyvis network"""
        from config.constants import GRAPH_NODE_COLORS
        
        for node in nodes:
            color = GRAPH_NODE_COLORS.get(node.type, GRAPH_NODE_COLORS['Default'])
            title = self._create_node_title(node)
            label = self._truncate_text(node.id, 30)
            
            net.add_node(
                node.id,
                label=label,
                title=title,
                color=color,
                size=settings.graph.graph_node_size,
                font={'size': 12}
            )
    
    def _add_edges_to_network(self, net: Network, relationships: List[GraphRelationship]) -> None:
        """Add edges to pyvis network"""
        for rel in relationships:
            net.add_edge(
                rel.source.id,
                rel.target.id,
                label=rel.type,
                title=f"{rel.type}: {rel.source.id} -> {rel.target.id}",
                width=settings.graph.graph_edge_width,
                color={'color': '#848484', 'highlight': '#ff6b6b'}
            )
    
    def _create_node_title(self, node: GraphNode) -> str:
        """Create detailed node title for hover"""
        title_parts = [f"ID: {node.id}", f"Type: {node.type}"]
        
        for key, value in node.properties.items():
            if key != 'name':  # Already shown in ID
                title_parts.append(f"{key.title()}: {self._truncate_text(str(value), 50)}")
        
        return "<br>".join(title_parts)
    
    def _truncate_text(self, text: str, max_length: int) -> str:
        """Truncate text with ellipsis"""
        return text[:max_length] + "..." if len(text) > max_length else text
    
    def _safe_save_visualization(self, net: Network, output_path: Path) -> str:
        """Safely save visualization with error handling"""
        try:
            net.save_graph(str(output_path))
            return str(output_path)
        except Exception as e:
            logger.warning(f"Failed to save to {output_path}: {e}")
            return self._fallback_save(net)
    
    def _fallback_save(self, net: Network) -> str:
        """Fallback save method"""
        try:
            fallback_path = self.storage_directory / "graph_visualization_fallback.html"
            if net:
                net.save_graph(str(fallback_path))
            else:
                # Create minimal HTML file
                with open(fallback_path, 'w') as f:
                    f.write("<html><body><h1>Graph visualization unavailable</h1></body></html>")
            return str(fallback_path)
        except Exception as e:
            logger.error(f"Fallback save failed: {e}")
            return "visualization_failed.html"


class SemanticGraphSearcher(GraphSearcher):
    """Semantic search within graph using embeddings"""
    
    def __init__(self, embedding_service: EmbeddingService):
        self.embedding_service = embedding_service
        self.graph = None
        self.node_details = {}
    
    def search(self, query: str, graph_data: GraphData, embeddings: Dict[str, List[float]], max_nodes: int) -> GraphSearchResult:
        """Search graph using semantic similarity"""
        if not graph_data.nodes or not embeddings:
            return GraphSearchResult()
        
        try:
            # Build NetworkX graph for traversal
            self._build_networkx_graph(graph_data)
            
            # Find semantic entry points
            entry_points = self._find_semantic_entry_points(query, embeddings)
            
            if not entry_points:
                logger.warning("No semantic entry points found")
                return GraphSearchResult()
            
            # Build relevant subgraph
            nodes, relationships = self._build_subgraph(entry_points, max_nodes)
            
            # Create context
            context = self._create_context(query, nodes, relationships)
            
            # Calculate scores
            scores = {node['id']: score for node, score in zip(nodes, [ep[1] for ep in entry_points[:len(nodes)]])}
            
            result = GraphSearchResult()
            result.entities = nodes
            result.relationships = relationships
            result.context = context
            result.scores = scores
            
            logger.info(f"Graph search found {len(nodes)} nodes and {len(relationships)} relationships")
            return result
            
        except Exception as e:
            logger.error(f"Graph search failed: {e}")
            return GraphSearchResult()
    
    def _build_networkx_graph(self, graph_data: GraphData) -> None:
        """Build NetworkX graph for traversal"""
        self.graph = nx.Graph()
        self.node_details = {}
        
        # Add nodes
        for node in graph_data.nodes:
            self.graph.add_node(node.id)
            self.node_details[node.id] = {
                'id': node.id,
                'type': node.type,
                'properties': node.properties
            }
        
        # Add edges
        for rel in graph_data.relationships:
            self.graph.add_edge(rel.source.id, rel.target.id, relationship_type=rel.type)
    
    def _find_semantic_entry_points(self, query: str, embeddings: Dict[str, List[float]]) -> List[Tuple[str, float]]:
        """Find most relevant nodes as entry points"""
        try:
            # Get query embedding
            query_embedding = self.embedding_service.encode([query])[0]
            
            # Calculate similarities
            similarities = []
            for node_id, node_embedding in embeddings.items():
                if node_id in self.node_details:
                    similarity = cosine_similarity([query_embedding], [node_embedding])[0][0]
                    similarities.append((node_id, similarity))
            
            # Sort by similarity and return top candidates
            similarities.sort(key=lambda x: x[1], reverse=True)
            
            # Filter by threshold and limit
            threshold = settings.retrieval.similarity_threshold
            entry_points = [(node_id, score) for node_id, score in similarities 
                           if score > threshold][:settings.retrieval.graph_entry_points_limit]
            
            logger.info(f"Found {len(entry_points)} semantic entry points")
            return entry_points
            
        except Exception as e:
            logger.error(f"Failed to find entry points: {e}")
            return []
    
    def _build_subgraph(self, entry_points: List[Tuple[str, float]], max_nodes: int) -> Tuple[List[Dict], List[Dict]]:
        """Build subgraph around entry points"""
        visited_nodes = set()
        result_nodes = []
        result_relationships = []
        
        # Start with entry points
        for node_id, score in entry_points:
            if len(visited_nodes) >= max_nodes:
                break
            
            if node_id not in visited_nodes:
                visited_nodes.add(node_id)
                node_details = self._get_node_details(node_id)
                if node_details:
                    node_details['similarity_score'] = score
                    result_nodes.append(node_details)
        
        # Expand to neighbors
        remaining_slots = max_nodes - len(visited_nodes)
        if remaining_slots > 0:
            for node_id, _ in entry_points:
                if remaining_slots <= 0:
                    break
                
                # Get neighbors
                neighbors = list(self.graph.neighbors(node_id))
                for neighbor_id in neighbors[:remaining_slots]:
                    if neighbor_id not in visited_nodes:
                        visited_nodes.add(neighbor_id)
                        neighbor_details = self._get_node_details(neighbor_id)
                        if neighbor_details:
                            neighbor_details['similarity_score'] = 0.5  # Default neighbor score
                            result_nodes.append(neighbor_details)
                            remaining_slots -= 1
        
        # Collect relationships between visited nodes
        for node_id in visited_nodes:
            for neighbor_id in self.graph.neighbors(node_id):
                if neighbor_id in visited_nodes:
                    rel_details = self._get_relationship_details(node_id, neighbor_id)
                    if rel_details and rel_details not in result_relationships:
                        result_relationships.append(rel_details)
        
        return result_nodes, result_relationships
    
    def _get_node_details(self, node_id: str) -> Optional[Dict]:
        """Get detailed node information"""
        return self.node_details.get(node_id)
    
    def _get_relationship_details(self, source_id: str, target_id: str) -> Optional[Dict]:
        """Get relationship details between two nodes"""
        if self.graph.has_edge(source_id, target_id):
            edge_data = self.graph.get_edge_data(source_id, target_id)
            return {
                'source': source_id,
                'target': target_id,
                'type': edge_data.get('relationship_type', 'CONNECTED'),
                'properties': {}
            }
        return None
    
    def _create_context(self, query: str, nodes: List[Dict], relationships: List[Dict]) -> str:
        """Create context string from nodes and relationships"""
        context_parts = [f"Query: {query}\n"]
        
        if nodes:
            context_parts.append("Relevant Entities:")
            for node in nodes[:10]:  # Limit for readability
                properties_str = self._format_properties(node.get('properties', {}))
                context_parts.append(f"- {node['id']} ({node['type']}): {properties_str}")
        
        if relationships:
            context_parts.append("\nRelationships:")
            for rel in relationships[:20]:  # Limit for readability
                context_parts.append(f"- {rel['source']} --[{rel['type']}]--> {rel['target']}")
        
        return "\n".join(context_parts)
    
    def _format_properties(self, properties: Dict) -> str:
        """Format node properties for display"""
        if not properties:
            return "No additional properties"
        
        formatted = []
        for key, value in list(properties.items())[:3]:  # Limit properties shown
            if isinstance(value, str) and len(value) > 50:
                value = value[:50] + "..."
            formatted.append(f"{key}: {value}")
        
        return "; ".join(formatted)


class SemanticGraphService(GraphService):
    """Main graph service implementation"""
    
    def __init__(self, config_adapter):
        self.config = config_adapter
        self.graph_data: Optional[GraphData] = None
        self.node_embeddings: Dict[str, List[float]] = {}
        
        # Initialize components
        self.embedding_service = SentenceTransformerEmbedding(self.config.EMBEDDING_MODEL)
        self.llm = OllamaChat(self.config.GRAPH_LLM_MODEL, self.config.OLLAMA_BASE_URL)
        self.transformer = LangChainGraphTransformer(self.llm)
        self.storage = FileGraphStorage(self.config.GRAPH_STORE_DIRECTORY)
        self.visualizer = PyvisGraphVisualizer(self.config.GRAPH_STORE_DIRECTORY)
        self.searcher = SemanticGraphSearcher(self.embedding_service)
        
        # Load existing data
        self._load_existing_data()
    
    def process_documents(self, documents: List[Document]) -> bool:
        """Process documents and create/update graph"""
        try:
            logger.info(f"Processing {len(documents)} documents for graph creation")
            
            # Transform documents to graph
            new_graph_data = self.transformer.transform_documents(documents)
            
            if not new_graph_data.nodes:
                logger.warning("No graph data created from documents")
                return False
            
            # Merge with existing data or replace
            if self.graph_data and self.graph_data.nodes:
                self._merge_graph_data(new_graph_data)
            else:
                self.graph_data = new_graph_data
            
            # Generate embeddings for new nodes
            self._generate_embeddings()
            
            # Save to storage
            self.storage.save(self.graph_data, self.node_embeddings)
            
            logger.info(f"Graph processing complete: {len(self.graph_data.nodes)} nodes, {len(self.graph_data.relationships)} relationships")
            return True
            
        except Exception as e:
            logger.error(f"Graph processing failed: {e}")
            return False
    
    def update_with_documents(self, documents: List[Document]) -> bool:
        """Update existing graph with new documents"""
        return self.process_documents(documents)  # Same process for updates
    
    def search(self, query: str, max_nodes: int = None) -> GraphSearchResult:
        """Search the knowledge graph"""
        if not self.has_data():
            return GraphSearchResult()
        
        max_nodes = max_nodes or settings.graph.max_graph_nodes
        
        try:
            return self.searcher.search(query, self.graph_data, self.node_embeddings, max_nodes)
        except Exception as e:
            logger.error(f"Graph search failed: {e}")
            return GraphSearchResult()
    
    def visualize(self, filename: str = "graph_visualization.html") -> str:
        """Create graph visualization"""
        if not self.has_data():
            raise GraphException("No graph data available for visualization")
        
        try:
            return self.visualizer.create_visualization(self.graph_data, filename)
        except Exception as e:
            logger.error(f"Graph visualization failed: {e}")
            raise GraphException(f"Visualization failed: {e}")
    
    def has_data(self) -> bool:
        """Check if graph has data"""
        return self.graph_data is not None and len(self.graph_data.nodes) > 0
    
    def get_stats(self) -> Dict[str, Any]:
        """Get graph statistics"""
        if not self.has_data():
            return {
                "nodes": 0,
                "relationships": 0,
                "node_types": [],
                "relationship_types": [],
                "has_data": False
            }
        
        node_types = list(set(node.type for node in self.graph_data.nodes))
        relationship_types = list(set(rel.type for rel in self.graph_data.relationships))
        
        return {
            "nodes": len(self.graph_data.nodes),
            "relationships": len(self.graph_data.relationships),
            "node_types": node_types,
            "relationship_types": relationship_types,
            "has_data": True,
            "embeddings_count": len(self.node_embeddings)
        }
    
    # Implementation of missing interface methods
    def load_graph_data(self) -> bool:
        """Load existing graph data"""
        return self._load_existing_data()
    
    def process_documents_to_graph(self, documents: List[Document]) -> bool:
        """Process documents and convert to graph format"""
        return self.process_documents(documents)
    
    def update_graph_with_documents(self, documents: List[Document]) -> bool:
        """Update graph with new document data"""
        return self.update_with_documents(documents)
    
    def has_graph_data(self) -> bool:
        """Check if graph data exists"""
        return self.has_data()
    
    # Private methods
    def _load_existing_data(self) -> bool:
        """Load existing graph data and embeddings"""
        try:
            self.graph_data, self.node_embeddings = self.storage.load()
            if self.graph_data:
                logger.info(f"Loaded existing graph: {len(self.graph_data.nodes)} nodes, {len(self.graph_data.relationships)} relationships")
                return True
            return False
        except Exception as e:
            logger.error(f"Failed to load existing graph data: {e}")
            return False
    
    def _generate_embeddings(self) -> None:
        """Generate embeddings for all nodes"""
        if not self.graph_data:
            return
        
        try:
            # Prepare texts for embedding
            node_texts = []
            node_ids = []
            
            for node in self.graph_data.nodes:
                # Create text representation of node
                text_parts = [node.id, node.type]
                
                # Add properties
                for key, value in node.properties.items():
                    if isinstance(value, str) and value.strip():
                        text_parts.append(f"{key}: {value}")
                
                node_text = " ".join(text_parts)
                node_texts.append(node_text)
                node_ids.append(node.id)
            
            # Generate embeddings
            embeddings = self.embedding_service.encode(node_texts)
            
            # Update embeddings dict
            for node_id, embedding in zip(node_ids, embeddings):
                self.node_embeddings[node_id] = embedding
            
            logger.info(f"Generated embeddings for {len(node_ids)} nodes")
            
        except Exception as e:
            logger.error(f"Failed to generate embeddings: {e}")
    
    def _merge_graph_data(self, new_graph_data: GraphData) -> None:
        """Merge new graph data with existing"""
        if not self.graph_data:
            self.graph_data = new_graph_data
            return
        
        # Add new nodes (avoiding duplicates)
        existing_node_ids = {node.id for node in self.graph_data.nodes}
        for node in new_graph_data.nodes:
            if node.id not in existing_node_ids:
                self.graph_data.add_node(node)
        
        # Add new relationships (avoiding duplicates)
        existing_rels = {(rel.source.id, rel.target.id, rel.type) for rel in self.graph_data.relationships}
        for rel in new_graph_data.relationships:
            rel_key = (rel.source.id, rel.target.id, rel.type)
            if rel_key not in existing_rels:
                # Find corresponding nodes in existing graph
                source_node = self.graph_data.get_node_by_id(rel.source.id)
                target_node = self.graph_data.get_node_by_id(rel.target.id)
                
                if source_node and target_node:
                    merged_rel = GraphRelationship(
                        source=source_node,
                        target=target_node,
                        type=rel.type,
                        properties=rel.properties
                    )
                    self.graph_data.add_relationship(merged_rel)
        
        logger.info(f"Merged graph data: total {len(self.graph_data.nodes)} nodes, {len(self.graph_data.relationships)} relationships")