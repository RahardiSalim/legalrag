import os
import json
import pickle
import logging
import numpy as np
from typing import List, Dict, Any, Optional, Set, Tuple
from pathlib import Path
from abc import ABC, abstractmethod
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

logger = logging.getLogger(__name__)


class GeminiChat(BaseChatModel):
    """Custom Gemini LLM wrapper for LangChain Graph Transformer"""
    
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
        if isinstance(input_data, dict):
            prompt = input_data.get("input", str(input_data))
        elif isinstance(input_data, str):
            prompt = input_data
        elif isinstance(input_data, list):
            prompt = "\n".join([m.content for m in input_data if isinstance(m, (HumanMessage, AIMessage))])
        else:
            prompt = str(input_data)
        
        response = self.model.generate_content(prompt)
        return AIMessage(content=response.text)

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[Any] = None,
        **kwargs: Any,
    ) -> ChatResult:
        prompt = "\n".join([
            m.content for m in messages 
            if isinstance(m, (HumanMessage, AIMessage)) and hasattr(m, 'content')
        ])
        
        try:
            response = self.model.generate_content(prompt)
            message = AIMessage(content=response.text)
            generation = ChatGeneration(message=message)
            return ChatResult(generations=[generation])
        except Exception as e:
            error_message = AIMessage(content=f"Error generating response: {str(e)}")
            generation = ChatGeneration(message=error_message)
            return ChatResult(generations=[generation])

    @property
    def _llm_type(self) -> str:
        return "gemini-chat"

    @property
    def _identifying_params(self) -> dict:
        return {"model_name": self.model_name}

    class Config:
        arbitrary_types_allowed = True


class GraphServiceInterface(ABC):
    @abstractmethod
    def process_documents_to_graph(self, documents: List[Document]) -> bool:
        pass
    
    @abstractmethod
    def search_graph(self, query: str) -> Dict[str, Any]:
        pass
    
    @abstractmethod
    def visualize_graph(self, filename: str = "graph_visualization.html") -> str:
        pass
    
    @abstractmethod
    def has_graph_data(self) -> bool:
        pass


class SemanticGraphService(GraphServiceInterface):
    """Enhanced graph service with semantic search and graph traversal"""
    
    def __init__(self, config: Config):
        self.config = config
        self.graph_documents = []
        self.combined_graph = None
        self.llm = None
        self.transformer = None
        self.semantic_model = None
        self.node_embeddings = {}
        self.nx_graph = None
        self._initialize_components()
    
    def _initialize_components(self):
        """Initialize graph processing components"""
        try:
            self.llm = GeminiChat(
                model_name=self.config.GRAPH_LLM_MODEL,
                api_key=self.config.GEMINI_API_KEY
            )
            self.transformer = LLMGraphTransformer(llm=self.llm)
            
            # Initialize semantic model for embeddings
            self.semantic_model = SentenceTransformer('all-MiniLM-L6-v2')
            
            logger.info("Enhanced graph service components initialized")
        except Exception as e:
            logger.error(f"Failed to initialize graph components: {e}")
            raise
    
    def process_documents_to_graph(self, documents: List[Document]) -> bool:
        """Process documents to create knowledge graph with semantic embeddings"""
        if not self.config.ENABLE_GRAPH_PROCESSING:
            logger.info("Graph processing disabled in config")
            return False
        
        try:
            logger.info(f"Processing {len(documents)} documents to graph...")
            
            combined_content = self._combine_documents_content(documents)
            combined_doc = Document(page_content=combined_content)
            
            self.graph_documents = self.transformer.convert_to_graph_documents([combined_doc])
            
            if not self.graph_documents or not self.graph_documents[0].nodes:
                logger.warning("No graph data generated from documents")
                return False
            
            self.combined_graph = self._create_combined_graph(self.graph_documents)
            self._build_networkx_graph()
            self._generate_node_embeddings()
            self._save_graph_data()
            
            logger.info(f"Graph processing complete: {len(self.combined_graph.nodes)} nodes, {len(self.combined_graph.relationships)} relationships")
            return True
            
        except Exception as e:
            logger.error(f"Graph processing failed: {e}")
            return False
    
    def update_graph_with_documents(self, new_documents: List[Document]) -> bool:
        """Incrementally update existing graph with new documents"""
        if not self.config.ENABLE_GRAPH_PROCESSING:
            return False
        
        try:
            logger.info(f"Incrementally updating graph with {len(new_documents)} new documents...")
            
            new_combined_content = self._combine_documents_content(new_documents)
            new_combined_doc = Document(page_content=new_combined_content)
            
            new_graph_documents = self.transformer.convert_to_graph_documents([new_combined_doc])
            
            if not new_graph_documents or not new_graph_documents[0].nodes:
                logger.warning("No new graph data generated")
                return False
            
            if self.combined_graph is None:
                self.combined_graph = self._create_combined_graph(new_graph_documents)
            else:
                self._merge_graph_data(new_graph_documents)
            
            self._build_networkx_graph()
            self._generate_node_embeddings()
            self._save_graph_data()
            
            logger.info(f"Graph updated: now has {len(self.combined_graph.nodes)} nodes, {len(self.combined_graph.relationships)} relationships")
            return True
            
        except Exception as e:
            logger.error(f"Incremental graph update failed: {e}")
            return False

    def search_graph(self, query: str, max_nodes: int = 20) -> Dict[str, Any]:
        """Enhanced graph search with semantic similarity and graph traversal"""
        if not self.has_graph_data():
            return {"context": "No graph data available", "relevant_entities": []}
        
        try:
            # Step 1: Find semantic entry points
            entry_nodes = self._find_semantic_entry_points(query, top_k=5)
            
            if not entry_nodes:
                return {"context": "No relevant entry points found", "relevant_entities": []}
            
            # Step 2: Build subgraph from entry points
            subgraph_nodes, subgraph_relationships = self._build_subgraph_from_entries(
                entry_nodes, max_nodes=max_nodes
            )
            
            # Step 3: Create rich context
            context = self._create_rich_context(query, subgraph_nodes, subgraph_relationships)
            
            return {
                "context": context,
                "relevant_entities": subgraph_nodes,
                "relevant_relationships": subgraph_relationships,
                "entry_points": [{"id": node_id, "score": score} for node_id, score in entry_nodes]
            }
            
        except Exception as e:
            logger.error(f"Enhanced graph search failed: {e}")
            return {"context": "Error searching graph", "relevant_entities": []}
    
    def _find_semantic_entry_points(self, query: str, top_k: int = 5) -> List[Tuple[str, float]]:
        """Find nodes that are semantically similar to the query"""
        if not self.node_embeddings:
            return []
        
        try:
            # Generate query embedding
            query_embedding = self.semantic_model.encode([query])
            
            # Calculate similarities with all nodes
            node_similarities = []
            for node_id, node_embedding in self.node_embeddings.items():
                similarity = cosine_similarity(query_embedding, [node_embedding])[0][0]
                node_similarities.append((node_id, float(similarity)))
            
            # Sort by similarity and return top k
            node_similarities.sort(key=lambda x: x[1], reverse=True)
            
            # Filter out very low similarity scores (< 0.3)
            filtered_similarities = [(node_id, score) for node_id, score in node_similarities if score > 0.3]
            
            return filtered_similarities[:top_k]
            
        except Exception as e:
            logger.error(f"Failed to find semantic entry points: {e}")
            return []
    
    def _build_subgraph_from_entries(self, entry_nodes: List[Tuple[str, float]], 
                                   max_nodes: int = 20) -> Tuple[List[Dict], List[Dict]]:
        """Build a subgraph by traversing from entry points"""
        if not self.nx_graph:
            return [], []
        
        visited_nodes = set()
        relevant_nodes = []
        relevant_relationships = []
        
        # Priority queue: (negative_score, node_id, depth)
        from heapq import heappush, heappop
        priority_queue = []
        
        # Add entry points to queue
        for node_id, score in entry_nodes:
            heappush(priority_queue, (-score, node_id, 0))
        
        while priority_queue and len(visited_nodes) < max_nodes:
            neg_score, current_node, depth = heappop(priority_queue)
            
            if current_node in visited_nodes:
                continue
                
            visited_nodes.add(current_node)
            
            # Find node details
            node_details = self._get_node_details(current_node)
            if node_details:
                relevant_nodes.append(node_details)
            
            # Add relationships and neighbors (limit depth to avoid explosion)
            if depth < 2:  # Max depth of 2 hops
                for neighbor in self.nx_graph.neighbors(current_node):
                    if neighbor not in visited_nodes:
                        # Add neighbor with reduced score
                        neighbor_score = -neg_score * 0.7  # Decay score for neighbors
                        heappush(priority_queue, (-neighbor_score, neighbor, depth + 1))
                    
                    # Add relationship
                    relationship = self._get_relationship_details(current_node, neighbor)
                    if relationship and relationship not in relevant_relationships:
                        relevant_relationships.append(relationship)
        
        return relevant_nodes, relevant_relationships
    
    def _create_rich_context(self, query: str, nodes: List[Dict], 
                        relationships: List[Dict]) -> str:
        """Create rich context from subgraph"""
        context_parts = []
        
        if nodes:
            context_parts.append("Entitas yang relevan dengan pertanyaan:")
            
            # Group nodes by type for better organization
            nodes_by_type = {}
            for node in nodes:
                node_type = node.get('type', 'Unknown')
                if node_type not in nodes_by_type:
                    nodes_by_type[node_type] = []
                nodes_by_type[node_type].append(node)
            
            # Present nodes by type - FIX THE INDENTATION HERE
            for node_type, type_nodes in nodes_by_type.items():
                if len(type_nodes) > 0:
                    context_parts.append(f"\n{node_type}:")
                    for node in type_nodes[:5]:  # Limit to 5 per type
                        properties_str = ""
                        if node.get('properties'):
                            key_props = list(node['properties'].keys())[:2]  # Top 2 properties
                            if key_props:
                                # Fix: Extract the f-string content to avoid nested quotes
                                prop_items = []
                                for k in key_props:
                                    prop_value = node['properties'][k]
                                    prop_items.append(f"{k}: {prop_value}")
                                properties_str = f" ({', '.join(prop_items)})"
                        context_parts.append(f"  - {node['id']}{properties_str}")
        
        if relationships:
            context_parts.append("\nHubungan antar entitas:")
            
            # Group relationships by type
            rel_by_type = {}
            for rel in relationships:
                rel_type = rel.get('type', 'RELATED')
                if rel_type not in rel_by_type:
                    rel_by_type[rel_type] = []
                rel_by_type[rel_type].append(rel)
            
            # Present relationships by type
            for rel_type, type_rels in rel_by_type.items():
                if len(type_rels) > 0:
                    context_parts.append(f"\n{rel_type}:")
                    for rel in type_rels[:5]:  # Limit to 5 per type
                        context_parts.append(f"  - {rel['source']} → {rel['target']}")
        
        if not context_parts:
            return "Tidak ditemukan informasi graph yang relevan dengan pertanyaan."
        
        return "\n".join(context_parts)
    
    def _get_node_details(self, node_id: str) -> Optional[Dict]:
        """Get detailed information about a node"""
        for node in self.combined_graph.nodes:
            if node.id == node_id:
                return {
                    "id": node.id,
                    "type": node.type,
                    "properties": getattr(node, 'properties', {})
                }
        return None
    
    def _get_relationship_details(self, source_id: str, target_id: str) -> Optional[Dict]:
        """Get relationship details between two nodes"""
        for rel in self.combined_graph.relationships:
            if ((rel.source.id == source_id and rel.target.id == target_id) or
                (rel.source.id == target_id and rel.target.id == source_id)):
                return {
                    "source": rel.source.id,
                    "target": rel.target.id,
                    "type": rel.type,
                    "properties": getattr(rel, 'properties', {})
                }
        return None
    
    def _build_networkx_graph(self):
        """Build NetworkX graph for efficient traversal"""
        if not self.combined_graph:
            return
        
        self.nx_graph = nx.Graph()
        
        # Add nodes
        for node in self.combined_graph.nodes:
            self.nx_graph.add_node(node.id, type=node.type, 
                                 properties=getattr(node, 'properties', {}))
        
        # Add edges
        for rel in self.combined_graph.relationships:
            self.nx_graph.add_edge(rel.source.id, rel.target.id, 
                                 type=rel.type, properties=getattr(rel, 'properties', {}))
        
        logger.info(f"NetworkX graph built: {len(self.nx_graph.nodes)} nodes, {len(self.nx_graph.edges)} edges")
    
    def _generate_node_embeddings(self):
        """Generate semantic embeddings for all nodes"""
        if not self.combined_graph or not self.semantic_model:
            return
        
        try:
            self.node_embeddings = {}
            
            # Prepare texts for embedding
            node_texts = []
            node_ids = []
            
            for node in self.combined_graph.nodes:
                # Create text representation of node
                text_parts = [node.id, node.type]
                
                # Add properties if available
                if hasattr(node, 'properties') and node.properties:
                    text_parts.extend([f"{k}: {str(v)}" for k, v in node.properties.items()])
                
                node_text = " ".join(text_parts)
                node_texts.append(node_text)
                node_ids.append(node.id)
            
            # Generate embeddings in batch
            if node_texts:
                embeddings = self.semantic_model.encode(node_texts)
                
                # Store embeddings
                for node_id, embedding in zip(node_ids, embeddings):
                    self.node_embeddings[node_id] = embedding
                
                logger.info(f"Generated embeddings for {len(self.node_embeddings)} nodes")
            
        except Exception as e:
            logger.error(f"Failed to generate node embeddings: {e}")
    
    def _combine_documents_content(self, documents: List[Document]) -> str:
        """Combine document contents for graph processing"""
        combined_content = []
        
        for i, doc in enumerate(documents):
            source = doc.metadata.get('source', f'Document_{i+1}')
            combined_content.append(f"=== {source} ===")
            combined_content.append(doc.page_content)
            combined_content.append("")
        
        return "\n".join(combined_content)
    
    def _create_combined_graph(self, graph_documents):
        """Create combined graph structure from graph documents"""
        class CombinedGraph:
            def __init__(self):
                self.nodes = []
                self.relationships = []
        
        combined = CombinedGraph()
        
        for graph_doc in graph_documents:
            combined.nodes.extend(graph_doc.nodes)
            combined.relationships.extend(graph_doc.relationships)
        
        # Remove duplicate nodes
        unique_nodes = {}
        for node in combined.nodes:
            node_key = f"{node.id}_{node.type}".lower()
            if node_key not in unique_nodes:
                unique_nodes[node_key] = node
        
        combined.nodes = list(unique_nodes.values())
        
        # Remove duplicate relationships
        unique_relationships = []
        seen_relationships = set()
        for rel in combined.relationships:
            rel_key = f"{rel.source.id}_{rel.type}_{rel.target.id}".lower()
            if rel_key not in seen_relationships:
                unique_relationships.append(rel)
                seen_relationships.add(rel_key)
        
        combined.relationships = unique_relationships
        
        return combined
    
    def _merge_graph_data(self, new_graph_documents):
        """Merge new graph data with existing graph"""
        if not self.combined_graph:
            # If no existing graph, treat as new creation
            self.combined_graph = self._create_combined_graph(new_graph_documents)
            return
        
        # Create temporary storage for merging
        existing_nodes = {f"{node.id}_{node.type}".lower(): node for node in self.combined_graph.nodes}
        existing_relationships = set()
        
        # Track existing relationships
        for rel in self.combined_graph.relationships:
            rel_key = f"{rel.source.id}_{rel.type}_{rel.target.id}".lower()
            existing_relationships.add(rel_key)
        
        # Track what we're adding
        nodes_added = 0
        relationships_added = 0
        nodes_updated = 0
        
        # Process new nodes
        for graph_doc in new_graph_documents:
            for node in graph_doc.nodes:
                node_key = f"{node.id}_{node.type}".lower()
                if node_key not in existing_nodes:
                    # Add new node
                    self.combined_graph.nodes.append(node)
                    existing_nodes[node_key] = node
                    nodes_added += 1
                else:
                    # Update existing node properties
                    existing_node = existing_nodes[node_key]
                    if hasattr(node, 'properties') and hasattr(existing_node, 'properties'):
                        if node.properties:  # Only update if new node has properties
                            if not existing_node.properties:
                                existing_node.properties = {}
                            existing_node.properties.update(node.properties)
                            nodes_updated += 1
            
            # Process new relationships
            for rel in graph_doc.relationships:
                rel_key = f"{rel.source.id}_{rel.type}_{rel.target.id}".lower()
                if rel_key not in existing_relationships:
                    # Check if both source and target nodes exist
                    source_key = f"{rel.source.id}_{rel.source.type}".lower()
                    target_key = f"{rel.target.id}_{rel.target.type}".lower()
                    
                    if source_key in existing_nodes and target_key in existing_nodes:
                        self.combined_graph.relationships.append(rel)
                        existing_relationships.add(rel_key)
                        relationships_added += 1
                    else:
                        logger.warning(f"Skipping relationship {rel_key} - missing nodes")
        
        logger.info(f"Graph merge complete: {nodes_added} new nodes, {nodes_updated} updated nodes, {relationships_added} new relationships")

    def visualize_graph(self, filename: str = "graph_visualization.html") -> str:
        """Create graph visualization using pyvis with better error handling"""
        if not self.has_graph_data():
            logger.warning("No graph data to visualize")
            return ""
        
        try:
            # Ensure directory exists
            graph_dir = Path(self.config.GRAPH_STORE_DIRECTORY)
            graph_dir.mkdir(parents=True, exist_ok=True)
            
            # Create pyvis network
            net = Network(height="800px", width="100%", bgcolor="#222222", font_color="white")
            
            node_colors = {
                'Person': '#ff6b6b',
                'Organization': '#4ecdc4',
                'Location': '#45b7d1',
                'Event': '#96ceb4',
                'Concept': '#feca57',
                'Document': '#ff9ff3',
                'Entity': '#a8e6cf',
                'Default': '#c7ecee'
            }
            
            added_nodes = set()
            
            # Add nodes with error handling
            for node in self.combined_graph.nodes:
                try:
                    if node.id not in added_nodes:
                        color = node_colors.get(node.type, node_colors['Default'])
                        
                        # Safely build title
                        title = f"ID: {node.id}\nType: {node.type}"
                        if hasattr(node, 'properties') and node.properties:
                            # Convert properties to string safely
                            props_str = str(node.properties)
                            if len(props_str) > 200:  # Limit length
                                props_str = props_str[:200] + "..."
                            title += f"\nProperties: {props_str}"
                        
                        # Ensure node ID is string and not too long
                        node_id = str(node.id)
                        if len(node_id) > 50:
                            node_id = node_id[:50] + "..."
                        
                        net.add_node(
                            node.id,  # Use original ID for internal reference
                            label=node_id,  # Use truncated ID for display
                            title=title,
                            color=color,
                            size=25
                        )
                        added_nodes.add(node.id)
                except Exception as e:
                    logger.warning(f"Failed to add node {node.id}: {e}")
                    continue
            
            # Add relationships with error handling
            relationship_count = 0
            for rel in self.combined_graph.relationships:
                try:
                    # Check if both nodes exist
                    if rel.source.id in added_nodes and rel.target.id in added_nodes:
                        edge_title = f"Relationship: {rel.type}"
                        if hasattr(rel, 'properties') and rel.properties:
                            props_str = str(rel.properties)
                            if len(props_str) > 100:  # Limit length
                                props_str = props_str[:100] + "..."
                            edge_title += f"\nProperties: {props_str}"
                        
                        net.add_edge(
                            rel.source.id,
                            rel.target.id,
                            label=str(rel.type)[:20],  # Limit label length
                            title=edge_title,
                            color={'color': '#848484'},
                            width=2
                        )
                        relationship_count += 1
                except Exception as e:
                    logger.warning(f"Failed to add relationship {rel.source.id} -> {rel.target.id}: {e}")
                    continue
            
            # Configure physics
            net.barnes_hut(
                gravity=-80000,
                central_gravity=0.01,
                spring_length=200,
                spring_strength=0.05,
                damping=0.09
            )
            
            # FIXED: Better file path handling and error recovery
            full_path = graph_dir / filename
            
            try:
                # Ensure the directory is writable
                test_file = graph_dir / "test_write.tmp"
                test_file.write_text("test")
                test_file.unlink()
                
                # Try to save the graph
                net.save_graph(str(full_path))
                
                # Verify file was created and has content
                if full_path.exists() and full_path.stat().st_size > 0:
                    logger.info(f"Graph visualization saved to {full_path}")
                    logger.info(f"Visualization includes {len(added_nodes)} nodes and {relationship_count} relationships")
                    return str(full_path)
                else:
                    raise Exception("File was not created or is empty")
                    
            except Exception as save_error:
                logger.error(f"Failed to save graph to {full_path}: {save_error}")
                
                # Try saving to a different location (temp directory)
                import tempfile
                temp_dir = Path(tempfile.gettempdir())
                alt_path = temp_dir / filename
                
                try:
                    net.save_graph(str(alt_path))
                    if alt_path.exists() and alt_path.stat().st_size > 0:
                        logger.info(f"Graph saved to temporary location: {alt_path}")
                        return str(alt_path)
                    else:
                        raise Exception("Alternative save location also failed")
                except Exception as alt_error:
                    logger.error(f"Failed to save to alternative path: {alt_error}")
                    
                    # Last resort: try to generate HTML content directly
                    try:
                        html_content = net.generate_html()
                        if html_content:
                            fallback_path = temp_dir / f"graph_fallback_{int(time.time())}.html"
                            fallback_path.write_text(html_content, encoding='utf-8')
                            logger.info(f"Graph saved using fallback method: {fallback_path}")
                            return str(fallback_path)
                    except Exception as fallback_error:
                        logger.error(f"Fallback save method also failed: {fallback_error}")
                        return ""
                
        except Exception as e:
            logger.error(f"Graph visualization failed: {e}")
            import traceback
            logger.error(f"Full traceback: {traceback.format_exc()}")
            return ""
    
    def has_graph_data(self) -> bool:
        """Check if graph data is available"""
        return (self.combined_graph is not None and 
                hasattr(self.combined_graph, 'nodes') and 
                len(self.combined_graph.nodes) > 0)
    
    def _save_graph_data(self):
        """Save graph data to disk"""
        try:
            graph_file = Path(self.config.GRAPH_STORE_DIRECTORY) / "graph_data.pkl"
            embeddings_file = Path(self.config.GRAPH_STORE_DIRECTORY) / "node_embeddings.pkl"
            
            # Save graph structure
            graph_data = {
                'nodes': [
                    {
                        'id': node.id,
                        'type': node.type,
                        'properties': getattr(node, 'properties', {})
                    }
                    for node in self.combined_graph.nodes
                ],
                'relationships': [
                    {
                        'source_id': rel.source.id,
                        'target_id': rel.target.id,
                        'type': rel.type,
                        'properties': getattr(rel, 'properties', {})
                    }
                    for rel in self.combined_graph.relationships
                ]
            }
            
            with open(graph_file, 'wb') as f:
                pickle.dump(graph_data, f)
            
            # Save embeddings
            if self.node_embeddings:
                with open(embeddings_file, 'wb') as f:
                    pickle.dump(self.node_embeddings, f)
            
            logger.info(f"Graph data and embeddings saved")
            
        except Exception as e:
            logger.error(f"Failed to save graph data: {e}")
    
    def load_graph_data(self) -> bool:
        """Load graph data from disk"""
        try:
            graph_file = Path(self.config.GRAPH_STORE_DIRECTORY) / "graph_data.pkl"
            embeddings_file = Path(self.config.GRAPH_STORE_DIRECTORY) / "node_embeddings.pkl"
            
            if not graph_file.exists():
                return False
            
            # Load graph structure
            with open(graph_file, 'rb') as f:
                graph_data = pickle.load(f)
            
            # Reconstruct graph
            class SimpleNode:
                def __init__(self, id, type, properties=None):
                    self.id = id
                    self.type = type
                    self.properties = properties or {}
            
            class SimpleRelationship:
                def __init__(self, source_id, target_id, type, properties=None):
                    self.source = SimpleNode(source_id, "")
                    self.target = SimpleNode(target_id, "")
                    self.type = type
                    self.properties = properties or {}
            
            class SimpleGraph:
                def __init__(self):
                    self.nodes = []
                    self.relationships = []
            
            self.combined_graph = SimpleGraph()
            
            for node_data in graph_data['nodes']:
                node = SimpleNode(
                    node_data['id'],
                    node_data['type'],
                    node_data.get('properties', {})
                )
                self.combined_graph.nodes.append(node)
            
            for rel_data in graph_data['relationships']:
                rel = SimpleRelationship(
                    rel_data['source_id'],
                    rel_data['target_id'],
                    rel_data['type'],
                    rel_data.get('properties', {})
                )
                self.combined_graph.relationships.append(rel)
            
            # Rebuild NetworkX graph
            self._build_networkx_graph()
            
            # Load embeddings
            if embeddings_file.exists():
                with open(embeddings_file, 'rb') as f:
                    self.node_embeddings = pickle.load(f)
                logger.info(f"Loaded embeddings for {len(self.node_embeddings)} nodes")
            else:
                # Regenerate embeddings if not found
                self._generate_node_embeddings()
            
            logger.info(f"Loaded graph data: {len(self.combined_graph.nodes)} nodes, {len(self.combined_graph.relationships)} relationships")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load graph data: {e}")
            return False
    
    def get_graph_stats(self) -> Dict[str, Any]:
        """Get graph statistics"""
        if not self.has_graph_data():
            return {"nodes": 0, "relationships": 0, "node_types": [], "relationship_types": []}
        
        node_types = list(set(node.type for node in self.combined_graph.nodes))
        relationship_types = list(set(rel.type for rel in self.combined_graph.relationships))
        
        return {
            "nodes": len(self.combined_graph.nodes),
            "relationships": len(self.combined_graph.relationships),
            "node_types": node_types,
            "relationship_types": relationship_types,
            "has_embeddings": len(self.node_embeddings) > 0,
            "embedding_coverage": len(self.node_embeddings) / len(self.combined_graph.nodes) if self.combined_graph.nodes else 0
        }


# For backward compatibility, alias the enhanced service
GraphService = SemanticGraphService