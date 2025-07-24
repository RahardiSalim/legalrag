import os
import json
import pickle
import logging
from typing import List, Dict, Any, Optional
from pathlib import Path
from abc import ABC, abstractmethod

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


class GraphService(GraphServiceInterface):
    """Service for processing documents into knowledge graphs"""
    
    def __init__(self, config: Config):
        self.config = config
        self.graph_documents = []
        self.combined_graph = None
        self.llm = None
        self.transformer = None
        self._initialize_components()
    
    def _initialize_components(self):
        """Initialize graph processing components"""
        try:
            self.llm = GeminiChat(
                model_name=self.config.GRAPH_LLM_MODEL,
                api_key=self.config.GEMINI_API_KEY
            )
            self.transformer = LLMGraphTransformer(llm=self.llm)
            logger.info("Graph service components initialized")
        except Exception as e:
            logger.error(f"Failed to initialize graph components: {e}")
            raise
    
    def process_documents_to_graph(self, documents: List[Document]) -> bool:
        """Process documents to create knowledge graph"""
        if not self.config.ENABLE_GRAPH_PROCESSING:
            logger.info("Graph processing disabled in config")
            return False
        
        try:
            logger.info(f"Processing {len(documents)} documents to graph...")
            
            # Combine document content for graph processing
            combined_content = self._combine_documents_content(documents)
            combined_doc = Document(page_content=combined_content)
            
            # Convert to graph
            self.graph_documents = self.transformer.convert_to_graph_documents([combined_doc])
            
            if not self.graph_documents or not self.graph_documents[0].nodes:
                logger.warning("No graph data generated from documents")
                return False
            
            # Create combined graph structure
            self.combined_graph = self._create_combined_graph(self.graph_documents)
            
            # Persist graph data
            self._save_graph_data()
            
            logger.info(f"Graph processing complete: {len(self.combined_graph.nodes)} nodes, {len(self.combined_graph.relationships)} relationships")
            return True
            
        except Exception as e:
            logger.error(f"Graph processing failed: {e}")
            return False
    
    def _combine_documents_content(self, documents: List[Document]) -> str:
        """Combine document contents for graph processing"""
        combined_content = []
        
        for i, doc in enumerate(documents):
            # Add document separator and metadata
            source = doc.metadata.get('source', f'Document_{i+1}')
            combined_content.append(f"=== {source} ===")
            combined_content.append(doc.page_content)
            combined_content.append("")  # Empty line separator
        
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
        
        # Remove duplicate nodes and relationships
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
    
    def update_graph_with_documents(self, new_documents: List[Document]) -> bool:
        """Incrementally update existing graph with new documents"""
        if not self.config.ENABLE_GRAPH_PROCESSING:
            return False
        
        try:
            logger.info(f"Incrementally updating graph with {len(new_documents)} new documents...")
            
            # Process new documents to graph
            new_combined_content = self._combine_documents_content(new_documents)
            new_combined_doc = Document(page_content=new_combined_content)
            
            new_graph_documents = self.transformer.convert_to_graph_documents([new_combined_doc])
            
            if not new_graph_documents or not new_graph_documents[0].nodes:
                logger.warning("No new graph data generated")
                return False
            
            # Merge with existing graph
            if self.combined_graph is None:
                # No existing graph, create new one
                self.combined_graph = self._create_combined_graph(new_graph_documents)
            else:
                # Merge with existing graph
                self._merge_graph_data(new_graph_documents)
            
            # Save updated graph
            self._save_graph_data()
            
            logger.info(f"Graph updated: now has {len(self.combined_graph.nodes)} nodes, {len(self.combined_graph.relationships)} relationships")
            return True
            
        except Exception as e:
            logger.error(f"Incremental graph update failed: {e}")
            return False

    def _merge_graph_data(self, new_graph_documents):
        """Merge new graph data with existing graph"""
        # Create a temporary combined structure
        temp_nodes = {}
        temp_relationships = []
        
        # Add existing nodes
        for node in self.combined_graph.nodes:
            node_key = f"{node.id}_{node.type}".lower()
            temp_nodes[node_key] = node
        
        # Add existing relationships
        temp_relationships.extend(self.combined_graph.relationships)
        
        # Add new nodes (avoiding duplicates)
        for graph_doc in new_graph_documents:
            for node in graph_doc.nodes:
                node_key = f"{node.id}_{node.type}".lower()
                if node_key not in temp_nodes:
                    temp_nodes[node_key] = node
                else:
                    # Merge properties if node exists
                    existing_node = temp_nodes[node_key]
                    if hasattr(node, 'properties') and hasattr(existing_node, 'properties'):
                        existing_node.properties.update(node.properties or {})
            
            # Add new relationships
            temp_relationships.extend(graph_doc.relationships)
        
        # Remove duplicate relationships
        unique_relationships = []
        seen_relationships = set()
        for rel in temp_relationships:
            rel_key = f"{rel.source.id}_{rel.type}_{rel.target.id}".lower()
            if rel_key not in seen_relationships:
                unique_relationships.append(rel)
                seen_relationships.add(rel_key)
        
        # Update combined graph
        self.combined_graph.nodes = list(temp_nodes.values())
        self.combined_graph.relationships = unique_relationships

    def search_graph(self, query: str) -> Dict[str, Any]:
        """Search graph for relevant nodes and relationships"""
        if not self.has_graph_data():
            return {"context": "No graph data available", "relevant_entities": []}
        
        try:
            query_lower = query.lower()
            relevant_nodes = []
            relevant_relationships = []
            
            # Find relevant nodes
            for node in self.combined_graph.nodes:
                if query_lower in node.id.lower() or query_lower in node.type.lower():
                    relevant_nodes.append({
                        "id": node.id,
                        "type": node.type,
                        "properties": getattr(node, 'properties', {})
                    })
            
            # Find relevant relationships
            for rel in self.combined_graph.relationships:
                if (query_lower in rel.type.lower() or 
                    query_lower in rel.source.id.lower() or 
                    query_lower in rel.target.id.lower()):
                    relevant_relationships.append({
                        "source": rel.source.id,
                        "target": rel.target.id,
                        "type": rel.type,
                        "properties": getattr(rel, 'properties', {})
                    })
            
            # Create context string
            context_parts = []
            
            if relevant_nodes:
                context_parts.append("Entitas yang relevan:")
                for node in relevant_nodes[:10]:  # Limit to top 10
                    context_parts.append(f"- {node['id']} (tipe: {node['type']})")
            
            if relevant_relationships:
                context_parts.append("\nHubungan yang relevan:")
                for rel in relevant_relationships[:10]:  # Limit to top 10
                    context_parts.append(f"- {rel['source']} --[{rel['type']}]--> {rel['target']}")
            
            context = "\n".join(context_parts) if context_parts else "Tidak ditemukan informasi graph yang relevan"
            
            return {
                "context": context,
                "relevant_entities": relevant_nodes,
                "relevant_relationships": relevant_relationships
            }
            
        except Exception as e:
            logger.error(f"Graph search failed: {e}")
            return {"context": "Error searching graph", "relevant_entities": []}
    

    
    def visualize_graph(self, filename: str = "graph_visualization.html") -> str:
        """Create graph visualization using pyvis"""
        if not self.has_graph_data():
            logger.warning("No graph data to visualize")
            return ""
        
        try:
            net = Network(height="800px", width="100%", bgcolor="#222222", font_color="white")
            
            # Node colors by type
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
            
            # Add nodes
            for node in self.combined_graph.nodes:
                if node.id not in added_nodes:
                    color = node_colors.get(node.type, node_colors['Default'])
                    
                    title = f"ID: {node.id}\nType: {node.type}"
                    if hasattr(node, 'properties') and node.properties:
                        title += f"\nProperties: {node.properties}"
                    
                    net.add_node(
                        node.id,
                        label=node.id,
                        title=title,
                        color=color,
                        size=25
                    )
                    added_nodes.add(node.id)
            
            # Add relationships
            for rel in self.combined_graph.relationships:
                edge_title = f"Relationship: {rel.type}"
                if hasattr(rel, 'properties') and rel.properties:
                    edge_title += f"\nProperties: {rel.properties}"
                
                net.add_edge(
                    rel.source.id,
                    rel.target.id,
                    label=rel.type,
                    title=edge_title,
                    color={'color': '#848484'},
                    width=2
                )
            
            net.barnes_hut()
            
            # Save to graph store directory
            full_path = Path(self.config.GRAPH_STORE_DIRECTORY) / filename
            net.save_graph(str(full_path))
            
            logger.info(f"Graph visualization saved to {full_path}")
            return str(full_path)
            
        except Exception as e:
            logger.error(f"Graph visualization failed: {e}")
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
            
            # Create serializable graph data
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
            
            logger.info(f"Graph data saved to {graph_file}")
            
        except Exception as e:
            logger.error(f"Failed to save graph data: {e}")
    
    def load_graph_data(self) -> bool:
        """Load graph data from disk"""
        try:
            graph_file = Path(self.config.GRAPH_STORE_DIRECTORY) / "graph_data.pkl"
            
            if not graph_file.exists():
                return False
            
            with open(graph_file, 'rb') as f:
                graph_data = pickle.load(f)
            
            # Reconstruct graph structure (simplified for search purposes)
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
            
            # Reconstruct nodes
            for node_data in graph_data['nodes']:
                node = SimpleNode(
                    node_data['id'],
                    node_data['type'],
                    node_data.get('properties', {})
                )
                self.combined_graph.nodes.append(node)
            
            # Reconstruct relationships
            for rel_data in graph_data['relationships']:
                rel = SimpleRelationship(
                    rel_data['source_id'],
                    rel_data['target_id'],
                    rel_data['type'],
                    rel_data.get('properties', {})
                )
                self.combined_graph.relationships.append(rel)
            
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
            "relationship_types": relationship_types
        }
    