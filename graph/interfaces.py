from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from langchain.schema import Document


@dataclass
class GraphNode:
    """Represents a node in the knowledge graph"""
    id: str
    type: str
    properties: Dict[str, Any]
    
    def __hash__(self):
        return hash(self.id)
    
    def __eq__(self, other):
        return isinstance(other, GraphNode) and self.id == other.id


@dataclass
class GraphRelationship:
    """Represents a relationship between two nodes"""
    source: GraphNode
    target: GraphNode
    type: str
    properties: Dict[str, Any]
    
    def __hash__(self):
        return hash((self.source.id, self.target.id, self.type))
    
    def __eq__(self, other):
        return (isinstance(other, GraphRelationship) and 
                self.source.id == other.source.id and 
                self.target.id == other.target.id and 
                self.type == other.type)


@dataclass
class GraphData:
    """Container for graph nodes and relationships"""
    nodes: List[GraphNode]
    relationships: List[GraphRelationship]
    
    def __init__(self):
        self.nodes = []
        self.relationships = []
    
    def add_node(self, node: GraphNode):
        """Add a node if it doesn't already exist"""
        if node not in self.nodes:
            self.nodes.append(node)
    
    def add_relationship(self, relationship: GraphRelationship):
        """Add a relationship if it doesn't already exist"""
        if relationship not in self.relationships:
            self.relationships.append(relationship)
    
    def get_node_by_id(self, node_id: str) -> Optional[GraphNode]:
        """Get node by ID"""
        for node in self.nodes:
            if node.id == node_id:
                return node
        return None
    
    def get_relationships_for_node(self, node_id: str) -> List[GraphRelationship]:
        """Get all relationships involving a specific node"""
        return [rel for rel in self.relationships 
                if rel.source.id == node_id or rel.target.id == node_id]


@dataclass
class GraphSearchResult:
    """Result from graph search operations"""
    entities: List[Dict[str, Any]]
    relationships: List[Dict[str, Any]]
    context: str
    scores: Dict[str, float]
    
    def __init__(self):
        self.entities = []
        self.relationships = []
        self.context = ""
        self.scores = {}


class EmbeddingService(ABC):
    """Abstract service for generating embeddings"""
    
    @abstractmethod
    def encode(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for a list of texts"""
        pass


class GraphTransformer(ABC):
    """Abstract service for transforming documents to graph format"""
    
    @abstractmethod
    def transform_documents(self, documents: List[Document]) -> GraphData:
        """Transform documents into graph data"""
        pass


class GraphStorage(ABC):
    """Abstract service for graph persistence"""
    
    @abstractmethod
    def save(self, graph_data: GraphData, embeddings: Dict[str, List[float]]) -> None:
        """Save graph data and embeddings to storage"""
        pass
    
    @abstractmethod
    def load(self) -> Tuple[Optional[GraphData], Dict[str, List[float]]]:
        """Load graph data and embeddings from storage"""
        pass


class GraphVisualizer(ABC):
    """Abstract service for graph visualization"""
    
    @abstractmethod
    def create_visualization(self, graph_data: GraphData, filename: str) -> str:
        """Create a visualization of the graph"""
        pass


class GraphSearcher(ABC):
    """Abstract service for searching within the graph"""
    
    @abstractmethod
    def search(self, query: str, graph_data: GraphData, 
              embeddings: Dict[str, List[float]], max_nodes: int) -> GraphSearchResult:
        """Search the graph for relevant information"""
        pass


class GraphService(ABC):
    """Main graph service interface"""
    
    @abstractmethod
    def process_documents(self, documents: List[Document]) -> bool:
        """Process documents and create/update the knowledge graph"""
        pass
    
    @abstractmethod
    def update_with_documents(self, documents: List[Document]) -> bool:
        """Update existing graph with new documents"""
        pass
    
    @abstractmethod
    def search(self, query: str, max_nodes: int = 20) -> GraphSearchResult:
        """Search the knowledge graph"""
        pass
    
    @abstractmethod
    def visualize(self, filename: str = "graph_visualization.html") -> str:
        """Create a visualization of the graph"""
        pass
    
    @abstractmethod
    def has_data(self) -> bool:
        """Check if graph has data"""
        pass
    
    @abstractmethod
    def get_stats(self) -> Dict[str, Any]:
        """Get graph statistics"""
        pass
    
    @abstractmethod
    def load_graph_data(self) -> bool:
        """Load existing graph data"""
        pass
    
    @abstractmethod
    def process_documents_to_graph(self, documents: List[Document]) -> bool:
        """Process documents and convert to graph format"""
        pass
    
    @abstractmethod
    def update_graph_with_documents(self, documents: List[Document]) -> bool:
        """Update graph with new document data"""
        pass
    
    @abstractmethod
    def has_graph_data(self) -> bool:
        """Check if graph data exists"""
        pass


# Additional helper interfaces for graph operations
class GraphAnalyzer(ABC):
    """Service for analyzing graph structure and properties"""
    
    @abstractmethod
    def get_central_nodes(self, graph_data: GraphData, limit: int = 10) -> List[GraphNode]:
        """Get most central/important nodes in the graph"""
        pass
    
    @abstractmethod
    def find_shortest_path(self, graph_data: GraphData, source_id: str, target_id: str) -> List[str]:
        """Find shortest path between two nodes"""
        pass
    
    @abstractmethod
    def get_node_neighbors(self, graph_data: GraphData, node_id: str, depth: int = 1) -> List[GraphNode]:
        """Get neighboring nodes within specified depth"""
        pass


class GraphQueryProcessor(ABC):
    """Service for processing natural language queries against the graph"""
    
    @abstractmethod
    def extract_entities_from_query(self, query: str) -> List[str]:
        """Extract potential entity mentions from a query"""
        pass
    
    @abstractmethod
    def find_relevant_subgraph(self, query: str, graph_data: GraphData, 
                              max_nodes: int = 20) -> GraphData:
        """Find subgraph most relevant to the query"""
        pass


class GraphMetrics(ABC):
    """Service for calculating graph metrics and statistics"""
    
    @abstractmethod
    def calculate_node_centrality(self, graph_data: GraphData) -> Dict[str, float]:
        """Calculate centrality scores for all nodes"""
        pass
    
    @abstractmethod
    def get_degree_distribution(self, graph_data: GraphData) -> Dict[int, int]:
        """Get degree distribution of the graph"""
        pass
    
    @abstractmethod
    def calculate_clustering_coefficient(self, graph_data: GraphData) -> float:
        """Calculate overall clustering coefficient"""
        pass


# Exception classes for graph operations
class GraphException(Exception):
    """Base exception for graph operations"""
    pass


class GraphTransformationException(GraphException):
    """Exception raised during document-to-graph transformation"""
    pass


class GraphSearchException(GraphException):
    """Exception raised during graph search operations"""
    pass


class GraphStorageException(GraphException):
    """Exception raised during graph storage operations"""
    pass


class GraphVisualizationException(GraphException):
    """Exception raised during graph visualization"""
    pass