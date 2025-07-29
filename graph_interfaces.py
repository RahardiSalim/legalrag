from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Tuple
from langchain_core.documents import Document


class EmbeddingService(ABC):
    @abstractmethod
    def encode(self, texts: List[str]) -> List[List[float]]:
        pass


class GraphNode:
    def __init__(self, node_id: str, node_type: str, properties: Dict[str, Any] = None):
        self.id = node_id
        self.type = node_type
        self.properties = properties or {}


class GraphRelationship:
    def __init__(self, source: GraphNode, target: GraphNode, rel_type: str, properties: Dict[str, Any] = None):
        self.source = source
        self.target = target
        self.type = rel_type
        self.properties = properties or {}


class GraphData:
    def __init__(self):
        self.nodes: List[GraphNode] = []
        self.relationships: List[GraphRelationship] = []


class GraphSearchResult:
    def __init__(self, context: str, entities: List[Dict], relationships: List[Dict], entry_points: List[Tuple[str, float]]):
        self.context = context
        self.entities = entities
        self.relationships = relationships
        self.entry_points = entry_points


class GraphTransformer(ABC):
    @abstractmethod
    def transform_documents(self, documents: List[Document]) -> GraphData:
        pass


class GraphStorage(ABC):
    @abstractmethod
    def save(self, graph_data: GraphData, embeddings: Dict[str, List[float]]) -> None:
        pass
    
    @abstractmethod
    def load(self) -> Tuple[Optional[GraphData], Dict[str, List[float]]]:
        pass


class GraphVisualizer(ABC):
    @abstractmethod
    def create_visualization(self, graph_data: GraphData, filename: str) -> str:
        pass


class GraphSearcher(ABC):
    @abstractmethod
    def search(self, query: str, graph_data: GraphData, embeddings: Dict[str, List[float]], max_nodes: int) -> GraphSearchResult:
        pass


class GraphService(ABC):
    @abstractmethod
    def process_documents(self, documents: List[Document]) -> bool:
        pass
    
    @abstractmethod
    def update_with_documents(self, documents: List[Document]) -> bool:
        pass
    
    @abstractmethod
    def search(self, query: str, max_nodes: int = 20) -> GraphSearchResult:
        pass
    
    @abstractmethod
    def visualize(self, filename: str) -> str:
        pass
    
    @abstractmethod
    def has_data(self) -> bool:
        pass
    
    @abstractmethod
    def get_stats(self) -> Dict[str, Any]:
        pass