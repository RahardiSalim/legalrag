from typing import List, Dict, Any, Optional, Set
from pydantic import BaseModel, Field
from datetime import datetime
from enum import Enum
import uuid


class NodeType(str, Enum):
    ENTITY = "entity"
    CONCEPT = "concept"
    TOPIC = "topic"
    REGULATION = "regulation"


class EdgeType(str, Enum):
    RELATES_TO = "relates_to"
    DEFINES = "defines"
    REFERENCES = "references"
    CONTAINS = "contains"
    MODIFIES = "modifies"
    SUPERSEDES = "supersedes"


class GraphNode(BaseModel):
    """Represents a node in the knowledge graph"""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str
    type: NodeType
    description: Optional[str] = None
    attributes: Dict[str, Any] = Field(default_factory=dict)
    chunk_ids: Set[str] = Field(default_factory=set)  # Source chunks
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)
    
    class Config:
        use_enum_values = True


class GraphEdge(BaseModel):
    """Represents an edge in the knowledge graph"""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    source_node_id: str
    target_node_id: str
    edge_type: EdgeType
    weight: float = 1.0
    description: Optional[str] = None
    chunk_ids: Set[str] = Field(default_factory=set)  # Source chunks
    created_at: datetime = Field(default_factory=datetime.now)
    
    class Config:
        use_enum_values = True


class GraphExtractionResult(BaseModel):
    """Result of graph extraction from chunks"""
    nodes: List[GraphNode]
    edges: List[GraphEdge]
    processed_chunks: List[str]
    extraction_time: float
    api_calls_made: int


class GraphQueryResult(BaseModel):
    """Result of graph-based query"""
    answer: str
    relevant_nodes: List[GraphNode]
    relevant_edges: List[GraphEdge]
    traversal_path: List[str]
    source_chunks: Set[str]
    processing_time: float