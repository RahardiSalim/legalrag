from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field, validator
from datetime import datetime
from enum import Enum

from config.settings import settings


class MessageRole(str, Enum):
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


class SearchType(str, Enum):
    VECTOR = "vector"
    GRAPH = "graph"
    HYBRID = "hybrid"


class SourceDocument(BaseModel):
    content: str
    metadata: Dict[str, Any] = Field(default_factory=dict)
    score: Optional[float] = None
    rerank_score: Optional[float] = None
    
    @validator('content')
    def content_not_empty(cls, v):
        if not v or not v.strip():
            raise ValueError('Document content cannot be empty')
        return v.strip()


class GraphEntity(BaseModel):
    id: str
    type: str
    properties: Dict[str, Any] = Field(default_factory=dict)


class GraphRelationship(BaseModel):
    source: str
    target: str
    type: str
    properties: Dict[str, Any] = Field(default_factory=dict)


class ChatResponse(BaseModel):
    answer: str
    source_documents: List[SourceDocument] = Field(default_factory=list)
    generated_question: Optional[str] = None
    enhanced_query: bool = False
    processing_time: Optional[float] = None
    tokens_used: Optional[int] = None
    search_type_used: Optional[SearchType] = SearchType.VECTOR
    graph_entities: List[GraphEntity] = Field(default_factory=list)
    graph_relationships: List[GraphRelationship] = Field(default_factory=list)
    
    @validator('answer')
    def answer_not_empty(cls, v):
        if not v or not v.strip():
            raise ValueError('Answer cannot be empty')
        return v.strip()


class QueryRequest(BaseModel):
    question: str
    use_enhanced_query: bool = False
    search_type: SearchType = SearchType.VECTOR
    max_results: Optional[int] = Field(
        default=settings.retrieval.max_results_default, 
        ge=1, 
        le=settings.retrieval.max_results_limit
    )
    include_metadata: bool = True
    
    @validator('question')
    def question_not_empty(cls, v):
        if not v or not v.strip():
            raise ValueError('Question cannot be empty')
        return v.strip()


class UploadResponse(BaseModel):
    success: bool
    message: str
    file_count: int = 0
    chunk_count: int = 0
    graph_processed: bool = False
    graph_nodes: int = 0
    graph_relationships: int = 0
    processing_time: Optional[float] = None
    errors: List[str] = Field(default_factory=list)


class ChunkInfo(BaseModel):
    content: str
    metadata: Dict[str, Any] = Field(default_factory=dict)
    score: Optional[float] = None
    rerank_score: Optional[float] = None
    chunk_id: Optional[str] = None
    
    @validator('content')
    def content_not_empty(cls, v):
        if not v or not v.strip():
            raise ValueError('Chunk content cannot be empty')
        return v.strip()


class GraphStats(BaseModel):
    nodes: int = 0
    relationships: int = 0
    node_types: List[str] = Field(default_factory=list)
    relationship_types: List[str] = Field(default_factory=list)
    has_data: bool = False


class HealthResponse(BaseModel):
    status: str
    system_initialized: bool
    chat_history_length: int
    vector_store_status: str
    graph_store_status: str = "not_initialized"
    api_status: str
    timestamp: datetime = Field(default_factory=datetime.now)


class DocumentMetadata(BaseModel):
    source: str
    page: Optional[int] = None
    chunk_id: Optional[str] = None
    doc_type: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.now)
    file_size: Optional[int] = None
    language: Optional[str] = settings.system.default_language


class ErrorResponse(BaseModel):
    error: str
    detail: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.now)
    request_id: Optional[str] = None


class FeedbackRequest(BaseModel):
    query: str
    response: str
    relevance_score: int = Field(
        ..., 
        ge=settings.feedback.score_range_min, 
        le=settings.feedback.score_range_max, 
        description=f"Relevance score from {settings.feedback.score_range_min}-{settings.feedback.score_range_max}"
    )
    quality_score: int = Field(
        ..., 
        ge=settings.feedback.score_range_min, 
        le=settings.feedback.score_range_max, 
        description=f"Quality score from {settings.feedback.score_range_min}-{settings.feedback.score_range_max}"
    )
    search_type: Optional[str] = None
    response_time: Optional[float] = None
    comments: Optional[str] = Field(None, max_length=settings.feedback.max_comment_length)
    user_id: Optional[str] = None
    
    @validator('query', 'response')
    def not_empty(cls, v):
        if not v or not v.strip():
            raise ValueError('Field cannot be empty')
        return v.strip()


class FeedbackResponse(BaseModel):
    success: bool
    message: str
    feedback_id: Optional[str] = None


class FeedbackStatsResponse(BaseModel):
    total_feedback: int
    average_relevance: float
    average_quality: float
    search_type_distribution: Dict[str, int] = Field(default_factory=dict)
    feedback_learning_enabled: bool = True


class EnhancedChatResponse(ChatResponse):
    """Extended ChatResponse with feedback learning information"""
    feedback_learning_applied: bool = False
    feedback_entries_used: int = 0
    documents_learned: int = 0
    query_with_feedback_time: Optional[float] = None