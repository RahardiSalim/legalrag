from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field, validator
from datetime import datetime
from enum import Enum


class MessageRole(str, Enum):
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


class ChatMessage(BaseModel):
    role: MessageRole
    content: str
    timestamp: datetime = Field(default_factory=datetime.now)
    metadata: Optional[Dict[str, Any]] = None
    
    @validator('content')
    def content_not_empty(cls, v):
        if not v or not v.strip():
            raise ValueError('Content cannot be empty')
        return v.strip()


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


class ChatResponse(BaseModel):
    answer: str
    source_documents: List[SourceDocument] = Field(default_factory=list)
    generated_question: Optional[str] = None
    enhanced_query: bool = False
    processing_time: Optional[float] = None
    tokens_used: Optional[int] = None
    
    @validator('answer')
    def answer_not_empty(cls, v):
        if not v or not v.strip():
            raise ValueError('Answer cannot be empty')
        return v.strip()


class QueryRequest(BaseModel):
    question: str
    use_enhanced_query: bool = False
    max_results: Optional[int] = Field(default=5, ge=1, le=20)
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
    processing_time: Optional[float] = None
    errors: List[str] = Field(default_factory=list)


class ChatHistoryResponse(BaseModel):
    messages: List[ChatMessage]
    total_messages: int
    
    @validator('total_messages')
    def total_messages_matches(cls, v, values):
        if 'messages' in values:
            actual_count = len(values['messages'])
            if v != actual_count:
                raise ValueError(f'total_messages ({v}) does not match actual count ({actual_count})')
        return v


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


class HealthResponse(BaseModel):
    status: str
    system_initialized: bool
    chat_history_length: int
    vector_store_status: str
    api_status: str
    timestamp: datetime = Field(default_factory=datetime.now)


class DocumentMetadata(BaseModel):
    source: str
    page: Optional[int] = None
    chunk_id: Optional[str] = None
    doc_type: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.now)
    file_size: Optional[int] = None
    language: Optional[str] = "id"  # Indonesian by default


class ErrorResponse(BaseModel):
    error: str
    detail: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.now)
    request_id: Optional[str] = None