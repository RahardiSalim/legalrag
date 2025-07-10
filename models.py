from typing import List, Optional, Dict, Any
from pydantic import BaseModel
from datetime import datetime


class ChatMessage(BaseModel):
    role: str  # "user" or "assistant"
    content: str
    timestamp: datetime = datetime.now()


class ChatResponse(BaseModel):
    answer: str
    source_documents: List[Dict[str, Any]]
    generated_question: Optional[str] = None
    enhanced_query: bool = False


class QueryRequest(BaseModel):
    question: str
    use_enhanced_query: bool = False


class UploadResponse(BaseModel):
    success: bool
    message: str
    file_count: int = 0
    chunk_count: int = 0


class ChatHistoryResponse(BaseModel):
    messages: List[ChatMessage]


class SourceDocument(BaseModel):
    content: str
    metadata: Dict[str, Any]
    
    
class ChunkInfo(BaseModel):
    content: str
    metadata: Dict[str, Any]
    score: Optional[float] = None