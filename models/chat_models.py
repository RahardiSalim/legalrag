from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field, validator
from datetime import datetime

from config.settings import settings
from models.api_models import MessageRole


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