from typing import List, Optional
from datetime import datetime

from models.chat_models import ChatMessage
from config.settings import settings


class ApplicationState:
    """Centralized application state management"""
    
    def __init__(self):
        self.chat_history: List[ChatMessage] = []
        self.system_initialized: bool = False
        self.graph_initialized: bool = False
        self.last_upload_time: Optional[datetime] = None
        self.document_count: int = 0
        self.chunk_count: int = 0
        self._max_chat_history = settings.memory.conversation_window_size * 2  # Keep more history than memory window
    
    def add_user_message(self, content: str):
        """Add a user message to chat history"""
        message = ChatMessage(
            role="user",
            content=content,
            timestamp=datetime.now()
        )
        self._add_message(message)
    
    def add_assistant_message(self, content: str):
        """Add an assistant message to chat history"""
        message = ChatMessage(
            role="assistant",
            content=content,
            timestamp=datetime.now()
        )
        self._add_message(message)
    
    def _add_message(self, message: ChatMessage):
        """Add message and maintain history size limit"""
        self.chat_history.append(message)
        
        # Trim history if it exceeds maximum size
        if len(self.chat_history) > self._max_chat_history:
            self.chat_history = self.chat_history[-self._max_chat_history:]
    
    def clear_history(self):
        """Clear chat history"""
        self.chat_history.clear()
    
    def update_upload_stats(self, file_count: int, chunk_count: int):
        """Update upload statistics"""
        self.last_upload_time = datetime.now()
        self.document_count += file_count
        self.chunk_count += chunk_count
    
    def get_recent_history(self, limit: int = None) -> List[ChatMessage]:
        """Get recent chat history"""
        if limit is None:
            limit = settings.memory.conversation_window_size
        
        return self.chat_history[-limit:] if self.chat_history else []
    
    def get_system_stats(self) -> dict:
        """Get system statistics"""
        return {
            "system_initialized": self.system_initialized,
            "graph_initialized": self.graph_initialized,
            "total_documents": self.document_count,
            "total_chunks": self.chunk_count,
            "chat_messages": len(self.chat_history),
            "last_upload": self.last_upload_time.isoformat() if self.last_upload_time else None
        }