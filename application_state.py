from typing import List, Optional
from datetime import datetime
from models import ChatMessage

class ApplicationState:
    def __init__(self):
        self.chat_history: List[ChatMessage] = []
        self.system_initialized: bool = False
        self.graph_initialized: bool = False
        self.last_upload_time: Optional[datetime] = None
        self.document_count: int = 0
        self.chunk_count: int = 0
    
    def add_user_message(self, content: str):
        message = ChatMessage(
            role="user",
            content=content,
            timestamp=datetime.now()
        )
        self.chat_history.append(message)
    
    def add_assistant_message(self, content: str):
        message = ChatMessage(
            role="assistant",
            content=content,
            timestamp=datetime.now()
        )
        self.chat_history.append(message)
    
    def clear_history(self):
        self.chat_history.clear()
    
    def update_upload_stats(self, file_count: int, chunk_count: int):
        self.last_upload_time = datetime.now()
        self.document_count += file_count
        self.chunk_count += chunk_count
