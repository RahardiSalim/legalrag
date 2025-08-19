from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Union
from langchain.schema import Document
from langchain_community.chat_models import ChatOllama
from langchain.embeddings.base import Embeddings


class ModelManagerInterface(ABC):
    @abstractmethod
    def get_llm(self) -> ChatOllama:
        pass
    
    @abstractmethod
    def get_embeddings(self) -> Embeddings:
        pass
    
    @abstractmethod
    def get_reranker(self) -> Any:
        pass


class DocumentProcessorInterface(ABC):
    @abstractmethod
    def process_documents(self, file_paths: List[str]) -> List[Document]:
        pass
    
    @abstractmethod
    def create_content_hash(self, content: str) -> str:
        pass


class VectorStoreInterface(ABC):
    @abstractmethod
    def create_store(self, documents: List[Document]) -> Any:
        pass
    
    @abstractmethod
    def load_store(self) -> bool:
        pass
    
    @abstractmethod
    def get_retriever(self) -> Any:
        pass
    
    @abstractmethod
    def add_documents(self, documents: List[Document]) -> bool:
        pass


class RAGServiceInterface(ABC):
    @abstractmethod
    def setup_chain(self) -> None:
        pass
    
    @abstractmethod
    def query(self, question: str, **kwargs) -> Dict[str, Any]:
        pass
    
    @abstractmethod
    def clear_memory(self) -> None:
        pass
    
    @abstractmethod
    def get_last_chunks(self) -> List[Document]:
        pass
    
    @abstractmethod
    def get_graph_stats(self) -> Dict[str, Any]:
        pass
    
    @abstractmethod
    def visualize_graph(self, filename: str) -> str:
        pass