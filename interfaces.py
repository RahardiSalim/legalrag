from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from langchain.schema import Document
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from sentence_transformers import CrossEncoder

class ModelManagerInterface(ABC):
    @abstractmethod
    def get_llm(self) -> ChatGoogleGenerativeAI:
        pass
    
    @abstractmethod
    def get_embeddings(self) -> GoogleGenerativeAIEmbeddings:
        pass
    
    @abstractmethod
    def get_reranker(self) -> CrossEncoder:
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

class GraphServiceInterface(ABC):
    @abstractmethod
    def process_documents_to_graph(self, documents: List[Document]) -> bool:
        pass
    
    @abstractmethod
    def search_graph(self, query: str) -> Dict[str, Any]:
        pass
    
    @abstractmethod
    def has_graph_data(self) -> bool:
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