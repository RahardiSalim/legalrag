import os
import hashlib
import logging
import tempfile
import time
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
from contextlib import contextmanager

from langchain.schema import Document
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.retrievers import BM25Retriever, EnsembleRetriever
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors.base import BaseDocumentCompressor
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferWindowMemory
from langchain.prompts import PromptTemplate
from langchain.callbacks.manager import Callbacks
from sentence_transformers import CrossEncoder
from pydantic import BaseModel, Field
import re

from config import Config
from models import SourceDocument, DocumentMetadata

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ServiceException(Exception):
    """Base exception for service errors"""
    pass


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


class ModelManager(ModelManagerInterface):
    """Manages AI models with lazy loading and caching"""
    
    def __init__(self, config: Config):
        self.config = config
        self._llm = None
        self._embeddings = None
        self._reranker = None
        
    def get_llm(self) -> ChatGoogleGenerativeAI:
        if self._llm is None:
            try:
                self._llm = ChatGoogleGenerativeAI(
                    model=self.config.LLM_MODEL,
                    google_api_key=self.config.GEMINI_API_KEY,
                    safety_settings=self.config.SAFETY_SETTINGS,
                    temperature=self.config.LLM_TEMPERATURE,
                    max_output_tokens=4096
                )
                logger.info(f"Initialized LLM: {self.config.LLM_MODEL}")
            except Exception as e:
                logger.error(f"Failed to initialize LLM: {e}")
                raise ServiceException(f"Failed to initialize LLM: {e}")
        return self._llm
    
    def get_embeddings(self) -> GoogleGenerativeAIEmbeddings:
        if self._embeddings is None:
            try:
                self._embeddings = GoogleGenerativeAIEmbeddings(
                    model=self.config.EMBEDDING_MODEL,
                    google_api_key=self.config.GEMINI_API_KEY
                )
                logger.info(f"Initialized embeddings: {self.config.EMBEDDING_MODEL}")
            except Exception as e:
                logger.error(f"Failed to initialize embeddings: {e}")
                raise ServiceException(f"Failed to initialize embeddings: {e}")
        return self._embeddings
    
    def get_reranker(self) -> CrossEncoder:
        if self._reranker is None:
            try:
                self._reranker = CrossEncoder(self.config.RERANKER_MODEL)
                logger.info(f"Initialized reranker: {self.config.RERANKER_MODEL}")
            except Exception as e:
                logger.error(f"Failed to initialize reranker: {e}")
                raise ServiceException(f"Failed to initialize reranker: {e}")
        return self._reranker


class DocumentProcessorInterface(ABC):
    @abstractmethod
    def process_documents(self, file_paths: List[str]) -> List[Document]:
        pass
    
    @abstractmethod
    def create_content_hash(self, content: str) -> str:
        pass


class DocumentProcessor(DocumentProcessorInterface):
    """Processes documents with improved error handling and duplicate detection"""
    
    def __init__(self, config: Config):
        self.config = config
        self.processed_hashes = set()
        self._text_splitter = None
        
    @property
    def text_splitter(self) -> RecursiveCharacterTextSplitter:
        if self._text_splitter is None:
            self._text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.config.CHUNK_SIZE,
                chunk_overlap=self.config.CHUNK_OVERLAP,
                length_function=len,
                separators=["\n\n\n", "\n\n", "\n", ".", "?", "!", " ", ""]
            )
        return self._text_splitter
    
    def process_documents(self, file_paths: List[str]) -> List[Document]:
        """Process multiple document files with comprehensive error handling"""
        all_documents = []
        processing_errors = []
        
        logger.info(f"Processing {len(file_paths)} files...")
        
        for file_path in file_paths:
            try:
                documents = self._load_single_document(file_path)
                if documents:
                    all_documents.extend(documents)
                    logger.info(f"Loaded {len(documents)} documents from {file_path}")
                else:
                    logger.warning(f"No documents loaded from {file_path}")
                    
            except Exception as e:
                error_msg = f"Error loading {file_path}: {str(e)}"
                logger.error(error_msg)
                processing_errors.append(error_msg)
        
        if not all_documents:
            raise ServiceException("No documents could be loaded successfully")
        
        chunks = self.text_splitter.split_documents(all_documents)
        logger.info(f"Created {len(chunks)} chunks from {len(all_documents)} documents")
        
        unique_chunks = self._process_chunks(chunks)
        logger.info(f"After deduplication: {len(unique_chunks)} unique chunks")
        
        return unique_chunks
    
    def _load_single_document(self, file_path: str) -> List[Document]:
        """Load a single document with proper error handling"""
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        if file_path.stat().st_size > self.config.MAX_DOCUMENT_SIZE:
            raise ValueError(f"File too large: {file_path} ({file_path.stat().st_size} bytes)")
        
        file_extension = file_path.suffix.lower()
        
        if file_extension == '.pdf':
            loader = PyPDFLoader(str(file_path))
        elif file_extension == '.txt':
            loader = TextLoader(str(file_path), encoding='utf-8')
        else:
            raise ValueError(f"Unsupported file type: {file_extension}")
        
        documents = loader.load()

        for doc in documents:
            doc.metadata.update({
                'source': str(file_path),
                'file_type': file_extension,
                'file_size': file_path.stat().st_size,
                'processed_at': time.time()
            })
        
        return documents
    
    def _process_chunks(self, chunks: List[Document]) -> List[Document]:
        """Process chunks with deduplication and metadata enhancement"""
        unique_chunks = []
        
        for i, chunk in enumerate(chunks):

            content = self._clean_content(chunk.page_content)
            
            if not content or len(content.strip()) < 50: 
                continue
            
            content_hash = self.create_content_hash(content)
            
            if content_hash not in self.processed_hashes:
                self.processed_hashes.add(content_hash)
                
                chunk.page_content = content
                chunk.metadata.update({
                    'chunk_id': f"chunk_{i}_{content_hash[:8]}",
                    'content_hash': content_hash,
                    'chunk_length': len(content),
                    'chunk_index': i
                })
                
                unique_chunks.append(chunk)
        
        return unique_chunks
    
    def _clean_content(self, content: str) -> str:
        """Clean and normalize text content"""

        content = re.sub(r'\s+', ' ', content)
        
        content = re.sub(r'[^\w\s\.\,\;\:\!\?\-\(\)\[\]\{\}\"\'\/\\]', '', content)
        
        return content.strip()
    
    def create_content_hash(self, content: str) -> str:
        """Create a hash for content deduplication"""
        normalized_content = re.sub(r'\s+', ' ', content.strip().lower())
        return hashlib.sha256(normalized_content.encode('utf-8')).hexdigest()


class CustomReranker(BaseDocumentCompressor):
    """Enhanced reranker with proper scoring and error handling"""

    model_name: str = "cross-encoder/ms-marco-MiniLM-L-12-v2"
    top_n: int = 10

    _cross_encoder: Optional[CrossEncoder] = None

    def __init__(self, **kwargs: Any):
        """
        Initialize the reranker.
        The __init__ now uses **kwargs to be compatible with Pydantic's initialization.
        """
        super().__init__(**kwargs)

        self._initialize_model()

    def _initialize_model(self):
        """Initialize the CrossEncoder model using the model_name field."""
        if self._cross_encoder is None:
            try:
                self._cross_encoder = CrossEncoder(self.model_name)
                logger.info(f"CustomReranker initialized with model: {self.model_name}")
            except Exception as e:
                logger.error(f"Failed to initialize CustomReranker: {e}")
                raise ServiceException(f"Failed to initialize reranker: {e}")

    @property
    def cross_encoder(self) -> CrossEncoder:
        """Lazy loading of the cross encoder model."""
        if self._cross_encoder is None:
            self._initialize_model()
        return self._cross_encoder
        
    def compress_documents(
        self,
        documents: List[Document],
        query: str,
        callbacks: Optional[Callbacks] = None,
    ) -> List[Document]:
        """Rerank documents based on query relevance."""
        if not documents or not query or self.cross_encoder is None:
            return []
        
        try:
            doc_pairs = [[query, doc.page_content] for doc in documents]
            scores = self.cross_encoder.predict(doc_pairs)
            
            for doc, score in zip(documents, scores):
                doc.metadata["rerank_score"] = float(score)
            
            sorted_docs = sorted(documents, key=lambda x: x.metadata["rerank_score"], reverse=True)
            
            logger.info(f"Reranked {len(documents)} documents, returning top {self.top_n}")
            return sorted_docs[:self.top_n]
            
        except Exception as e:
            logger.error(f"Reranking failed: {e}")
            return documents[:self.top_n]

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


class VectorStoreManager(VectorStoreInterface):
    """Manages vector store with proper persistence and hybrid retrieval"""
    
    def __init__(self, config: Config, model_manager: ModelManager):
        self.config = config
        self.model_manager = model_manager
        self.vector_store = None
        self.hybrid_retriever = None
        self._documents_cache = []
        
    def load_store(self) -> bool:
        """Load existing vector store from persistence"""
        persist_dir = Path(self.config.PERSIST_DIRECTORY)
        
        if not persist_dir.exists() or not (persist_dir / "chroma.sqlite3").exists():
            logger.info("No existing vector store found")
            return False
        
        try:
            logger.info(f"Loading vector store from {persist_dir}")
            embeddings = self.model_manager.get_embeddings()
            
            self.vector_store = Chroma(
                persist_directory=str(persist_dir),
                embedding_function=embeddings,
                collection_name=self.config.COLLECTION_NAME
            )
            
            collection = self.vector_store.get()
            if not collection['documents']:
                logger.warning("Vector store exists but is empty")
                return False
            
            logger.info(f"Loaded {len(collection['documents'])} documents from vector store")
            
            self._setup_hybrid_retriever_from_store()
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to load vector store: {e}")
            return False
    
    def create_store(self, documents: List[Document]) -> Any:
        """Create new vector store with documents"""
        if not documents:
            raise ValueError("No documents provided for vector store creation")
        
        try:
            logger.info(f"Creating vector store with {len(documents)} documents")
            embeddings = self.model_manager.get_embeddings()
            
            self.vector_store = Chroma.from_documents(
                documents=documents,
                embedding=embeddings,
                persist_directory=self.config.PERSIST_DIRECTORY,
                collection_name=self.config.COLLECTION_NAME
            )
            
            self._documents_cache = documents

            self._setup_hybrid_retriever(documents)
            
            logger.info("Vector store created successfully")
            return self.vector_store
            
        except Exception as e:
            logger.error(f"Failed to create vector store: {e}")
            raise ServiceException(f"Failed to create vector store: {e}")
    
    def _setup_hybrid_retriever(self, documents: List[Document]):
        """Setup hybrid retriever with vector and BM25 components"""
        try:
            vector_retriever = self.vector_store.as_retriever(
                search_type="similarity",
                search_kwargs={"k": self.config.SEARCH_K}
            )

            bm25_retriever = BM25Retriever.from_documents(
                documents=documents,
                k=self.config.SEARCH_K
            )

            ensemble_retriever = EnsembleRetriever(
                retrievers=[vector_retriever, bm25_retriever],
                weights=[self.config.VECTOR_WEIGHT, self.config.BM25_WEIGHT]
            )

            reranker = CustomReranker(
                model_name=self.config.RERANKER_MODEL,
                top_n=self.config.RERANK_K
            )
            
            self.hybrid_retriever = ContextualCompressionRetriever(
                base_compressor=reranker,
                base_retriever=ensemble_retriever
            )
            
            logger.info("Hybrid retriever setup complete")
            
        except Exception as e:
            logger.error(f"Failed to setup hybrid retriever: {e}")
            raise ServiceException(f"Failed to setup hybrid retriever: {e}")
    
    def _setup_hybrid_retriever_from_store(self):
        """Setup hybrid retriever from existing vector store"""
        try:
            collection = self.vector_store.get()
            documents = [
                Document(page_content=doc, metadata=meta or {})
                for doc, meta in zip(collection['documents'], collection['metadatas'])
            ]
            
            self._setup_hybrid_retriever(documents)
            
        except Exception as e:
            logger.error(f"Failed to setup hybrid retriever from store: {e}")
            raise ServiceException(f"Failed to setup hybrid retriever from store: {e}")
    
    def get_retriever(self) -> Any:
        """Get the configured hybrid retriever"""
        if self.hybrid_retriever is None:
            raise ServiceException("Retriever not initialized")
        return self.hybrid_retriever


class RAGServiceInterface(ABC):
    @abstractmethod
    def setup_chain(self) -> None:
        pass
    
    @abstractmethod
    def query(self, question: str, use_enhanced_query: bool = False) -> Dict[str, Any]:
        pass
    
    @abstractmethod
    def get_last_chunks(self) -> List[Document]:
        pass
    
    @abstractmethod
    def clear_memory(self) -> None:
        pass


class RAGService(RAGServiceInterface):
    """Main RAG service with enhanced query processing"""
    
    def __init__(self, config: Config, model_manager: ModelManager, vector_store_manager: VectorStoreManager):
        self.config = config
        self.model_manager = model_manager
        self.vector_store_manager = vector_store_manager
        self.last_retrieved_chunks = []
        self.chain = None
        self.memory = None
        
    def setup_chain(self) -> None:
        """Setup the conversational retrieval chain"""
        try:
            self.memory = ConversationBufferWindowMemory(
                memory_key="chat_history",
                return_messages=True,
                output_key="answer",
                k=5 
            )
            
            qa_prompt = PromptTemplate(
                template=self.config.QA_TEMPLATE,
                input_variables=["context", "question"]
            )
            
            self.chain = ConversationalRetrievalChain.from_llm(
                llm=self.model_manager.get_llm(),
                retriever=self.vector_store_manager.get_retriever(),
                memory=self.memory,
                return_source_documents=True,
                return_generated_question=True,
                combine_docs_chain_kwargs={"prompt": qa_prompt},
                verbose=True
            )
            
            logger.info("RAG chain setup complete")
            
        except Exception as e:
            logger.error(f"Failed to setup RAG chain: {e}")
            raise ServiceException(f"Failed to setup RAG chain: {e}")
    
    def query(self, question: str, use_enhanced_query: bool = False, chat_history: List = None) -> Dict[str, Any]:
        """Process query with optional enhancement and chat history"""
        if not self.chain:
            raise ServiceException("RAG chain not initialized")
        
        start_time = time.time()
        
        try:
            processed_question = question
            if use_enhanced_query:
                processed_question = self._enhance_query(question, chat_history)
                logger.info(f"Enhanced query: {processed_question}")

            result = self.chain({"question": processed_question})

            self.last_retrieved_chunks = result.get("source_documents", [])

            source_documents = [
                SourceDocument(
                    content=doc.page_content,
                    metadata=doc.metadata,
                    score=doc.metadata.get("score"),
                    rerank_score=doc.metadata.get("rerank_score")
                )
                for doc in self.last_retrieved_chunks
            ]
            
            processing_time = time.time() - start_time
            
            return {
                "answer": result["answer"],
                "source_documents": source_documents,
                "generated_question": result.get("generated_question"),
                "enhanced_query": use_enhanced_query,
                "processing_time": processing_time
            }
            
        except Exception as e:
            logger.error(f"Query processing failed: {e}")
            raise ServiceException(f"Query processing failed: {e}")

    def _enhance_query(self, query: str, chat_history: List = None) -> str:
        """Enhance query using LLM with chat history context"""
        try:
            llm = self.model_manager.get_llm()

            history_context = self._format_chat_history(chat_history)
            
            enhancement_prompt = PromptTemplate(
                template=self.config.QUERY_ENHANCEMENT_TEMPLATE,
                input_variables=["query", "chat_history"]
            )
            
            formatted_prompt = enhancement_prompt.format(
                query=query, 
                chat_history=history_context
            )
            response = llm.invoke(formatted_prompt)
            
            enhanced_query = response.content.strip()
            return enhanced_query if enhanced_query else query
            
        except Exception as e:
            logger.error(f"Query enhancement failed: {e}")
            return query

    def _format_chat_history(self, chat_history: List = None) -> str:
        """Format chat history for query enhancement context"""
        if not chat_history:
            return "Tidak ada riwayat percakapan sebelumnya."

        recent_history = chat_history[-6:] if len(chat_history) > 6 else chat_history
        
        formatted_history = []
        for msg in recent_history:
            role = "Pengguna" if msg.role == "user" else "Asisten"
            content = msg.content[:300] + "..." if len(msg.content) > 300 else msg.content
            formatted_history.append(f"{role}: {content}")
        
        return "\n".join(formatted_history) if formatted_history else "Tidak ada riwayat percakapan sebelumnya."
    
    def get_last_chunks(self) -> List[Document]:
        """Get chunks from last query"""
        return self.last_retrieved_chunks
    
    def clear_memory(self) -> None:
        """Clear conversation memory"""
        if self.memory:
            self.memory.clear()
            logger.info("Memory cleared")