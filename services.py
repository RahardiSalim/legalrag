import os
import hashlib
import logging
import tempfile
import re
import time
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from langchain.schema import Document

from pathlib import Path
from contextlib import contextmanager

from langchain.schema import Document
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain_community.vectorstores.utils import filter_complex_metadata
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
from models import SourceDocument, DocumentMetadata, SearchType, GraphEntity, GraphRelationship
from graph_service import SemanticGraphService

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

@dataclass
class LegalChunk:
    """Represents a hierarchical legal document chunk"""
    content: str
    level: int
    section_type: str
    section_number: str
    parent_sections: List[str]
    metadata: Dict[str, Any]


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
                    max_output_tokens=65536
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


class HierarchicalLegalSplitter:
    """Enhanced text splitter for Indonesian legal documents"""
    
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        # Indonesian legal document patterns
        self.patterns = {
            'bab': r'(?i)^BAB\s+([IVXLCDM]+|[0-9]+)[\s\.\-]*(.*)',
            'bagian': r'(?i)^BAGIAN\s+([IVXLCDM]+|[0-9]+)[\s\.\-]*(.*)',
            'pasal': r'(?i)^PASAL\s+([0-9]+)[\s\.\-]*(.*)',
            'ayat': r'(?i)^\(([0-9]+)\)[\s]*(.*)',
            'huruf': r'(?i)^([a-z])\.\s*(.*)',
            'angka': r'(?i)^([0-9]+)\.\s*(.*)',
            'paragraf': r'(?i)^PARAGRAF\s+([0-9]+)[\s\.\-]*(.*)',
            'sub_bagian': r'(?i)^SUB\s+BAGIAN\s+([0-9]+)[\s\.\-]*(.*)'
        }
                
        # Hierarchy levels (lower number = higher level)
        self.hierarchy_levels = {
            'bab': 1,
            'bagian': 2,
            'sub_bagian': 3,
            'paragraf': 4,
            'pasal': 5,
            'ayat': 6,
            'huruf': 7,
            'angka': 8
        }
    
    def split_documents(self, documents: List[Document]) -> List[Document]:
        """Split documents using hierarchical legal structure"""
        all_chunks = []
        
        for doc in documents:
            legal_chunks = self._parse_legal_structure(doc)
            document_chunks = self._create_hierarchical_chunks(legal_chunks, doc)
            all_chunks.extend(document_chunks)
        
        return all_chunks
    
    def _parse_legal_structure(self, document: Document) -> List[LegalChunk]:
        """Parse document into hierarchical legal chunks"""
        lines = document.page_content.split('\n')
        legal_chunks = []
        current_hierarchy = {}
        
        for line_num, line in enumerate(lines):
            line = line.strip()
            if not line:
                continue
            
            # Check if line matches any legal pattern
            matched_pattern = None
            for pattern_name, pattern in self.patterns.items():
                match = re.match(pattern, line)
                if match:
                    matched_pattern = pattern_name
                    section_number = match.group(1)
                    section_title = match.group(2) if len(match.groups()) > 1 else ""
                    break
            
            if matched_pattern:
                # Update current hierarchy
                level = self.hierarchy_levels[matched_pattern]
                
                # Clear lower levels
                current_hierarchy = {k: v for k, v in current_hierarchy.items() 
                                   if self.hierarchy_levels[k] < level}
                
                # Add current section
                current_hierarchy[matched_pattern] = {
                    'number': section_number,
                    'title': section_title,
                    'line_start': line_num
                }
                
                # Create parent section path
                parent_sections = []
                for hierarchy_type in sorted(current_hierarchy.keys(), 
                                           key=lambda x: self.hierarchy_levels[x]):
                    if hierarchy_type != matched_pattern:
                        parent_sections.append(f"{hierarchy_type.upper()} {current_hierarchy[hierarchy_type]['number']}")
                
                # Collect content until next section of same or higher level
                content_lines = [line]
                for next_line_num in range(line_num + 1, len(lines)):
                    next_line = lines[next_line_num].strip()
                    if not next_line:
                        content_lines.append("")
                        continue
                    
                    # Check if this is a section of same or higher level
                    is_higher_section = False
                    for check_pattern, check_regex in self.patterns.items():
                        if re.match(check_regex, next_line):
                            check_level = self.hierarchy_levels[check_pattern]
                            if check_level <= level:
                                is_higher_section = True
                                break
                    
                    if is_higher_section:
                        break
                    
                    content_lines.append(next_line)
                
                # Create legal chunk
                legal_chunk = LegalChunk(
                    content='\n'.join(content_lines),
                    level=level,
                    section_type=matched_pattern,
                    section_number=section_number,
                    parent_sections=parent_sections,
                    metadata=document.metadata.copy()
                )
                
                legal_chunks.append(legal_chunk)
        
        return legal_chunks
    
    def _create_hierarchical_chunks(self, legal_chunks: List[LegalChunk], 
                                  original_doc: Document) -> List[Document]:
        """Create final document chunks with proper hierarchy"""
        chunks = []
        
        for legal_chunk in legal_chunks:
            # If chunk is too large, split it while preserving context
            if len(legal_chunk.content) > self.chunk_size:
                sub_chunks = self._split_large_chunk(legal_chunk)
                chunks.extend(sub_chunks)
            else:
                chunks.append(self._create_document_chunk(legal_chunk, original_doc))
        
        return chunks
    
    def _split_large_chunk(self, legal_chunk: LegalChunk) -> List[Document]:
        """Split large chunks while preserving legal context"""
        from langchain.text_splitter import RecursiveCharacterTextSplitter
        
        # Use recursive splitter for large chunks
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=["\n\n\n", "\n\n", "\n", ".", "?", "!", " ", ""]
        )
        
        sub_chunks = splitter.split_text(legal_chunk.content)
        documents = []
        
        for i, sub_chunk in enumerate(sub_chunks):
            # Create context header for sub-chunks
            context_header = self._create_context_header(legal_chunk)
            full_content = f"{context_header}\n\n{sub_chunk}"
            
            metadata = legal_chunk.metadata.copy()
            metadata.update({
                'section_type': legal_chunk.section_type,
                'section_number': legal_chunk.section_number,
                'hierarchy_level': legal_chunk.level,
                'parent_sections': " > ".join(legal_chunk.parent_sections) if legal_chunk.parent_sections else "",
                'sub_chunk_index': i,
                'total_sub_chunks': len(sub_chunks),
                'is_sub_chunk': True
            })
            
            doc = Document(page_content=full_content, metadata=metadata)
            documents.append(doc)
        
        return documents
    
    def _create_document_chunk(self, legal_chunk: LegalChunk, 
                             original_doc: Document) -> Document:
        """Create a document chunk from legal chunk"""
        context_header = self._create_context_header(legal_chunk)
        full_content = f"{context_header}\n\n{legal_chunk.content}"
        
        metadata = legal_chunk.metadata.copy()
        metadata.update({
            'section_type': legal_chunk.section_type,
            'section_number': legal_chunk.section_number,
            'hierarchy_level': legal_chunk.level,
            'parent_sections': " > ".join(legal_chunk.parent_sections) if legal_chunk.parent_sections else "",
            'is_sub_chunk': False
        })
        
        return Document(page_content=full_content, metadata=metadata)
    
    def _create_context_header(self, legal_chunk: LegalChunk) -> str:
        """Create context header for better understanding"""
        context_parts = []
        
        # Add parent sections for context
        if legal_chunk.parent_sections:
            context_parts.extend(legal_chunk.parent_sections)
        
        # Add current section
        current_section = f"{legal_chunk.section_type.upper()} {legal_chunk.section_number}"
        context_parts.append(current_section)
        
        return " > ".join(context_parts)


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

    def _create_hierarchical_text_splitter(self) -> HierarchicalLegalSplitter:
        """Create hierarchical text splitter for legal documents"""
        return HierarchicalLegalSplitter(
            chunk_size=self.config.CHUNK_SIZE,
            chunk_overlap=self.config.CHUNK_OVERLAP
        )

    def _detect_document_type(self, content: str) -> str:
        """Detect if document is legal/regulatory"""
        legal_indicators = [
            'pasal', 'bab', 'bagian', 'ayat', 'peraturan', 'undang-undang',
            'keputusan', 'surat edaran', 'ojk', 'otoritas jasa keuangan',
            'peraturan otoritas jasa keuangan', 'pojk'
        ]
        
        content_lower = content.lower()
        legal_score = sum(1 for indicator in legal_indicators if indicator in content_lower)
        
        return 'legal' if legal_score >= 3 else 'general'
        
    @property
    def text_splitter(self) -> RecursiveCharacterTextSplitter:
        if self._text_splitter is None:
            self._text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.config.CHUNK_SIZE,
                chunk_overlap=self.config.CHUNK_OVERLAP,
                length_function=len,
                separators=[]
            )
        return self._text_splitter
    
    def process_documents(self, file_paths: List[str]) -> List[Document]:
        """Enhanced document processing with hierarchical chunking"""
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
        
        # Detect document types and choose appropriate splitter
        legal_docs = []
        general_docs = []
        
        for doc in all_documents:
            doc_type = self._detect_document_type(doc.page_content)
            if doc_type == 'legal':
                legal_docs.append(doc)
            else:
                general_docs.append(doc)
        
        chunks = []
        
        # Use hierarchical splitter for legal documents
        if legal_docs:
            hierarchical_splitter = self._create_hierarchical_text_splitter()
            legal_chunks = hierarchical_splitter.split_documents(legal_docs)
            chunks.extend(legal_chunks)
            logger.info(f"Created {len(legal_chunks)} hierarchical chunks from {len(legal_docs)} legal documents")
        
        # Use regular splitter for general documents
        if general_docs:
            regular_chunks = self.text_splitter.split_documents(general_docs)
            chunks.extend(regular_chunks)
            logger.info(f"Created {len(regular_chunks)} regular chunks from {len(general_docs)} general documents")
        
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
        """Enhanced chunk processing with legal document detection"""
        unique_chunks = []
        
        for i, chunk in enumerate(chunks):
            content = self._clean_content(chunk.page_content)
            
            if not content or len(content.strip()) < 50:
                continue
            
            # Detect document type
            doc_type = self._detect_document_type(content)
            
            content_hash = self.create_content_hash(content)
            
            if content_hash not in self.processed_hashes:
                self.processed_hashes.add(content_hash)
                
                chunk.page_content = content
                chunk.metadata.update({
                    'chunk_id': f"chunk_{i}_{content_hash[:8]}",
                    'content_hash': content_hash,
                    'chunk_length': len(content),
                    'chunk_index': i,
                    'document_type': doc_type
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
        self.graph_service = None
        
    def add_documents(self, documents: List[Document]) -> bool:
        """Add new documents to existing vector store and update graph"""
        if not self.vector_store:
            raise ServiceException("Vector store not initialized")
        
        try:
            logger.info(f"Adding {len(documents)} new documents to vector store")
            
            # Filter and clean documents
            filtered_documents = filter_complex_metadata(documents)
            logger.info(f"Filtered complex metadata from {len(documents)} documents")
            
            # Additional cleanup to ensure no complex types remain
            for doc in filtered_documents:
                cleaned_metadata = {}
                for key, value in doc.metadata.items():
                    if isinstance(value, (str, int, float, bool)) or value is None:
                        cleaned_metadata[key] = value
                    elif isinstance(value, list):
                        cleaned_metadata[key] = " > ".join(str(item) for item in value) if value else ""
                    else:
                        cleaned_metadata[key] = str(value)
                doc.metadata = cleaned_metadata
            
            # Add to vector store
            self.vector_store.add_documents(filtered_documents)
            self._documents_cache.extend(filtered_documents)
            
            # Update hybrid retriever with new documents
            self._setup_hybrid_retriever(self._documents_cache)
            
            # FIXED: Improved graph update with better error handling
            if self.graph_service and self.config.ENABLE_GRAPH_PROCESSING:
                try:
                    logger.info("Updating graph database incrementally with new documents")
                    
                    # Use incremental update if graph exists, otherwise full creation
                    if self.graph_service.has_graph_data():
                        graph_updated = self.graph_service.update_graph_with_documents(filtered_documents)
                        if graph_updated:
                            logger.info("Graph database updated incrementally successfully")
                        else:
                            logger.warning("Incremental graph update failed, attempting full rebuild")
                            # Fallback to full rebuild if incremental fails
                            graph_updated = self.graph_service.process_documents_to_graph(self._documents_cache)
                            if graph_updated:
                                logger.info("Graph database rebuilt successfully")
                            else:
                                logger.error("Both incremental and full graph updates failed")
                    else:
                        # No existing graph, create new one
                        graph_updated = self.graph_service.process_documents_to_graph(filtered_documents)
                        if graph_updated:
                            logger.info("New graph database created successfully")
                        else:
                            logger.error("Failed to create new graph database")
                            
                except Exception as e:
                    logger.error(f"Graph update failed: {e}")
                    # Don't fail the entire operation if just graph update fails
            
            logger.info(f"Successfully added {len(filtered_documents)} documents")
            return True
            
        except Exception as e:
            logger.error(f"Failed to add documents: {e}")
            return False

    def create_store(self, documents: List[Document]) -> Any:
        """Create new vector store with documents and update graph"""
        if not documents:
            raise ValueError("No documents provided for vector store creation")
        
        try:
            logger.info(f"Creating vector store with {len(documents)} documents")
            
            # Filter complex metadata before creating vector store
            filtered_documents = filter_complex_metadata(documents)
            logger.info(f"Filtered complex metadata from {len(documents)} documents")
            
            # Additional cleanup to ensure no complex types remain
            for doc in filtered_documents:
                cleaned_metadata = {}
                for key, value in doc.metadata.items():
                    if isinstance(value, (str, int, float, bool)) or value is None:
                        cleaned_metadata[key] = value
                    elif isinstance(value, list):
                        cleaned_metadata[key] = " > ".join(str(item) for item in value) if value else ""
                    else:
                        cleaned_metadata[key] = str(value)
                doc.metadata = cleaned_metadata
            
            embeddings = self.model_manager.get_embeddings()
            
            self.vector_store = Chroma.from_documents(
                documents=filtered_documents,
                embedding=embeddings,
                persist_directory=self.config.PERSIST_DIRECTORY,
                collection_name=self.config.COLLECTION_NAME
            )
            
            self._documents_cache = filtered_documents
            self._setup_hybrid_retriever(filtered_documents)
            
            # AUTOMATICALLY UPDATE GRAPH DATABASE
            if self.graph_service:
                try:
                    logger.info("Automatically updating graph database with new documents")
                    graph_updated = self.graph_service.process_documents_to_graph(filtered_documents)
                    if graph_updated:
                        logger.info("Graph database updated successfully")
                    else:
                        logger.warning("Graph database update failed")
                except Exception as e:
                    logger.error(f"Failed to update graph database: {e}")
                    # Don't fail the entire process if graph update fails
            
            logger.info("Vector store created successfully")
            return self.vector_store
            
        except Exception as e:
            logger.error(f"Failed to create vector store: {e}")
            raise ServiceException(f"Failed to create vector store: {e}")
    
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
    def query(self, question: str, search_type: SearchType = SearchType.VECTOR, use_enhanced_query: bool = False) -> Dict[str, Any]:
        pass
    
    @abstractmethod
    def get_last_chunks(self) -> List[Document]:
        pass
    
    @abstractmethod
    def clear_memory(self) -> None:
        pass


class RAGService(RAGServiceInterface):
    """Main RAG service with enhanced query processing and graph integration"""
    
    def __init__(self, config: Config, model_manager: ModelManager, vector_store_manager: VectorStoreManager):
        self.config = config
        self.model_manager = model_manager
        self.vector_store_manager = vector_store_manager
        self.graph_service = SemanticGraphService(config) if config.ENABLE_GRAPH_PROCESSING else None
        
        # Pass graph service to vector store manager for automatic updates
        self.vector_store_manager.graph_service = self.graph_service
        
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
            
            # Try to load existing graph data
            if self.graph_service:
                self.graph_service.load_graph_data()
            
            logger.info("RAG chain setup complete")
            
        except Exception as e:
            logger.error(f"Failed to setup RAG chain: {e}")
            raise ServiceException(f"Failed to setup RAG chain: {e}")
    
    def query(self, question: str, search_type: SearchType = SearchType.VECTOR, use_enhanced_query: bool = False, chat_history: List = None) -> Dict[str, Any]:
        """Process query with different search types"""
        if not self.chain:
            raise ServiceException("RAG chain not initialized")
        
        start_time = time.time()
        
        try:
            if search_type == SearchType.GRAPH:
                return self._query_graph(question, use_enhanced_query, chat_history, start_time)
            elif search_type == SearchType.HYBRID:
                return self._query_hybrid(question, use_enhanced_query, chat_history, start_time)
            else:
                return self._query_vector(question, use_enhanced_query, chat_history, start_time)
                
        except Exception as e:
            logger.error(f"Query processing failed: {e}")
            raise ServiceException(f"Query processing failed: {e}")
    
    def _query_vector(self, question: str, use_enhanced_query: bool, chat_history: List, start_time: float) -> Dict[str, Any]:
        """Process query using vector search"""
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
            "search_type_used": SearchType.VECTOR,
            "processing_time": processing_time,
            "graph_entities": [],
            "graph_relationships": []
        }
    
    def _query_graph(self, question: str, use_enhanced_query: bool, chat_history: List, start_time: float) -> Dict[str, Any]:
        """Process query using enhanced semantic graph search"""
        if not self.graph_service or not self.graph_service.has_graph_data():
            # Fallback to vector search if graph not available
            logger.warning("Graph search requested but no graph data available, falling back to vector search")
            return self._query_vector(question, use_enhanced_query, chat_history, start_time)
        
        processed_question = question
        if use_enhanced_query:
            processed_question = self._enhance_query(question, chat_history)
        
        # Search graph using enhanced semantic search with traversal
        graph_result = self.graph_service.search_graph(processed_question)
        
        # Create graph-specific prompt
        graph_prompt = PromptTemplate(
            template=self.config.GRAPH_QA_TEMPLATE,
            input_variables=["graph_context", "question"]
        )
        
        # Generate answer using graph context
        llm = self.model_manager.get_llm()
        formatted_prompt = graph_prompt.format(
            graph_context=graph_result["context"],
            question=processed_question
        )
        
        response = llm.invoke(formatted_prompt)
        answer = response.content
        
        # Convert graph entities and relationships to response format
        graph_entities = [
            GraphEntity(
                id=entity["id"],
                type=entity["type"],
                properties=entity.get("properties", {})
            )
            for entity in graph_result.get("relevant_entities", [])
        ]
        
        graph_relationships = [
            GraphRelationship(
                source=rel["source"],
                target=rel["target"],
                type=rel["type"],
                properties=rel.get("properties", {})
            )
            for rel in graph_result.get("relevant_relationships", [])
        ]
        
        processing_time = time.time() - start_time
        
        return {
            "answer": answer,
            "source_documents": [],  # Graph search doesn't use chunked documents
            "generated_question": processed_question if use_enhanced_query else None,
            "enhanced_query": use_enhanced_query,
            "search_type_used": SearchType.GRAPH,
            "processing_time": processing_time,
            "graph_entities": graph_entities,
            "graph_relationships": graph_relationships
        }
    
    def _query_hybrid(self, question: str, use_enhanced_query: bool, chat_history: List, start_time: float) -> Dict[str, Any]:
        """Process query using both enhanced vector and graph search"""
        # Get vector search results
        vector_result = self._query_vector(question, use_enhanced_query, chat_history, start_time)
        
        # Get graph search results if available
        graph_entities = []
        graph_relationships = []
        
        if self.graph_service and self.graph_service.has_graph_data():
            processed_question = question
            if use_enhanced_query:
                processed_question = self._enhance_query(question, chat_history)
            
            # Use enhanced semantic graph search
            graph_result = self.graph_service.search_graph(processed_question)
            
            graph_entities = [
                GraphEntity(
                    id=entity["id"],
                    type=entity["type"],
                    properties=entity.get("properties", {})
                )
                for entity in graph_result.get("relevant_entities", [])
            ]
            
            graph_relationships = [
                GraphRelationship(
                    source=rel["source"],
                    target=rel["target"],
                    type=rel["type"],
                    properties=rel.get("properties", {})
                )
                for rel in graph_result.get("relevant_relationships", [])
            ]
            
            # Enhance answer with graph context if available
            if graph_result["context"] and graph_result["context"] != "Tidak ditemukan informasi graph yang relevan dengan pertanyaan.":
                enhanced_prompt = f"""
                Konteks dari pencarian vektor: {vector_result['answer']}
                
                Informasi tambahan dari graph dengan semantic search dan traversal: {graph_result['context']}
                
                Entry points yang ditemukan: {graph_result.get('entry_points', [])}
                
                Pertanyaan: {question}
                
                Berikan jawaban yang menggabungkan informasi dari kedua sumber di atas dengan fokus pada hubungan semantik:
                """
                
                llm = self.model_manager.get_llm()
                enhanced_response = llm.invoke(enhanced_prompt)
                vector_result["answer"] = enhanced_response.content
        
        # Update result with hybrid information
        vector_result.update({
            "search_type_used": SearchType.HYBRID,
            "graph_entities": graph_entities,
            "graph_relationships": graph_relationships,
            "processing_time": time.time() - start_time
        })
        
        return vector_result

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
    
    def get_graph_stats(self) -> Dict[str, Any]:
        """Get enhanced graph statistics"""
        if not self.graph_service:
            return {"has_graph": False}
        
        stats = self.graph_service.get_graph_stats()
        stats["has_graph"] = True
        return stats
    
    def visualize_graph(self, filename: str = "graph_visualization.html") -> str:
        """Create graph visualization"""
        if not self.graph_service:
            return ""
        
        return self.graph_service.visualize_graph(filename)