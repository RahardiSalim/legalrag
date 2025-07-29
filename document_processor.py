import hashlib
import logging
import time
import re
from typing import List
from pathlib import Path
from langchain.schema import Document
from langchain.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

from config import Config
from interfaces import DocumentProcessorInterface
from exceptions import DocumentProcessingException
from text_splitters import HierarchicalLegalSplitter

logger = logging.getLogger(__name__)

class DocumentProcessor(DocumentProcessorInterface):
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
                separators=[]
            )
        return self._text_splitter

    def process_documents(self, file_paths: List[str]) -> List[Document]:
        all_documents = []
        processing_errors = []
        
        logger.info(f"Processing {len(file_paths)} files...")
        
        for file_path in file_paths:
            try:
                documents = self._load_single_document(file_path)
                if documents:
                    all_documents.extend(documents)
                    logger.info(f"Loaded {len(documents)} documents from {file_path}")
            except Exception as e:
                error_msg = f"Error loading {file_path}: {str(e)}"
                logger.error(error_msg)
                processing_errors.append(error_msg)
        
        if not all_documents:
            raise DocumentProcessingException("No documents could be loaded successfully")
        
        return self._split_and_process_documents(all_documents)
    
    def _split_and_process_documents(self, documents: List[Document]) -> List[Document]:
        legal_docs, general_docs = self._categorize_documents(documents)
        chunks = []
        
        if legal_docs:
            hierarchical_splitter = HierarchicalLegalSplitter(
                chunk_size=self.config.CHUNK_SIZE,
                chunk_overlap=self.config.CHUNK_OVERLAP
            )
            legal_chunks = hierarchical_splitter.split_documents(legal_docs)
            chunks.extend(legal_chunks)
            logger.info(f"Created {len(legal_chunks)} hierarchical chunks from {len(legal_docs)} legal documents")
        
        if general_docs:
            regular_chunks = self.text_splitter.split_documents(general_docs)
            chunks.extend(regular_chunks)
            logger.info(f"Created {len(regular_chunks)} regular chunks from {len(general_docs)} general documents")
        
        unique_chunks = self._process_chunks(chunks)
        logger.info(f"After deduplication: {len(unique_chunks)} unique chunks")
        
        return unique_chunks
    
    def _categorize_documents(self, documents: List[Document]):
        legal_docs = []
        general_docs = []
        
        for doc in documents:
            doc_type = self._detect_document_type(doc.page_content)
            if doc_type == 'legal':
                legal_docs.append(doc)
            else:
                general_docs.append(doc)
        
        return legal_docs, general_docs
    
    def _detect_document_type(self, content: str) -> str:
        legal_indicators = [
            'pasal', 'bab', 'bagian', 'ayat', 'peraturan', 'undang-undang',
            'keputusan', 'surat edaran', 'ojk', 'otoritas jasa keuangan',
            'peraturan otoritas jasa keuangan', 'pojk'
        ]
        
        content_lower = content.lower()
        legal_score = sum(1 for indicator in legal_indicators if indicator in content_lower)
        
        return 'legal' if legal_score >= 3 else 'general'
    
    def _load_single_document(self, file_path: str) -> List[Document]:
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        if file_path.stat().st_size > self.config.MAX_DOCUMENT_SIZE:
            raise ValueError(f"File too large: {file_path} ({file_path.stat().st_size} bytes)")
        
        loader = self._get_document_loader(file_path)
        documents = loader.load()

        for doc in documents:
            doc.metadata.update({
                'source': str(file_path),
                'file_type': file_path.suffix.lower(),
                'file_size': file_path.stat().st_size,
                'processed_at': time.time()
            })
        
        return documents
    
    def _get_document_loader(self, file_path: Path):
        file_extension = file_path.suffix.lower()
        
        if file_extension == '.pdf':
            return PyPDFLoader(str(file_path))
        elif file_extension == '.txt':
            return TextLoader(str(file_path), encoding='utf-8')
        else:
            raise ValueError(f"Unsupported file type: {file_extension}")
    
    def _process_chunks(self, chunks: List[Document]) -> List[Document]:
        unique_chunks = []
        
        for i, chunk in enumerate(chunks):
            content = self._clean_content(chunk.page_content)
            
            if not content or len(content.strip()) < 50:
                continue
            
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
        content = re.sub(r'\s+', ' ', content)
        content = re.sub(r'[^\w\s\.\,\;\:\!\?\-\(\)\[\]\{\}\"\'\/\\]', '', content)
        return content.strip()
    
    def create_content_hash(self, content: str) -> str:
        normalized_content = re.sub(r'\s+', ' ', content.strip().lower())
        return hashlib.sha256(normalized_content.encode('utf-8')).hexdigest()