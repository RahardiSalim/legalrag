from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
import hashlib
import re
from langchain.schema import Document
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from sentence_transformers import CrossEncoder
from config import Config


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
    def __init__(self, config: Config):
        self.config = config
        self._llm = None
        self._embeddings = None
        self._reranker = None
    
    def get_llm(self) -> ChatGoogleGenerativeAI:
        if self._llm is None:
            self._llm = ChatGoogleGenerativeAI(
                model=self.config.LLM_MODEL,
                google_api_key=self.config.GEMINI_API_KEY,
                safety_settings=self.config.SAFETY_SETTINGS,
                temperature=self.config.LLM_TEMPERATURE
            )
        return self._llm
    
    def get_embeddings(self) -> GoogleGenerativeAIEmbeddings:
        if self._embeddings is None:
            self._embeddings = GoogleGenerativeAIEmbeddings(
                model=self.config.EMBEDDING_MODEL,
                google_api_key=self.config.GEMINI_API_KEY
            )
        return self._embeddings
    
    def get_reranker(self) -> CrossEncoder:
        if self._reranker is None:
            self._reranker = CrossEncoder("cross-encoder/mmarco-mMiniLMv2-L12-H384-v1") #cross-encoder/ms-marco-MiniLM-L-2-v2
        return self._reranker


class DocumentProcessorInterface(ABC):
    @abstractmethod
    def process_documents(self, file_paths: List[str]) -> List[Document]:
        pass
    
    @abstractmethod
    def create_content_hash(self, content: str) -> str:
        pass


class DocumentProcessor(DocumentProcessorInterface):
    def __init__(self, config: Config):
        self.config = config
        self.processed_hashes = set()
    
    def process_documents(self, file_paths: List[str]) -> List[Document]:
        from langchain.document_loaders import PyPDFLoader, TextLoader
        from langchain.text_splitter import RecursiveCharacterTextSplitter
        from pathlib import Path
        
        all_documents = []
        
        # Load documents
        for file_path in file_paths:
            try:
                file_extension = Path(file_path).suffix.lower()
                
                if file_extension == '.pdf':
                    loader = PyPDFLoader(file_path)
                elif file_extension == '.txt':
                    loader = TextLoader(file_path)
                else:
                    continue
                
                documents = loader.load()
                all_documents.extend(documents)
                
            except Exception as e:
                print(f"Error loading {file_path}: {e}")
        
        # Split documents
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.config.CHUNK_SIZE,
            chunk_overlap=self.config.CHUNK_OVERLAP,
            separators=["\n\n\n\n", "\n\n", "\n", ".", "?", "!", " ", ""]
        )
        
        chunks = text_splitter.split_documents(all_documents)
        
        # Remove duplicates
        unique_chunks = self._remove_duplicates(chunks)
        
        return unique_chunks
    
    def _remove_duplicates(self, chunks: List[Document]) -> List[Document]:
        unique_chunks = []
        
        for chunk in chunks:
            content_hash = self.create_content_hash(chunk.page_content)
            
            if content_hash not in self.processed_hashes:
                self.processed_hashes.add(content_hash)
                unique_chunks.append(chunk)
        
        return unique_chunks
    
    def create_content_hash(self, content: str) -> str:
        normalized_content = re.sub(r'\s+', ' ', content.strip().lower())
        return hashlib.sha256(normalized_content.encode('utf-8')).hexdigest()


class VectorStoreInterface(ABC):
    @abstractmethod
    def create_store(self, documents: List[Document]) -> Any:
        pass
    
    @abstractmethod
    def get_retriever(self) -> Any:
        pass


class VectorStoreManager(VectorStoreInterface):
    def __init__(self, config: Config, model_manager: ModelManager):
        self.config = config
        self.model_manager = model_manager
        self.vector_store = None
        self.hybrid_retriever = None
    
    def create_store(self, documents: List[Document]) -> Any:
        from langchain.vectorstores import Chroma
        from langchain.retrievers import BM25Retriever, EnsembleRetriever
        
        embeddings = self.model_manager.get_embeddings()
        
        # Create vector store
        self.vector_store = Chroma.from_documents(
            documents,
            embeddings,
            persist_directory=self.config.PERSIST_DIRECTORY
        )
        
        # Setup hybrid retriever
        vector_retriever = self.vector_store.as_retriever(
            search_kwargs={"k": self.config.SEARCH_K}
        )
        
        bm25_retriever = BM25Retriever.from_documents(
            documents,
            k=self.config.SEARCH_K
        )
        
        self.hybrid_retriever = EnsembleRetriever(
            retrievers=[vector_retriever, bm25_retriever],
            weights=[self.config.VECTOR_WEIGHT, self.config.BM25_WEIGHT]
        )
        
        return self.vector_store
    
    def get_retriever(self) -> Any:
        return self.hybrid_retriever


class RAGServiceInterface(ABC):
    @abstractmethod
    def query(self, question: str, use_enhanced_query: bool = False) -> Dict[str, Any]:
        pass
    
    @abstractmethod
    def get_last_chunks(self) -> List[Document]:
        pass


class RAGService(RAGServiceInterface):
    def __init__(self, config: Config, model_manager: ModelManager, vector_store_manager: VectorStoreManager):
        self.config = config
        self.model_manager = model_manager
        self.vector_store_manager = vector_store_manager
        self.last_retrieved_chunks = []
        self.chain = None
    
    def setup_chain(self):
        from langchain.chains import ConversationalRetrievalChain
        from langchain.memory import ConversationBufferMemory
        from langchain.prompts import PromptTemplate
        
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            output_key="answer"
        )
        
        # Create custom prompt
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
            combine_docs_chain_kwargs={"prompt": qa_prompt}
        )
    
    def query(self, question: str, use_enhanced_query: bool = False) -> Dict[str, Any]:
        if not self.chain:
            raise ValueError("RAG chain not initialized")
        
        if use_enhanced_query:
            question = self._enhance_query(question)
        
        result = self.chain({"question": question})
        self.last_retrieved_chunks = result.get("source_documents", [])
        
        return {
            "answer": result["answer"],
            "source_documents": [
                {
                    "content": doc.page_content,
                    "metadata": doc.metadata
                }
                for doc in self.last_retrieved_chunks
            ],
            "generated_question": result.get("generated_question"),
            "enhanced_query": use_enhanced_query
        }
    
    def _enhance_query(self, query: str) -> str:
        """Simple query enhancement - can be expanded"""
        llm = self.model_manager.get_llm()
        enhancement_prompt = f"""
        Pertanyaan berikut ini adalah tentang peraturan hukum dan OJK. 
        Reformulasikan pertanyaan ini agar lebih spesifik dan mudah dicari dalam dokumen hukum.
        
        Pertanyaan asli: {query}
        
        Pertanyaan yang diperbaiki:
        """
        
        response = llm.invoke(enhancement_prompt)
        return response.content.strip()
    
    def get_last_chunks(self) -> List[Document]:
        return self.last_retrieved_chunks
    
    def clear_memory(self):
        if hasattr(self, 'memory') and self.memory:
            self.memory.clear()