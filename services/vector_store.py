import logging
from typing import List, Any
from pathlib import Path
from langchain.schema import Document
from langchain_community.vectorstores import Chroma
from langchain_community.vectorstores.utils import filter_complex_metadata
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever, ContextualCompressionRetriever

from config.settings import settings
from core.interfaces import VectorStoreInterface, ModelManagerInterface
from core.exceptions import VectorStoreException
from services.reranker import CustomReranker

logger = logging.getLogger(__name__)


class VectorStoreManager(VectorStoreInterface):
    def __init__(self, model_manager: ModelManagerInterface):
        self.model_manager = model_manager
        self.vector_store = None
        self.hybrid_retriever = None
        self._documents_cache = []
        self.graph_service = None
        
    def create_store(self, documents: List[Document]) -> Any:
        if not documents:
            raise ValueError("No documents provided for vector store creation")
        
        try:
            logger.info(f"Creating vector store with {len(documents)} documents")
            
            filtered_documents = self._filter_and_clean_documents(documents)
            embeddings = self.model_manager.get_embeddings()
            
            self.vector_store = Chroma.from_documents(
                documents=filtered_documents,
                embedding=embeddings,
                persist_directory=settings.storage.persist_directory,
                collection_name=settings.storage.collection_name
            )
            
            self._documents_cache = filtered_documents
            self._setup_hybrid_retriever(filtered_documents)
            
            if self.graph_service:
                self._update_with_documents(filtered_documents)
            
            logger.info("Vector store created successfully")
            return self.vector_store
            
        except Exception as e:
            logger.error(f"Failed to create vector store: {e}")
            raise VectorStoreException(f"Failed to create vector store: {e}")
    
    def load_store(self) -> bool:
        persist_dir = Path(settings.storage.persist_directory)
        
        if not persist_dir.exists() or not (persist_dir / "chroma.sqlite3").exists():
            logger.info("No existing vector store found")
            return False
        
        try:
            logger.info(f"Loading vector store from {persist_dir}")
            embeddings = self.model_manager.get_embeddings()
            
            self.vector_store = Chroma(
                persist_directory=str(persist_dir),
                embedding_function=embeddings,
                collection_name=settings.storage.collection_name
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
    
    def add_documents(self, documents: List[Document]) -> bool:
        if not self.vector_store:
            raise VectorStoreException("Vector store not initialized")
        
        try:
            logger.info(f"Adding {len(documents)} new documents to vector store")
            
            filtered_documents = self._filter_and_clean_documents(documents)
            self.vector_store.add_documents(filtered_documents)
            self._documents_cache.extend(filtered_documents)
            
            self._setup_hybrid_retriever(self._documents_cache)
            
            if self.graph_service:
                self._update_graph_incrementally(filtered_documents)
            
            logger.info(f"Successfully added {len(filtered_documents)} documents")
            return True
            
        except Exception as e:
            logger.error(f"Failed to add documents: {e}")
            return False
    
    def get_retriever(self) -> Any:
        if self.hybrid_retriever is None:
            raise VectorStoreException("Retriever not initialized")
        return self.hybrid_retriever
    
    def _filter_and_clean_documents(self, documents: List[Document]) -> List[Document]:
        filtered_documents = filter_complex_metadata(documents)
        logger.info(f"Filtered complex metadata from {len(documents)} documents")
        
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
        
        return filtered_documents
    
    def _setup_hybrid_retriever(self, documents: List[Document]):
        try:
            vector_retriever = self.vector_store.as_retriever(
                search_type="similarity",
                search_kwargs={"k": settings.retrieval.search_k}
            )

            bm25_retriever = BM25Retriever.from_documents(
                documents=documents,
                k=settings.retrieval.search_k
            )

            ensemble_retriever = EnsembleRetriever(
                retrievers=[vector_retriever, bm25_retriever],
                weights=[settings.retrieval.vector_weight, settings.retrieval.bm25_weight]
            )

            logger.info(f"Setting up reranker with model: {settings.model.reranker_model}")
            
            try:
                from services.reranker import QwenCustomReranker
                reranker = QwenCustomReranker(
                    model_path=settings.model.reranker_model,
                    top_n=settings.retrieval.rerank_k,
                    device=settings.model.reranker_device
                )
                logger.info("Using Qwen reranker")
            except Exception as qwen_error:
                logger.warning(f"Qwen reranker failed: {qwen_error}")
                logger.info("Falling back to sentence-transformers reranker")
                
                try:
                    from services.reranker import FallbackReranker
                    reranker = FallbackReranker(top_n=settings.retrieval.rerank_k)
                    logger.info("Using fallback CrossEncoder reranker")
                except Exception as fallback_error:
                    logger.error(f"All rerankers failed: {fallback_error}")
                    from services.reranker import CustomReranker
                    reranker = CustomReranker(top_n=settings.retrieval.rerank_k)
            
            self.hybrid_retriever = ContextualCompressionRetriever(
                base_compressor=reranker,
                base_retriever=ensemble_retriever
            )
            
            logger.info("Hybrid retriever setup complete with reranker")
            
        except Exception as e:
            logger.error(f"Failed to setup hybrid retriever: {e}")
            raise VectorStoreException(f"Failed to setup hybrid retriever: {e}")
    
    def _setup_hybrid_retriever_from_store(self):
        try:
            collection = self.vector_store.get()
            documents = [
                Document(page_content=doc, metadata=meta or {})
                for doc, meta in zip(collection['documents'], collection['metadatas'])
            ]
            
            self._setup_hybrid_retriever(documents)
            
        except Exception as e:
            logger.error(f"Failed to setup hybrid retriever from store: {e}")
            raise VectorStoreException(f"Failed to setup hybrid retriever from store: {e}")
    
    def _update_with_documents(self, documents: List[Document]):
        try:
            logger.info("Updating graph database with new documents")
            graph_updated = self.graph_service.process_documents(documents)
            if graph_updated:
                logger.info("Graph database updated successfully")
            else:
                logger.warning("Graph database update failed")
        except Exception as e:
            logger.error(f"Failed to update graph database: {e}")
    
    def _update_graph_incrementally(self, documents: List[Document]):
        try:
            logger.info("Updating graph database incrementally with new documents")
            
            if self.graph_service.has_data():
                graph_updated = self.graph_service.update_with_documents(documents)
                if graph_updated:
                    logger.info("Graph database updated incrementally successfully")
                else:
                    logger.warning("Incremental graph update failed, attempting full rebuild")
                    graph_updated = self.graph_service.process_documents(self._documents_cache)
                    if graph_updated:
                        logger.info("Graph database rebuilt successfully")
                    else:
                        logger.error("Both incremental and full graph updates failed")
            else:
                graph_updated = self.graph_service.process_documents(documents)
                if graph_updated:
                    logger.info("New graph database created successfully")
                else:
                    logger.error("Failed to create new graph database")
                    
        except Exception as e:
            logger.error(f"Graph update failed: {e}")