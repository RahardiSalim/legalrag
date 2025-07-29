import logging
from typing import List, Optional, Any
from langchain.schema import Document
from langchain.retrievers.document_compressors.base import BaseDocumentCompressor
from langchain.callbacks.manager import Callbacks
from sentence_transformers import CrossEncoder

from exceptions import ServiceException

logger = logging.getLogger(__name__)

class CustomReranker(BaseDocumentCompressor):
    model_name: str = "cross-encoder/ms-marco-MiniLM-L-12-v2"
    top_n: int = 10
    _cross_encoder: Optional[CrossEncoder] = None

    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)
        self._initialize_model()

    def _initialize_model(self):
        if self._cross_encoder is None:
            try:
                self._cross_encoder = CrossEncoder(self.model_name)
                logger.info(f"CustomReranker initialized with model: {self.model_name}")
            except Exception as e:
                logger.error(f"Failed to initialize CustomReranker: {e}")
                raise ServiceException(f"Failed to initialize reranker: {e}")

    @property
    def cross_encoder(self) -> CrossEncoder:
        if self._cross_encoder is None:
            self._initialize_model()
        return self._cross_encoder
        
    def compress_documents(
        self,
        documents: List[Document],
        query: str,
        callbacks: Optional[Callbacks] = None,
    ) -> List[Document]:
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