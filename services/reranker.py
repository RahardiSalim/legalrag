import logging
from typing import List, Optional, Any
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from langchain.schema import Document
from langchain.retrievers.document_compressors.base import BaseDocumentCompressor
from langchain.callbacks.manager import Callbacks

from config.settings import settings
from core.exceptions import ServiceException

logger = logging.getLogger(__name__)


class QwenReranker:
    """Custom Qwen Reranker using local model"""
    
    def __init__(self, model_path: str = None, device: str = None):
        self.model_path = model_path or settings.model.reranker_model
        self.device = device or settings.model.reranker_device
        self.tokenizer = None
        self.model = None
        self._initialize_model()

    def _initialize_model(self):
        """Initialize the Qwen reranker model"""
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_path, 
                trust_remote_code=True
            )
            
            try:
                self.model = AutoModelForSequenceClassification.from_pretrained(
                    self.model_path, 
                    trust_remote_code=True
                ).to(self.device)
            except:
                from transformers import AutoModel
                self.model = AutoModel.from_pretrained(
                    self.model_path, 
                    trust_remote_code=True
                ).to(self.device)
            
            self.model.eval()
            
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            logger.info(f"Qwen reranker initialized: {self.model_path}")
            
        except Exception as e:
            logger.error(f"Failed to initialize Qwen reranker: {e}")
            raise ServiceException(f"Failed to initialize Qwen reranker: {e}")

    def predict(self, pairs: List[List[str]]) -> List[float]:
        """Predict relevance scores for query-document pairs"""
        if not self.model or not self.tokenizer:
            raise RuntimeError("Model not initialized")
        
        scores = []
        max_length = settings.document.max_content_truncate
        
        for query, document in pairs:
            try:
                # Truncate document if too long
                if len(document) > max_length:
                    document = document[:max_length]
                
                score = self._score_pair(query, document)
                scores.append(score)
            except Exception as e:
                logger.warning(f"Failed to score pair: {e}")
                scores.append(0.0)
        
        return scores
    
    def _score_pair(self, query: str, document: str) -> float:
        """Score a single query-document pair"""
        try:
            input_text = f"Query: {query} Document: {document}"
            
            inputs = self.tokenizer(
                input_text,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512
            ).to(self.device)
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                
                if hasattr(outputs, 'logits'):
                    logits = outputs.logits
                    if logits.shape[-1] == 1:
                        score = torch.sigmoid(logits).item()
                    else:
                        score = torch.softmax(logits, dim=-1)[0, -1].item()
                else:
                    if hasattr(outputs, 'pooler_output'):
                        score = torch.sigmoid(outputs.pooler_output.mean()).item()
                    else:
                        score = torch.sigmoid(outputs.last_hidden_state.mean()).item()
            
            return float(score)
            
        except Exception as e:
            logger.warning(f"Error scoring pair: {e}")
            return 0.5


class QwenCustomReranker(BaseDocumentCompressor):
    """LangChain-compatible Qwen reranker"""
    
    def __init__(self, model_path: str = None, top_n: int = None, device: str = None, **kwargs: Any):
        base_kwargs = {}
        for key, value in kwargs.items():
            if key in ['callbacks']:
                base_kwargs[key] = value
        
        super().__init__(**base_kwargs)
        
        self._model_path = model_path or settings.model.reranker_model
        self._top_n = top_n or settings.retrieval.rerank_k
        self._device = device or settings.model.reranker_device
        self._qwen_reranker: Optional[QwenReranker] = None
        self._initialize_reranker()

    def _initialize_reranker(self):
        """Initialize the Qwen reranker"""
        if self._qwen_reranker is None:
            try:
                self._qwen_reranker = QwenReranker(self._model_path, self._device)
                logger.info(f"QwenCustomReranker initialized with model: {self._model_path}")
            except Exception as e:
                logger.error(f"Failed to initialize QwenCustomReranker: {e}")
                raise ServiceException(f"Failed to initialize Qwen reranker: {e}")

    @property
    def qwen_reranker(self) -> QwenReranker:
        if self._qwen_reranker is None:
            self._initialize_reranker()
        return self._qwen_reranker
        
    def compress_documents(
        self,
        documents: List[Document],
        query: str,
        callbacks: Optional[Callbacks] = None,
    ) -> List[Document]:
        if not documents or not query or self.qwen_reranker is None:
            return documents[:self._top_n] if documents else []
        
        try:
            logger.info(f"Reranking {len(documents)} documents with Qwen reranker")
            
            doc_pairs = []
            max_content = settings.document.max_content_truncate
            
            for doc in documents:
                content = doc.page_content[:max_content] if len(doc.page_content) > max_content else doc.page_content
                doc_pairs.append([query, content])
            
            scores = self.qwen_reranker.predict(doc_pairs)
            
            for doc, score in zip(documents, scores):
                doc.metadata["rerank_score"] = float(score)
            
            sorted_docs = sorted(
                documents, 
                key=lambda x: x.metadata.get("rerank_score", 0), 
                reverse=True
            )
            
            top_scores = [doc.metadata.get("rerank_score", 0) for doc in sorted_docs[:3]]
            logger.info(f"Qwen reranking complete. Top 3 scores: {top_scores}")
            
            return sorted_docs[:self._top_n]
            
        except Exception as e:
            logger.error(f"Qwen reranking failed: {e}")
            for doc in documents:
                doc.metadata["rerank_score"] = 0.5
            return documents[:self._top_n]


class CustomReranker(BaseDocumentCompressor):
    """Default reranker using configurable settings"""
    
    def __init__(self, top_n: int = None, device: str = None, **kwargs: Any):
        base_kwargs = {}
        for key, value in kwargs.items():
            if key in ['callbacks']:
                base_kwargs[key] = value
                
        super().__init__(**base_kwargs)
        
        self._top_n = top_n or settings.retrieval.rerank_k
        self._device = device or settings.model.reranker_device
        self._model_path = settings.model.reranker_model
        self._qwen_reranker: Optional[QwenReranker] = None
        self._initialize_reranker()

    def _initialize_reranker(self):
        """Initialize the Qwen reranker"""
        if self._qwen_reranker is None:
            try:
                self._qwen_reranker = QwenReranker(self._model_path, self._device)
                logger.info(f"CustomReranker initialized with Qwen model: {self._model_path}")
            except Exception as e:
                logger.error(f"Failed to initialize CustomReranker with Qwen: {e}")
                self._qwen_reranker = None

    def compress_documents(
        self,
        documents: List[Document],
        query: str,
        callbacks: Optional[Callbacks] = None,
    ) -> List[Document]:
        if not documents or not query:
            return documents[:self._top_n] if documents else []
        
        if self._qwen_reranker is not None:
            try:
                logger.info(f"Reranking {len(documents)} documents with Qwen reranker")
                
                doc_pairs = []
                max_content = settings.document.max_content_truncate
                
                for doc in documents:
                    content = doc.page_content[:max_content] if len(doc.page_content) > max_content else doc.page_content
                    doc_pairs.append([query, content])
                
                scores = self._qwen_reranker.predict(doc_pairs)
                
                for doc, score in zip(documents, scores):
                    doc.metadata["rerank_score"] = float(score)
                
                sorted_docs = sorted(
                    documents, 
                    key=lambda x: x.metadata.get("rerank_score", 0), 
                    reverse=True
                )
                
                return sorted_docs[:self._top_n]
                
            except Exception as e:
                logger.error(f"Qwen reranking failed: {e}")
        
        logger.info("Using fallback scoring (no reranking)")
        for doc in documents:
            doc.metadata["rerank_score"] = 0.5
            
        return documents[:self._top_n]


class FallbackReranker(BaseDocumentCompressor):
    """Fallback reranker using sentence-transformers"""
    
    def __init__(self, top_n: int = None, **kwargs: Any):
        base_kwargs = {}
        for key, value in kwargs.items():
            if key in ['callbacks']:
                base_kwargs[key] = value
                
        super().__init__(**base_kwargs)
        
        self._top_n = top_n or settings.retrieval.rerank_k
        self._model_name = "cross-encoder/ms-marco-MiniLM-L-12-v2"
        self._cross_encoder = None
        self._initialize_model()

    def _initialize_model(self):
        try:
            from sentence_transformers import CrossEncoder
            self._cross_encoder = CrossEncoder(self._model_name)
            logger.info(f"Fallback reranker initialized: {self._model_name}")
        except Exception as e:
            logger.error(f"Failed to initialize fallback reranker: {e}")
            
    def compress_documents(
        self,
        documents: List[Document],
        query: str,
        callbacks: Optional[Callbacks] = None,
    ) -> List[Document]:
        if not documents or not query or not self._cross_encoder:
            for doc in documents:
                doc.metadata["rerank_score"] = 0.5
            return documents[:self._top_n] if documents else []
        
        try:
            scores = []
            max_content = settings.document.max_content_truncate
            
            for doc in documents:
                try:
                    content = doc.page_content[:max_content] if len(doc.page_content) > max_content else doc.page_content
                    doc_pair = [query, content]
                    score = self._cross_encoder.predict([doc_pair])
                    scores.append(float(score[0]) if isinstance(score, (list, tuple)) else float(score))
                except:
                    scores.append(0.0)
            
            for doc, score in zip(documents, scores):
                doc.metadata["rerank_score"] = score
            
            sorted_docs = sorted(documents, key=lambda x: x.metadata.get("rerank_score", 0), reverse=True)
            return sorted_docs[:self._top_n]
            
        except Exception as e:
            logger.error(f"Fallback reranking failed: {e}")
            for doc in documents:
                doc.metadata["rerank_score"] = 0.5
            return documents[:self._top_n]