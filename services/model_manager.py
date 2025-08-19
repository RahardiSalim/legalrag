import logging
from typing import Optional, List
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
from langchain.embeddings.base import Embeddings
from langchain_community.llms import Ollama
from langchain_community.chat_models import ChatOllama
from sentence_transformers import CrossEncoder

from config.settings import settings
from core.interfaces import ModelManagerInterface
from core.exceptions import ModelInitializationException

logger = logging.getLogger(__name__)


class QwenEmbeddings(Embeddings):
    """LangChain-compatible Qwen embeddings wrapper"""
    
    def __init__(self, model_path: str, device: str = None):
        self.model_path = model_path
        self.device = device or settings.model.embedding_device
        self.tokenizer = None
        self.model = None
        self._initialize_model()

    def _initialize_model(self):
        """Initialize the Qwen embedding model"""
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_path, 
                trust_remote_code=True
            )
            self.model = AutoModel.from_pretrained(
                self.model_path, 
                trust_remote_code=True
            ).to(self.device)
            self.model.eval()
            logger.info(f"Qwen embedding model initialized: {self.model_path}")
        except Exception as e:
            logger.error(f"Failed to initialize Qwen embedding model: {e}")
            raise ModelInitializationException(f"Failed to initialize Qwen embeddings: {e}")

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of documents"""
        return [self._embed_single(text) for text in texts]

    def embed_query(self, text: str) -> List[float]:
        """Embed a single query"""
        return self._embed_single(text)

    def _embed_single(self, text: str) -> List[float]:
        """Embed a single text"""
        if not self.model or not self.tokenizer:
            raise RuntimeError("Model not initialized")
        
        try:
            inputs = self.tokenizer(
                text, 
                return_tensors="pt", 
                truncation=True, 
                padding=True,
                max_length=512
            ).to(self.device)
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                cls_embedding = outputs.last_hidden_state[:, 0, :]
                normalized = F.normalize(cls_embedding, p=2, dim=1)
            
            return normalized.cpu().numpy()[0].tolist()
            
        except Exception as e:
            logger.error(f"Failed to generate embedding for text: {e}")
            return [0.0] * 768


class ModelManager(ModelManagerInterface):
    def __init__(self):
        self._llm: Optional[ChatOllama] = None
        self._embeddings: Optional[QwenEmbeddings] = None
        self._reranker: Optional[CrossEncoder] = None

    def get_llm(self) -> ChatOllama:
        if self._llm is None:
            self._llm = self._create_llm()
        return self._llm

    def get_embeddings(self) -> QwenEmbeddings:
        if self._embeddings is None:
            self._embeddings = self._create_embeddings()
        return self._embeddings

    def get_reranker(self) -> CrossEncoder:
        if self._reranker is None:
            self._reranker = self._create_reranker()
        return self._reranker

    def _create_llm(self) -> ChatOllama:
        try:
            llm = ChatOllama(
                model=settings.model.llm_model,
                base_url=settings.model.ollama_base_url,
                temperature=settings.model.llm_temperature,
                repeat_penalty=settings.model.llm_repeat_penalty,
                top_k=settings.model.llm_top_k,
                top_p=settings.model.llm_top_p,
            )

            # Skip test if in offline mode
            if not self._is_offline():
                try:
                    test_response = llm.invoke("Test connection")
                    logger.info(f"Ollama LLM connection successful: {settings.model.llm_model}")
                except Exception as e:
                    logger.warning(f"Ollama LLM test failed: {e}")

            return llm
        except Exception as e:
            logger.error(f"Failed to initialize Ollama LLM: {e}")
            raise ModelInitializationException(f"Failed to initialize Ollama LLM: {e}", e)

    def _create_embeddings(self) -> QwenEmbeddings:
        try:
            embeddings = QwenEmbeddings(
                model_path=settings.model.embedding_model,
                device=settings.model.embedding_device
            )

            try:
                test_embedding = embeddings.embed_query("test query")
                logger.info(f"Qwen embeddings initialized successfully: {settings.model.embedding_model}")
                logger.info(f"Embedding dimension: {len(test_embedding)}")
            except Exception as e:
                logger.warning(f"Qwen embedding test failed: {e}")

            return embeddings
        except Exception as e:
            logger.error(f"Failed to initialize Qwen embeddings: {e}")
            raise ModelInitializationException(f"Failed to initialize Qwen embeddings: {e}", e)

    def _create_reranker(self):
        """Create Qwen reranker or fallback to CrossEncoder"""
        try:
            from services.reranker import QwenReranker
            reranker = QwenReranker(settings.model.reranker_model, settings.model.reranker_device)
            
            try:
                test_scores = reranker.predict([["test query", "test document"]])
                logger.info(f"Initialized Qwen reranker: {settings.model.reranker_model}")
                logger.info(f"Qwen reranker test score: {test_scores[0]:.4f}")
            except Exception as e:
                logger.warning(f"Qwen reranker test failed: {e}")
                
            return reranker
        except Exception as e:
            logger.warning(f"Failed to initialize Qwen reranker: {e}")
            logger.info("Falling back to CrossEncoder reranker...")
            
            try:
                reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-12-v2")
                logger.info("Initialized fallback CrossEncoder reranker")
                return reranker
            except Exception as fallback_error:
                logger.error(f"Failed to initialize fallback reranker: {fallback_error}")
                raise ModelInitializationException(f"Failed to initialize any reranker: {fallback_error}", fallback_error)

    def _is_offline(self) -> bool:
        """Check if system is in offline mode"""
        try:
            import requests
            response = requests.get(f"{settings.model.ollama_base_url}/api/tags", timeout=2)
            return response.status_code != 200
        except Exception:
            return True