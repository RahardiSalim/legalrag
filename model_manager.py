import logging
from typing import Optional
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from sentence_transformers import CrossEncoder

from config import Config
from interfaces import ModelManagerInterface
from exceptions import ModelInitializationException

logger = logging.getLogger(__name__)


class ModelManager(ModelManagerInterface):
    def __init__(self, config: Config):
        self.config = config
        self._llm: Optional[ChatGoogleGenerativeAI] = None
        self._embeddings: Optional[GoogleGenerativeAIEmbeddings] = None
        self._reranker: Optional[CrossEncoder] = None
        
    def get_llm(self) -> ChatGoogleGenerativeAI:
        if self._llm is None:
            self._llm = self._create_llm()
        return self._llm
    
    def get_embeddings(self) -> GoogleGenerativeAIEmbeddings:
        if self._embeddings is None:
            self._embeddings = self._create_embeddings()
        return self._embeddings
    
    def get_reranker(self) -> CrossEncoder:
        if self._reranker is None:
            self._reranker = self._create_reranker()
        return self._reranker
    
    def _create_llm(self) -> ChatGoogleGenerativeAI:
        try:
            llm = ChatGoogleGenerativeAI(
                model=self.config.LLM_MODEL,
                google_api_key=self.config.GEMINI_API_KEY,
                safety_settings=self.config.SAFETY_SETTINGS,
                temperature=self.config.LLM_TEMPERATURE,
                max_output_tokens=65536
            )
            logger.info(f"Initialized LLM: {self.config.LLM_MODEL}")
            return llm
        except Exception as e:
            logger.error(f"Failed to initialize LLM: {e}")
            raise ModelInitializationException(f"Failed to initialize LLM: {e}", e)
    
    def _create_embeddings(self) -> GoogleGenerativeAIEmbeddings:
        try:
            embeddings = GoogleGenerativeAIEmbeddings(
                model=self.config.EMBEDDING_MODEL,
                google_api_key=self.config.GEMINI_API_KEY
            )
            logger.info(f"Initialized embeddings: {self.config.EMBEDDING_MODEL}")
            return embeddings
        except Exception as e:
            logger.error(f"Failed to initialize embeddings: {e}")
            raise ModelInitializationException(f"Failed to initialize embeddings: {e}", e)
    
    def _create_reranker(self) -> CrossEncoder:
        try:
            reranker = CrossEncoder(self.config.RERANKER_MODEL)
            logger.info(f"Initialized reranker: {self.config.RERANKER_MODEL}")
            return reranker
        except Exception as e:
            logger.error(f"Failed to initialize reranker: {e}")
            raise ModelInitializationException(f"Failed to initialize reranker: {e}", e)