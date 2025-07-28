import logging
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from sentence_transformers import CrossEncoder

from config import Config
from interfaces import ModelManagerInterface
from exceptions import ServiceException

logger = logging.getLogger(__name__)

class ModelManager(ModelManagerInterface):
    def __init__(self, config: Config):
        self.config = config
        self._llm = None
        self._embeddings = None
        self._reranker = None
        
    def get_llm(self) -> ChatGoogleGenerativeAI:
        if self._llm is None:
            self._llm = self._initialize_llm()
        return self._llm
    
    def get_embeddings(self) -> GoogleGenerativeAIEmbeddings:
        if self._embeddings is None:
            self._embeddings = self._initialize_embeddings()
        return self._embeddings
    
    def get_reranker(self) -> CrossEncoder:
        if self._reranker is None:
            self._reranker = self._initialize_reranker()
        return self._reranker
    
    def _initialize_llm(self) -> ChatGoogleGenerativeAI:
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
            raise ServiceException(f"Failed to initialize LLM: {e}")
    
    def _initialize_embeddings(self) -> GoogleGenerativeAIEmbeddings:
        try:
            embeddings = GoogleGenerativeAIEmbeddings(
                model=self.config.EMBEDDING_MODEL,
                google_api_key=self.config.GEMINI_API_KEY
            )
            logger.info(f"Initialized embeddings: {self.config.EMBEDDING_MODEL}")
            return embeddings
        except Exception as e:
            logger.error(f"Failed to initialize embeddings: {e}")
            raise ServiceException(f"Failed to initialize embeddings: {e}")
    
    def _initialize_reranker(self) -> CrossEncoder:
        try:
            reranker = CrossEncoder(self.config.RERANKER_MODEL)
            logger.info(f"Initialized reranker: {self.config.RERANKER_MODEL}")
            return reranker
        except Exception as e:
            logger.error(f"Failed to initialize reranker: {e}")
            raise ServiceException(f"Failed to initialize reranker: {e}")