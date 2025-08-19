import logging
from typing import List, Dict, Any, Optional

from config.settings import settings
from graph.interfaces import GraphService
from graph.implementations import SemanticGraphService as BaseSemanticGraphService
from langchain_core.documents import Document

logger = logging.getLogger(__name__)


class SemanticGraphService(BaseSemanticGraphService):
    """Semantic Graph Service with configuration integration"""
    
    def __init__(self):
        # Use settings instead of config parameter
        super().__init__(self._create_config_object())
    
    def _create_config_object(self):
        """Create a config object from settings for backward compatibility"""
        class ConfigAdapter:
            def __init__(self):
                # Graph Configuration
                self.ENABLE_GRAPH_PROCESSING = settings.graph.enable_graph_processing
                self.GRAPH_STORE_DIRECTORY = settings.graph.graph_store_directory
                self.GRAPH_LLM_MODEL = settings.model.graph_llm_model
                self.OLLAMA_BASE_URL = settings.model.ollama_base_url
                self.EMBEDDING_MODEL = settings.model.embedding_model
                
                # Document processing limits
                self.MAX_DOCS_FOR_GRAPH = settings.document.max_docs_for_graph
                self.DOC_TRUNCATE_LENGTH = settings.document.doc_truncate_length
                self.MAX_COMBINED_CONTENT_LENGTH = settings.document.max_combined_content_length
                
                # Graph search parameters
                self.MAX_GRAPH_NODES = settings.graph.max_graph_nodes
                self.GRAPH_SEARCH_DEPTH = settings.graph.graph_search_depth
                self.SIMILARITY_THRESHOLD = settings.retrieval.similarity_threshold
                
        return ConfigAdapter()


class GraphServiceFactory:
    """Factory for creating graph services"""
    
    @staticmethod
    def create_graph_service() -> Optional[GraphService]:
        """Create a graph service based on configuration"""
        if not settings.graph.enable_graph_processing:
            logger.info("Graph processing disabled in configuration")
            return None
        
        try:
            return SemanticGraphService()
        except Exception as e:
            logger.error(f"Failed to create graph service: {e}")
            return None