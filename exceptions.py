class ServiceException(Exception):
    """Base exception for service errors"""
    def __init__(self, message: str, cause: Exception = None):
        super().__init__(message)
        self.cause = cause


class DocumentProcessingException(ServiceException):
    """Exception for document processing errors"""
    pass


class GraphServiceException(ServiceException):
    """Exception for graph service errors"""
    pass


class VectorStoreException(ServiceException):
    """Exception for vector store errors"""
    pass


class RAGException(ServiceException):
    """Exception for RAG processing errors"""
    pass


class ConfigurationException(ServiceException):
    """Exception for configuration errors"""
    pass


class ModelInitializationException(ServiceException):
    """Exception for model initialization errors"""
    pass