class ServiceException(Exception):
    """Base exception for service errors"""
    pass

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