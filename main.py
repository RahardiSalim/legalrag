# DISABLE ALL ANALYTICS FIRST - Import configuration to handle this
from config.settings import settings  # This handles analytics disabling

from fastapi import FastAPI, File, UploadFile, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from typing import List
import tempfile
import shutil
from pathlib import Path
from datetime import datetime
import logging

# Import refactored modules
from models.api_models import (
    QueryRequest, UploadResponse, ChatResponse,
    ChunkInfo, HealthResponse, ErrorResponse, GraphStats, FeedbackRequest,
    FeedbackResponse, FeedbackStatsResponse, EnhancedChatResponse
)
from models.chat_models import (
    ChatMessage, ChatHistoryResponse
)
from services.feedback_system import EnhancedRAGService
from core.exceptions import ServiceException
from services.model_manager import ModelManager
from services.document_processor import DocumentProcessor
from services.vector_store import VectorStoreManager
from services.rag_service import RAGService
from storage.application_state import ApplicationState
from graph.services import GraphServiceFactory

# Configure logging
logging.basicConfig(
    level=getattr(logging, settings.system.log_level),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="Legal RAG API with Graph Support",
    description="Advanced RAG system for legal document processing with knowledge graph capabilities",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize services
model_manager = ModelManager()
document_processor = DocumentProcessor()
vector_store_manager = VectorStoreManager(model_manager)
rag_service = RAGService(model_manager, vector_store_manager)

# Initialize application state
app_state = ApplicationState()

# Initialize graph service
graph_service = GraphServiceFactory.create_graph_service()
if graph_service:
    vector_store_manager.graph_service = graph_service
    rag_service.graph_service = graph_service

# Initialize enhanced services
enhanced_rag_service = EnhancedRAGService(rag_service)

# Note: API handlers would be imported here
# from api.handlers import UploadHandler, ChatHandler, etc.

@app.exception_handler(ServiceException)
async def service_exception_handler(request, exc: ServiceException):
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=ErrorResponse(
            error="Service Error",
            detail=str(exc)
        ).dict()
    )

@app.exception_handler(Exception)
async def general_exception_handler(request, exc: Exception):
    logger.error(f"Unexpected error: {exc}")
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=ErrorResponse(
            error="Internal Server Error",
            detail="An unexpected error occurred"
        ).dict()
    )

@app.on_event("startup")
async def startup_event():
    logger.info("🚀 Starting Legal RAG API with Graph Support...")
    
    try:
        if vector_store_manager.load_store():
            rag_service.setup_chain()
            app_state.system_initialized = True
            
            if graph_service and graph_service.has_data():
                app_state.graph_initialized = True
                logger.info("✅ System initialized with existing database and graph data")
            else:
                logger.info("✅ System initialized with existing database (no graph data)")
        else:
            logger.info("⚠️ System not initialized. Please upload documents to create a new database")
            
    except Exception as e:
        logger.error(f"Failed to initialize system: {e}")

@app.on_event("shutdown")
async def shutdown_event():
    logger.info("🔥 Shutting down Legal RAG API...")
    
    temp_dir = Path(settings.storage.temp_dir)
    if temp_dir.exists():
        shutil.rmtree(temp_dir, ignore_errors=True)
    
    logger.info("✅ Shutdown complete")

# Health check endpoint
@app.get("/health", response_model=HealthResponse)
async def health_check():
    return HealthResponse(
        status="healthy" if app_state.system_initialized else "not_initialized",
        system_initialized=app_state.system_initialized,
        chat_history_length=len(app_state.chat_history),
        vector_store_status="loaded" if app_state.system_initialized else "not_loaded",
        graph_store_status="loaded" if app_state.graph_initialized else "not_loaded",
        api_status="running"
    )

# Basic stats endpoint
@app.get("/stats")
async def get_system_stats():
    stats = app_state.get_system_stats()
    stats.update({
        "configuration": {
            "graph_processing_enabled": settings.graph.enable_graph_processing,
            "chunk_size": settings.document.chunk_size,
            "search_k": settings.retrieval.search_k,
            "rerank_k": settings.retrieval.rerank_k
        }
    })
    return stats

if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        app,
        host=settings.system.api_host,
        port=settings.system.api_port,
        log_level=settings.system.log_level.lower(),
        reload=settings.system.api_reload
    )