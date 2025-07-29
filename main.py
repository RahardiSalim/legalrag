from fastapi import FastAPI, File, UploadFile, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from typing import List
import tempfile
import shutil
from pathlib import Path
from datetime import datetime
import logging

from config import Config
from models import (
    QueryRequest, UploadResponse, ChatResponse, ChatHistoryResponse,
    ChunkInfo, HealthResponse, ErrorResponse, GraphStats
)
from exceptions import ServiceException
from model_manager import ModelManager
from document_processor import DocumentProcessor
from vector_store_manager import VectorStoreManager
from rag_service import RAGService
from application_state import ApplicationState
from api_handlers import UploadHandler, ChatHandler, SystemHandler, GraphHandler

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Legal RAG API with Graph Support",
    description="Advanced RAG system for legal document processing with knowledge graph capabilities",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

config = Config()
model_manager = ModelManager(config)
document_processor = DocumentProcessor(config)
vector_store_manager = VectorStoreManager(config, model_manager)
rag_service = RAGService(config, model_manager, vector_store_manager)

app_state = ApplicationState()
upload_handler = UploadHandler(config, document_processor, vector_store_manager, rag_service, app_state)
chat_handler = ChatHandler(rag_service, app_state)
system_handler = SystemHandler(rag_service, vector_store_manager, app_state)
graph_handler = GraphHandler(config, rag_service, app_state)

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
        if config.ENABLE_GRAPH_PROCESSING:
            vector_store_manager.graph_service = rag_service.graph_service
            
        if vector_store_manager.load_store():
            rag_service.setup_chain()
            app_state.system_initialized = True
            
            if rag_service.graph_service:
                if rag_service.graph_service.load_graph_data():
                    app_state.graph_initialized = True
                    logger.info("✅ System initialized with existing database and graph data")
                else:
                    logger.info("✅ System initialized with existing database (no graph data)")
            else:
                logger.info("✅ System initialized with existing database (graph processing disabled)")
        else:
            logger.info("⚠️ System not initialized. Please upload documents to create a new database")
            
    except Exception as e:
        logger.error(f"Failed to initialize system: {e}")

@app.on_event("shutdown")
async def shutdown_event():
    logger.info("🔄 Shutting down Legal RAG API...")
    
    temp_dir = Path(config.TEMP_DIR)
    if temp_dir.exists():
        shutil.rmtree(temp_dir, ignore_errors=True)
    
    logger.info("✅ Shutdown complete")

@app.post("/upload", response_model=UploadResponse)
async def upload_files(
    files: List[UploadFile] = File(...),
    enable_graph_processing: bool = True
):
    return await upload_handler.handle_upload(files, enable_graph_processing)

@app.post("/chat", response_model=ChatResponse)
async def chat(request: QueryRequest):
    return await chat_handler.handle_chat(request)

@app.get("/history", response_model=ChatHistoryResponse)
async def get_chat_history():
    return system_handler.get_chat_history()

@app.post("/clear-history")
async def clear_chat_history():
    return system_handler.clear_chat_history()

@app.get("/chunks", response_model=List[ChunkInfo])
async def get_last_chunks():
    return system_handler.get_last_chunks()

@app.get("/graph/stats", response_model=GraphStats)
async def get_graph_stats():
    return system_handler.get_graph_stats()

@app.get("/graph/visualize")
async def visualize_graph_get(filename: str = "graph_visualization.html"):
    return await graph_handler.handle_visualization(filename)

@app.post("/graph/visualize") 
async def visualize_graph_post(request: dict = None):
    filename = "graph_visualization.html"
    if request and "filename" in request:
        filename = request["filename"]
    return await graph_handler.handle_visualization(filename)

@app.get("/graph/visualize/{filename}")
async def serve_graph_visualization(filename: str):
    file_path = Path(config.GRAPH_STORE_DIRECTORY) / filename
    
    if not file_path.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Visualization file not found"
        )
    
    return FileResponse(
        path=str(file_path),
        media_type="text/html",
        filename=filename
    )

@app.get("/health", response_model=HealthResponse)
async def health_check():
    return system_handler.get_health_status()

@app.get("/stats")
async def get_system_stats():
    return system_handler.get_system_stats()

if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info",
        reload=False
    )