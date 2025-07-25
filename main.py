from fastapi import FastAPI, File, UploadFile, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from typing import List, Optional
import tempfile
import os
import zipfile
import shutil
from pathlib import Path
from datetime import datetime
import logging

from config import Config
from models import (
    ChatMessage, ChatResponse, QueryRequest, UploadResponse, 
    ChatHistoryResponse, ChunkInfo, HealthResponse, ErrorResponse,
    SearchType, GraphStats
)
from services import ModelManager, DocumentProcessor, VectorStoreManager, RAGService, ServiceException

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
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

# Global instances
config = Config()
model_manager = ModelManager(config)
document_processor = DocumentProcessor(config)
vector_store_manager = VectorStoreManager(config, model_manager)
rag_service = RAGService(config, model_manager, vector_store_manager)

# Application state
class ApplicationState:
    def __init__(self):
        self.chat_history: List[ChatMessage] = []
        self.system_initialized: bool = False
        self.graph_initialized: bool = False
        self.last_upload_time: Optional[datetime] = None
        self.document_count: int = 0
        self.chunk_count: int = 0

app_state = ApplicationState()

# Exception handlers
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

# Startup event
@app.on_event("startup")
async def startup_event():
    """Initialize services on startup"""
    logger.info("🚀 Starting Legal RAG API with Graph Support...")
    
    try:
        # FIXED: Connect graph service to vector store manager FIRST
        if config.ENABLE_GRAPH_PROCESSING:
            vector_store_manager.graph_service = rag_service.graph_service
            
        # Try to load existing vector store
        if vector_store_manager.load_store():
            rag_service.setup_chain()
            app_state.system_initialized = True
            
            # FIXED: Try to load existing graph data
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

# Shutdown event
@app.on_event("shutdown")
async def shutdown_event():
    """Clean up resources on shutdown"""
    logger.info("🔄 Shutting down Legal RAG API...")
    
    # Clean up temporary files
    temp_dir = Path(config.TEMP_DIR)
    if temp_dir.exists():
        shutil.rmtree(temp_dir, ignore_errors=True)
    
    logger.info("✅ Shutdown complete")

# Helper functions
def _extract_files_from_zip(zip_path: str, extract_to: str) -> List[str]:
    """Extract files from ZIP archive"""
    extracted_files = []
    
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        for file_info in zip_ref.filelist:
            if not file_info.is_dir():
                file_extension = Path(file_info.filename).suffix.lower()
                if file_extension in config.SUPPORTED_EXTENSIONS:
                    extracted_path = zip_ref.extract(file_info, extract_to)
                    extracted_files.append(extracted_path)
    
    return extracted_files

def _save_uploaded_file(file: UploadFile, temp_dir: str) -> str:
    """Save uploaded file to temporary directory"""
    file_path = Path(temp_dir) / file.filename
    
    with open(file_path, 'wb') as f:
        content = file.file.read()
        f.write(content)
    
    return str(file_path)

def _validate_file(file: UploadFile) -> bool:
    """Validate uploaded file"""
    if not file.filename:
        return False
    
    file_extension = Path(file.filename).suffix.lower()
    return file_extension in config.SUPPORTED_EXTENSIONS or file_extension == '.zip'

# API Endpoints
@app.post("/upload", response_model=UploadResponse)
async def upload_files(
    files: List[UploadFile] = File(...),
    enable_graph_processing: bool = True
):
    """Upload and process documents with optional graph processing"""
    if not files:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No files provided"
        )
    
    # Validate files
    for file in files:
        if not _validate_file(file):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Unsupported file type: {file.filename}"
            )
    
    temp_dir = None
    try:
        # Create temporary directory
        temp_dir = tempfile.mkdtemp(dir=config.TEMP_DIR)
        file_paths = []
        
        # Process each uploaded file
        for file in files:
            file_path = _save_uploaded_file(file, temp_dir)
            
            if file.filename.endswith('.zip'):
                # Extract ZIP contents
                extracted_files = _extract_files_from_zip(file_path, temp_dir)
                file_paths.extend(extracted_files)
            else:
                file_paths.append(file_path)
        
        if not file_paths:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No supported files found in uploaded files"
            )
        
        # Process documents
        documents = document_processor.process_documents(file_paths)
        
        if not documents:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No documents could be processed successfully"
            )
        
        if config.ENABLE_GRAPH_PROCESSING and enable_graph_processing and rag_service.graph_service:
            vector_store_manager.graph_service = rag_service.graph_service
        
        # Handle vector store creation/update
        is_initial_upload = not app_state.system_initialized
        
        if is_initial_upload:
            # Create new vector store - this will automatically process graph
            vector_store_manager.create_store(documents)
        else:
            # Update existing vector store - this will automatically update graph incrementally
            success = vector_store_manager.add_documents(documents)
            if not success:
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail="Failed to add documents to vector store"
                )
            
        # Setup RAG chain
        rag_service.setup_chain()
        
        # The graph processing now happens automatically in vector_store_manager
        # But we still need to track the results for the response
        graph_processed = False
        graph_nodes = 0
        graph_relationships = 0
        
        if config.ENABLE_GRAPH_PROCESSING and enable_graph_processing and rag_service.graph_service:
            if rag_service.graph_service.has_graph_data():
                graph_processed = True
                graph_stats = rag_service.get_graph_stats()
                graph_nodes = graph_stats.get("nodes", 0)
                graph_relationships = graph_stats.get("relationships", 0)
                app_state.graph_initialized = True
                
                action = "created" if is_initial_upload else "updated"
                logger.info(f"Graph {action}: {graph_nodes} nodes, {graph_relationships} relationships")
        
        # Update application state
        app_state.system_initialized = True
        app_state.last_upload_time = datetime.now()
        app_state.document_count += len(file_paths)
        app_state.chunk_count += len(documents)
        
        action_text = "processed" if is_initial_upload else "added"
        logger.info(f"Successfully {action_text} {len(file_paths)} files into {len(documents)} chunks")
        
        return UploadResponse(
            success=True,
            message=f"Successfully {action_text} {len(file_paths)} files",
            file_count=len(file_paths),
            chunk_count=len(documents),
            graph_processed=graph_processed,
            graph_nodes=graph_nodes,
            graph_relationships=graph_relationships
        )
        
    except ServiceException as e:
        logger.error(f"Service error during upload: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Unexpected error during upload: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error processing files: {str(e)}"
        )
    finally:
        # Clean up temporary directory
        if temp_dir and Path(temp_dir).exists():
            shutil.rmtree(temp_dir, ignore_errors=True)

@app.post("/chat", response_model=ChatResponse)
async def chat(request: QueryRequest):
    """Chat with the RAG system using different search types"""
    if not app_state.system_initialized:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="System not initialized. Please upload documents first."
        )
    
    # Check if graph search is requested but not available
    if request.search_type == SearchType.GRAPH and not app_state.graph_initialized:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Graph search requested but no graph data available. Please upload documents with graph processing enabled."
        )
    
    try:
        # Add user message to history
        user_message = ChatMessage(
            role="user",
            content=request.question,
            timestamp=datetime.now()
        )
        app_state.chat_history.append(user_message)
        
        # Query RAG system
        result = rag_service.query(
            question=request.question,
            search_type=request.search_type,
            use_enhanced_query=request.use_enhanced_query,
            chat_history=app_state.chat_history
        )
        
        # Add assistant message to history
        assistant_message = ChatMessage(
            role="assistant",
            content=result["answer"],
            timestamp=datetime.now()
        )
        app_state.chat_history.append(assistant_message)
        
        logger.info(f"Query processed successfully in {result.get('processing_time', 0):.2f}s using {result.get('search_type_used', 'unknown')} search")
        
        return ChatResponse(
            answer=result["answer"],
            source_documents=result["source_documents"],
            generated_question=result.get("generated_question"),
            enhanced_query=request.use_enhanced_query,
            processing_time=result.get("processing_time"),
            tokens_used=result.get("tokens_used"),
            search_type_used=result.get("search_type_used", SearchType.VECTOR),
            graph_entities=result.get("graph_entities", []),
            graph_relationships=result.get("graph_relationships", [])
        )
        
    except ServiceException as e:
        logger.error(f"Service error during chat: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Unexpected error during chat: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error processing query: {str(e)}"
        )

@app.get("/history", response_model=ChatHistoryResponse)
async def get_chat_history():
    """Get chat history"""
    return ChatHistoryResponse(
        messages=app_state.chat_history,
        total_messages=len(app_state.chat_history)
    )

@app.post("/clear-history")
async def clear_chat_history():
    """Clear chat history"""
    app_state.chat_history.clear()
    rag_service.clear_memory()
    
    logger.info("Chat history cleared")
    return {"message": "Chat history cleared successfully"}

@app.get("/chunks", response_model=List[ChunkInfo])
async def get_last_chunks():
    """Get chunks used in the last query"""
    if not app_state.system_initialized:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="System not initialized"
        )
    
    chunks = rag_service.get_last_chunks()
    
    return [
        ChunkInfo(
            content=chunk.page_content,
            metadata=chunk.metadata,
            score=chunk.metadata.get("score"),
            rerank_score=chunk.metadata.get("rerank_score"),
            chunk_id=chunk.metadata.get("chunk_id")
        )
        for chunk in chunks
    ]

@app.get("/graph/stats", response_model=GraphStats)
async def get_graph_stats():
    """Get graph statistics"""
    if not app_state.graph_initialized:
        return GraphStats(has_data=False)
    
    try:
        stats = rag_service.get_graph_stats()
        return GraphStats(
            nodes=stats.get("nodes", 0),
            relationships=stats.get("relationships", 0),
            node_types=stats.get("node_types", []),
            relationship_types=stats.get("relationship_types", []),
            has_data=stats.get("nodes", 0) > 0
        )
    except Exception as e:
        logger.error(f"Failed to get graph stats: {e}")
        return GraphStats(has_data=False)

@app.get("/graph/visualize")
async def visualize_graph_get(filename: str = "graph_visualization.html"):
    """Create graph visualization via GET request"""
    return await visualize_graph_internal(filename)

@app.post("/graph/visualize") 
async def visualize_graph_post(request: dict = None):
    """Create graph visualization via POST request"""
    filename = "graph_visualization.html"
    if request and "filename" in request:
        filename = request["filename"]
    return await visualize_graph_internal(filename)

async def visualize_graph_internal(filename: str):
    """Internal function to handle graph visualization"""
    try:
        if not rag_service:
            raise HTTPException(status_code=503, detail="RAG service not initialized")
        
        if not rag_service.graph_service:
            raise HTTPException(status_code=404, detail="Graph service not available")
        
        if not rag_service.graph_service.has_graph_data():
            raise HTTPException(status_code=404, detail="No graph data available for visualization")
        
        visualization_path = rag_service.visualize_graph(filename)
        
        if not visualization_path:
            raise HTTPException(status_code=500, detail="Failed to create graph visualization")
        
        # Check if file actually exists
        if not Path(visualization_path).exists():
            raise HTTPException(status_code=500, detail="Visualization file was not created")
        
        return {
            "message": "Graph visualization created successfully",
            "file_path": visualization_path,
            "filename": filename,
            "success": True
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Graph visualization endpoint error: {e}")
        import traceback
        logger.error(f"Full traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")
    
@app.get("/graph/visualize/{filename}")
async def serve_graph_visualization(filename: str):
    """Serve graph visualization file"""
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
    """Health check endpoint"""
    try:
        # Check vector store status
        vector_store_status = "healthy" if vector_store_manager.vector_store else "not_initialized"
        
        # Check graph store status
        graph_store_status = "healthy" if app_state.graph_initialized else "not_initialized"
        
        # Check API status
        api_status = "healthy"
        
        return HealthResponse(
            status="healthy",
            system_initialized=app_state.system_initialized,
            chat_history_length=len(app_state.chat_history),
            vector_store_status=vector_store_status,
            graph_store_status=graph_store_status,
            api_status=api_status
        )
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return HealthResponse(
            status="unhealthy",
            system_initialized=False,
            chat_history_length=0,
            vector_store_status="error",
            graph_store_status="error",
            api_status="error"
        )

@app.get("/stats")
async def get_system_stats():
    """Get system statistics"""
    graph_stats = {}
    if app_state.graph_initialized:
        try:
            graph_stats = rag_service.get_graph_stats()
        except Exception as e:
            logger.error(f"Failed to get graph stats: {e}")
    
    return {
        "system_initialized": app_state.system_initialized,
        "graph_initialized": app_state.graph_initialized,
        "chat_history_length": len(app_state.chat_history),
        "last_upload_time": app_state.last_upload_time,
        "document_count": app_state.document_count,
        "chunk_count": app_state.chunk_count,
        "graph_stats": graph_stats,
        "search_types_available": [
            SearchType.VECTOR.value,
            SearchType.HYBRID.value if app_state.graph_initialized else None,
            SearchType.GRAPH.value if app_state.graph_initialized else None
        ],
        "uptime": datetime.now() - datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    }

# Run the app
if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info",
        reload=False
    )