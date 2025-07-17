from fastapi import FastAPI, File, UploadFile, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
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
    ChatHistoryResponse, ChunkInfo, HealthResponse, ErrorResponse
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
    title="Legal RAG API",
    description="Advanced RAG system for legal document processing and query answering",
    version="1.0.0",
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
    logger.info("🚀 Starting Legal RAG API...")
    
    try:
        # Try to load existing vector store
        if vector_store_manager.load_store():
            rag_service.setup_chain()
            app_state.system_initialized = True
            logger.info("✅ System initialized from existing database")
        else:
            logger.info("⚠️ System not initialized. Please upload documents to create a new database")
            
    except Exception as e:
        logger.error(f"Failed to initialize system: {e}")
        # Continue startup even if initialization fails

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
async def upload_files(files: List[UploadFile] = File(...)):
    """Upload and process documents"""
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
        
        # Create vector store
        vector_store_manager.create_store(documents)
        
        # Setup RAG chain
        rag_service.setup_chain()
        
        # Update application state
        app_state.system_initialized = True
        app_state.last_upload_time = datetime.now()
        app_state.document_count = len(file_paths)
        app_state.chunk_count = len(documents)
        
        logger.info(f"Successfully processed {len(file_paths)} files into {len(documents)} chunks")
        
        return UploadResponse(
            success=True,
            message=f"Successfully processed {len(file_paths)} files",
            file_count=len(file_paths),
            chunk_count=len(documents)
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
    """Chat with the RAG system"""
    if not app_state.system_initialized:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="System not initialized. Please upload documents first."
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
        
        logger.info(f"Query processed successfully in {result.get('processing_time', 0):.2f}s")
        
        return ChatResponse(
            answer=result["answer"],
            source_documents=result["source_documents"],
            generated_question=result.get("generated_question"),
            enhanced_query=request.use_enhanced_query,
            processing_time=result.get("processing_time"),
            tokens_used=result.get("tokens_used")
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

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    try:
        # Check vector store status
        vector_store_status = "healthy" if vector_store_manager.vector_store else "not_initialized"
        
        # Check API status
        api_status = "healthy"
        
        return HealthResponse(
            status="healthy",
            system_initialized=app_state.system_initialized,
            chat_history_length=len(app_state.chat_history),
            vector_store_status=vector_store_status,
            api_status=api_status
        )
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return HealthResponse(
            status="unhealthy",
            system_initialized=False,
            chat_history_length=0,
            vector_store_status="error",
            api_status="error"
        )

@app.get("/stats")
async def get_system_stats():
    """Get system statistics"""
    return {
        "system_initialized": app_state.system_initialized,
        "chat_history_length": len(app_state.chat_history),
        "last_upload_time": app_state.last_upload_time,
        "document_count": app_state.document_count,
        "chunk_count": app_state.chunk_count,
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