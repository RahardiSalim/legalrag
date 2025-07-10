from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Optional
import tempfile
import os
import zipfile
from pathlib import Path
from datetime import datetime

from config import Config
from models import ChatMessage, ChatResponse, QueryRequest, UploadResponse, ChatHistoryResponse, ChunkInfo
from services import ModelManager, DocumentProcessor, VectorStoreManager, RAGService

app = FastAPI(title="Legal RAG API", version="1.0.0")

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

# Chat history storage (in production, use a proper database)
chat_history: List[ChatMessage] = []
system_initialized = False


@app.on_event("startup")
async def startup_event():
    """Initialize services on startup"""
    print("🚀 Starting Legal RAG API...")


@app.post("/upload", response_model=UploadResponse)
async def upload_files(files: List[UploadFile] = File(...)):
    """Upload and process documents"""
    global system_initialized
    
    try:
        temp_dir = tempfile.mkdtemp()
        file_paths = []
        
        for file in files:
            if file.filename.endswith(('.pdf', '.txt', '.zip')):
                temp_path = os.path.join(temp_dir, file.filename)
                
                with open(temp_path, 'wb') as f:
                    content = await file.read()
                    f.write(content)
                
                # Handle ZIP files
                if file.filename.endswith('.zip'):
                    with zipfile.ZipFile(temp_path, 'r') as zip_ref:
                        zip_ref.extractall(temp_dir)
                    
                    # Find extracted files
                    for root, dirs, files in os.walk(temp_dir):
                        for extracted_file in files:
                            if extracted_file.endswith(('.pdf', '.txt')):
                                file_paths.append(os.path.join(root, extracted_file))
                else:
                    file_paths.append(temp_path)
        
        if not file_paths:
            raise HTTPException(status_code=400, detail="No supported files found")
        
        # Process documents
        documents = document_processor.process_documents(file_paths)
        
        if not documents:
            raise HTTPException(status_code=400, detail="No documents could be processed")
        
        # Create vector store
        vector_store_manager.create_store(documents)
        
        # Setup RAG chain
        rag_service.setup_chain()
        
        system_initialized = True
        
        return UploadResponse(
            success=True,
            message=f"Successfully processed {len(file_paths)} files",
            file_count=len(file_paths),
            chunk_count=len(documents)
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing files: {str(e)}")


@app.post("/chat", response_model=ChatResponse)
async def chat(request: QueryRequest):
    """Chat with the RAG system"""
    global system_initialized
    
    if not system_initialized:
        raise HTTPException(status_code=400, detail="System not initialized. Please upload documents first.")
    
    try:
        # Add user message to history
        user_message = ChatMessage(role="user", content=request.question)
        chat_history.append(user_message)
        
        # Query RAG system
        result = rag_service.query(request.question, request.use_enhanced_query)
        
        # Add assistant message to history
        assistant_message = ChatMessage(role="assistant", content=result["answer"])
        chat_history.append(assistant_message)
        
        return ChatResponse(
            answer=result["answer"],
            source_documents=result["source_documents"],
            generated_question=result.get("generated_question"),
            enhanced_query=request.use_enhanced_query
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")


@app.get("/history", response_model=ChatHistoryResponse)
async def get_chat_history():
    """Get chat history"""
    return ChatHistoryResponse(messages=chat_history)


@app.post("/clear-history")
async def clear_chat_history():
    """Clear chat history"""
    global chat_history
    chat_history = []
    
    # Clear RAG service memory
    rag_service.clear_memory()
    
    return {"message": "Chat history cleared"}


@app.get("/chunks", response_model=List[ChunkInfo])
async def get_last_chunks():
    """Get chunks used in the last query"""
    chunks = rag_service.get_last_chunks()
    
    return [
        ChunkInfo(
            content=chunk.page_content,
            metadata=chunk.metadata
        )
        for chunk in chunks
    ]


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "system_initialized": system_initialized,
        "chat_history_length": len(chat_history)
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)