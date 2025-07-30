import logging
import tempfile
import shutil
from pathlib import Path
from typing import List
from fastapi import HTTPException, status, UploadFile
from datetime import datetime

from config import Config
from models import (
    UploadResponse, ChatResponse, QueryRequest, ChatHistoryResponse,
    ChunkInfo, HealthResponse, GraphStats, SearchType
)
from interfaces import (
    DocumentProcessorInterface, VectorStoreInterface, RAGServiceInterface
)
from exceptions import ServiceException
from file_handlers import FileHandler
from application_state import ApplicationState

logger = logging.getLogger(__name__)

class UploadHandler:
    def __init__(
        self,
        config: Config,
        document_processor: DocumentProcessorInterface,
        vector_store_manager: VectorStoreInterface,
        rag_service: RAGServiceInterface,
        app_state: ApplicationState
    ):
        self.config = config
        self.document_processor = document_processor
        self.vector_store_manager = vector_store_manager
        self.rag_service = rag_service
        self.app_state = app_state
        self.file_handler = FileHandler(config)
    
    async def handle_upload(
        self,
        files: List[UploadFile],
        enable_graph_processing: bool = True
    ) -> UploadResponse:
        if not files:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No files provided"
            )
        
        self._validate_files(files)
        
        temp_dir = None
        try:
            file_paths = self.file_handler.process_uploaded_files(files)
            
            if not file_paths:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="No supported files found in uploaded files"
                )
            
            documents = self.document_processor.process_documents(file_paths)
            
            if not documents:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="No documents could be processed successfully"
                )
            
            is_initial_upload = not self.app_state.system_initialized
            
            if is_initial_upload:
                self.vector_store_manager.create_store(documents)
            else:
                success = self.vector_store_manager.add_documents(documents)
                if not success:
                    raise HTTPException(
                        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                        detail="Failed to add documents to vector store"
                    )
            
            self.rag_service.setup_chain()
            
            graph_stats = self._get_graph_stats(enable_graph_processing)
            
            self.app_state.system_initialized = True
            self.app_state.update_upload_stats(len(file_paths), len(documents))
            
            if graph_stats["graph_processed"]:
                self.app_state.graph_initialized = True
            
            action_text = "processed" if is_initial_upload else "added"
            logger.info(f"Successfully {action_text} {len(file_paths)} files into {len(documents)} chunks")
            
            return UploadResponse(
                success=True,
                message=f"Successfully {action_text} {len(file_paths)} files",
                file_count=len(file_paths),
                chunk_count=len(documents),
                **graph_stats
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
            if temp_dir and Path(temp_dir).exists():
                shutil.rmtree(temp_dir, ignore_errors=True)
    
    def _validate_files(self, files: List[UploadFile]):
        for file in files:
            if not self.file_handler.validate_file(file):
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Unsupported file type: {file.filename}"
                )
    
    def _get_graph_stats(self, enable_graph_processing: bool) -> dict:
        graph_processed = False
        graph_nodes = 0
        graph_relationships = 0
        
        if self.config.ENABLE_GRAPH_PROCESSING and enable_graph_processing and self.rag_service.graph_service:
            if self.rag_service.graph_service.has_data():
                graph_processed = True
                stats = self.rag_service.get_graph_stats()
                graph_nodes = stats.get("nodes", 0)
                graph_relationships = stats.get("relationships", 0)
        
        return {
            "graph_processed": graph_processed,
            "graph_nodes": graph_nodes,
            "graph_relationships": graph_relationships
        }


class ChatHandler:
    def __init__(
        self,
        rag_service: RAGServiceInterface,
        app_state: ApplicationState
    ):
        self.rag_service = rag_service
        self.app_state = app_state
    
    async def handle_chat(self, request: QueryRequest) -> ChatResponse:
        if not self.app_state.system_initialized:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="System not initialized. Please upload documents first."
            )
        
        if request.search_type == SearchType.GRAPH and not self.app_state.graph_initialized:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Graph search requested but no graph data available. Please upload documents with graph processing enabled."
            )
        
        try:
            self.app_state.add_user_message(request.question)
            
            result = self.rag_service.query(
                question=request.question,
                search_type=request.search_type,
                use_enhanced_query=request.use_enhanced_query,
                chat_history=self.app_state.chat_history
            )
            
            self.app_state.add_assistant_message(result["answer"])
            
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


class SystemHandler:
    def __init__(
        self,
        rag_service: RAGServiceInterface,
        vector_store_manager: VectorStoreInterface,
        app_state: ApplicationState
    ):
        self.rag_service = rag_service
        self.vector_store_manager = vector_store_manager
        self.app_state = app_state
    
    def get_chat_history(self) -> ChatHistoryResponse:
        return ChatHistoryResponse(
            messages=self.app_state.chat_history,
            total_messages=len(self.app_state.chat_history)
        )
    
    def clear_chat_history(self) -> dict:
        self.app_state.clear_history()
        self.rag_service.clear_memory()
        logger.info("Chat history cleared")
        return {"message": "Chat history cleared successfully"}
    
    def get_last_chunks(self) -> List[ChunkInfo]:
        if not self.app_state.system_initialized:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="System not initialized"
            )
        
        chunks = self.rag_service.get_last_chunks()
        
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
    
    def get_graph_stats(self) -> GraphStats:
        if not self.app_state.graph_initialized:
            return GraphStats(has_data=False)
        
        try:
            stats = self.rag_service.get_graph_stats()
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
    
    def get_health_status(self) -> HealthResponse:
        try:
            vector_store_status = "healthy" if self.vector_store_manager.vector_store else "not_initialized"
            graph_store_status = "healthy" if self.app_state.graph_initialized else "not_initialized"
            api_status = "healthy"
            
            return HealthResponse(
                status="healthy",
                system_initialized=self.app_state.system_initialized,
                chat_history_length=len(self.app_state.chat_history),
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
    
    def get_system_stats(self) -> dict:
        graph_stats = {}
        if self.app_state.graph_initialized:
            try:
                graph_stats = self.rag_service.get_graph_stats()
            except Exception as e:
                logger.error(f"Failed to get graph stats: {e}")
        
        return {
            "system_initialized": self.app_state.system_initialized,
            "graph_initialized": self.app_state.graph_initialized,
            "chat_history_length": len(self.app_state.chat_history),
            "last_upload_time": self.app_state.last_upload_time,
            "document_count": self.app_state.document_count,
            "chunk_count": self.app_state.chunk_count,
            "graph_stats": graph_stats,
            "search_types_available": [
                SearchType.VECTOR.value,
                SearchType.HYBRID.value if self.app_state.graph_initialized else None,
                SearchType.GRAPH.value if self.app_state.graph_initialized else None
            ],
            "uptime": datetime.now() - datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        }


class GraphHandler:
    def __init__(
        self,
        config: Config,
        rag_service: RAGServiceInterface,
        app_state: ApplicationState
    ):
        self.config = config
        self.rag_service = rag_service
        self.app_state = app_state
    
    async def handle_visualization(self, filename: str = "graph_visualization.html") -> dict:
        try:
            if not self.rag_service:
                raise HTTPException(status_code=503, detail="RAG service not initialized")
            
            if not self.rag_service.graph_service:
                raise HTTPException(status_code=404, detail="Graph service not available")
            
            if not self.rag_service.graph_service.has_data():
                raise HTTPException(status_code=404, detail="No graph data available for visualization")
            
            visualization_path = self.rag_service.visualize_graph(filename)
            
            if not visualization_path:
                raise HTTPException(status_code=500, detail="Failed to create graph visualization")
            
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
            raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")
