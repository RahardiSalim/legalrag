import time
import logging
from typing import List, Dict, Any, Optional
from pathlib import Path
from datetime import datetime
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferWindowMemory
from langchain.memory import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain.prompts import PromptTemplate
from langchain.schema import Document

from config.settings import settings
from core.interfaces import RAGServiceInterface, ModelManagerInterface, VectorStoreInterface
from models.api_models import SearchType, SourceDocument, GraphEntity, GraphRelationship
from core.exceptions import RAGException
from graph.interfaces import GraphService
from graph.services import GraphServiceFactory
from utils.formatters import ContextFormatter, QueryEnhancer

logger = logging.getLogger(__name__)


class RAGService(RAGServiceInterface):
    def __init__(self, model_manager: ModelManagerInterface, vector_store_manager: VectorStoreInterface):
        self.model_manager = model_manager
        self.vector_store_manager = vector_store_manager
        self.graph_service: Optional[GraphService] = self._initialize_graph_service()
        
        self.context_formatter = ContextFormatter()
        self.query_enhancer = QueryEnhancer(model_manager)
        
        self.last_retrieved_chunks = []
        self.chain = None
        self.memory = None
        
        # Link graph service to vector store manager
        if self.graph_service:
            vector_store_manager.graph_service = self.graph_service

    def setup_chain(self) -> None:
        """Initialize the conversational retrieval chain"""
        try:
            logger.info("Setting up RAG chain...")
            
            # Create a simple in-memory chat history
            self.chat_history = ChatMessageHistory()
            
            # Use the simpler memory approach
            self.memory = ConversationBufferWindowMemory(
                memory_key="chat_history",
                return_messages=True,
                output_key="answer",
                k=settings.memory.conversation_window_size,
                chat_memory=self.chat_history
            )
            
            qa_prompt = PromptTemplate(
                template=settings.prompt_templates["qa_template"],
                input_variables=["context", "question"]
            )
            
            self.chain = ConversationalRetrievalChain.from_llm(
                llm=self.model_manager.get_llm(),
                retriever=self.vector_store_manager.get_retriever(),
                memory=self.memory,
                return_source_documents=True,
                return_generated_question=True,
                combine_docs_chain_kwargs={"prompt": qa_prompt},
                verbose=False
            )

            logger.info("RAG chain setup completed successfully")
            
        except Exception as e:
            logger.error(f"Failed to setup RAG chain: {e}")
            raise RAGException(f"Failed to setup RAG chain: {e}")
        
    def query(self, question: str, search_type: SearchType = SearchType.VECTOR, 
              use_enhanced_query: bool = False, chat_history: List = None) -> Dict[str, Any]:
        """Process a query using the specified search type"""
        if not self.chain:
            raise RAGException("RAG chain not initialized. Call setup_chain() first.")
        
        start_time = time.time()
        
        try:
            logger.info(f"Processing query with {search_type.value} search: {question[:50]}...")
            
            if search_type == SearchType.VECTOR:
                return self._query_vector(question, use_enhanced_query, chat_history, start_time)
            elif search_type == SearchType.GRAPH:
                return self._query_graph(question, use_enhanced_query, chat_history, start_time)
            elif search_type == SearchType.HYBRID:
                return self._query_hybrid(question, use_enhanced_query, chat_history, start_time)
            else:
                raise RAGException(f"Unsupported search type: {search_type}")
                
        except Exception as e:
            logger.error(f"Query processing failed: {e}")
            raise RAGException(f"Query processing failed: {e}")
    
    def clear_memory(self) -> None:
        """Clear conversation memory"""
        if self.memory:
            self.memory.clear()
            logger.info("Conversation memory cleared")
    
    def get_last_chunks(self) -> List[Document]:
        """Get the last retrieved document chunks"""
        return self.last_retrieved_chunks
    
    def get_graph_stats(self) -> Dict[str, Any]:
        """Get graph statistics"""
        if not self.graph_service:
            return {"has_graph": False, "nodes": 0, "relationships": 0}
        
        stats = self.graph_service.get_stats()
        stats["has_graph"] = True
        return stats
    
    def visualize_graph(self, filename: str = "graph_visualization.html") -> str:
        """Create graph visualization"""
        if not self.graph_service:
            raise RAGException("Graph service not initialized")
        
        return self.graph_service.visualize(filename)
    
    def _initialize_graph_service(self) -> Optional[GraphService]:
        """Initialize graph service if enabled"""
        if not settings.graph.enable_graph_processing:
            logger.info("Graph processing disabled in configuration")
            return None
        
        try:
            return GraphServiceFactory.create_graph_service()
        except Exception as e:
            logger.error(f"Failed to initialize graph service: {e}")
            return None
    
    def _query_vector(self, question: str, use_enhanced_query: bool, chat_history: List, start_time: float) -> Dict[str, Any]:
        """Process vector-based query"""
        processed_question = self._get_processed_question(question, use_enhanced_query, chat_history)
        
        # Use invoke instead of deprecated __call__
        result = self.chain.invoke({"question": processed_question})
        self.last_retrieved_chunks = result.get("source_documents", [])

        # Add relevance scores to chunks metadata
        if self.last_retrieved_chunks:
            for i, doc in enumerate(self.last_retrieved_chunks):
                if "score" not in doc.metadata:
                    doc.metadata["score"] = 1.0 - (i * 0.1)  # Simple relevance scoring
        
        source_documents = [
            SourceDocument(
                content=doc.page_content,
                metadata=doc.metadata,
                score=doc.metadata.get("score"),
                rerank_score=doc.metadata.get("rerank_score")
            )
            for doc in self.last_retrieved_chunks
        ]
        
        return {
            "answer": result["answer"],
            "source_documents": source_documents,
            "generated_question": result.get("generated_question"),
            "enhanced_query": use_enhanced_query,
            "search_type_used": SearchType.VECTOR,
            "processing_time": time.time() - start_time,
            "graph_entities": [],
            "graph_relationships": []
        }
    
    def _query_graph(self, question: str, use_enhanced_query: bool, chat_history: List, start_time: float) -> Dict[str, Any]:
        """Process graph-based query"""
        if not self.graph_service or not self.graph_service.has_data():
            raise RAGException("Graph service not available or no graph data exists")
        
        processed_question = self._get_processed_question(question, use_enhanced_query, chat_history)
        
        try:
            graph_result = self.graph_service.search(processed_question)
            
            graph_prompt = PromptTemplate(
                template=settings.prompt_templates["graph_qa_template"],
                input_variables=["graph_context", "question"]
            )
            
            llm = self.model_manager.get_llm()
            formatted_prompt = graph_prompt.format(
                graph_context=graph_result.context,
                question=processed_question
            )
            
            response = llm.invoke(formatted_prompt)
            
            # Convert graph entities and relationships
            graph_entities = [
                GraphEntity(
                    id=entity["id"],
                    type=entity["type"],
                    properties=entity.get("properties", {})
                )
                for entity in graph_result.entities
            ]
            
            graph_relationships = [
                GraphRelationship(
                    source=rel["source"],
                    target=rel["target"],
                    type=rel["type"],
                    properties=rel.get("properties", {})
                )
                for rel in graph_result.relationships
            ]
            
            return {
                "answer": response.content,
                "source_documents": [],
                "generated_question": processed_question if use_enhanced_query else None,
                "enhanced_query": use_enhanced_query,
                "search_type_used": SearchType.GRAPH,
                "processing_time": time.time() - start_time,
                "graph_entities": graph_entities,
                "graph_relationships": graph_relationships
            }
            
        except Exception as e:
            logger.error(f"Graph query failed: {e}")
            # Fallback to vector search
            logger.info("Falling back to vector search due to graph query failure")
            return self._query_vector(question, use_enhanced_query, chat_history, start_time)
    
    def _query_hybrid(self, question: str, use_enhanced_query: bool, chat_history: List, start_time: float) -> Dict[str, Any]:
        """Process hybrid query combining vector and graph search"""
        # Start with vector search
        vector_result = self._query_vector(question, use_enhanced_query, chat_history, start_time)
        
        graph_entities = []
        graph_relationships = []
        
        # Enhance with graph data if available
        if self.graph_service and self.graph_service.has_data():
            try:
                processed_question = self._get_processed_question(question, use_enhanced_query, chat_history)
                graph_result = self.graph_service.search(processed_question)
                
                # Convert graph entities and relationships
                graph_entities = [
                    GraphEntity(
                        id=entity["id"],
                        type=entity["type"],
                        properties=entity.get("properties", {})
                    )
                    for entity in graph_result.entities
                ]
                
                graph_relationships = [
                    GraphRelationship(
                        source=rel["source"],
                        target=rel["target"],
                        type=rel["type"],
                        properties=rel.get("properties", {})
                    )
                    for rel in graph_result.relationships
                ]
                
                # Enhance the answer with graph context if available
                if graph_result.context and graph_entities:
                    enhanced_context = f"\n\nInformasi tambahan dari knowledge graph:\n{graph_result.context}"
                    vector_result["answer"] += enhanced_context
                
                logger.info(f"Hybrid search enhanced with {len(graph_entities)} entities and {len(graph_relationships)} relationships")
                
            except Exception as e:
                logger.warning(f"Graph enhancement failed in hybrid search: {e}")
        
        # Update result with hybrid information
        vector_result.update({
            "search_type_used": SearchType.HYBRID,
            "graph_entities": graph_entities,
            "graph_relationships": graph_relationships,
            "processing_time": time.time() - start_time
        })
        
        return vector_result
    
    def _get_processed_question(self, question: str, use_enhanced_query: bool, chat_history: List) -> str:
        """Process question with optional enhancement"""
        if use_enhanced_query:
            try:
                enhanced_question = self.query_enhancer.enhance_query(question, chat_history)
                logger.info(f"Query enhanced from '{question}' to '{enhanced_question}'")
                return enhanced_question
            except Exception as e:
                logger.warning(f"Query enhancement failed: {e}")
        
        return question
    
    def update_memory(self, question: str, answer: str) -> None:
        """Update conversation memory with new Q&A pair"""
        if self.memory:
            try:
                # Add to memory manually if needed
                self.memory.chat_memory.add_user_message(question)
                self.memory.chat_memory.add_ai_message(answer)
                logger.debug("Memory updated with new Q&A pair")
            except Exception as e:
                logger.warning(f"Failed to update memory: {e}")
    
    def get_conversation_history(self) -> List[Dict[str, str]]:
        """Get formatted conversation history"""
        if not self.memory:
            return []
        
        try:
            messages = self.memory.chat_memory.messages
            history = []
            
            for i in range(0, len(messages), 2):
                if i + 1 < len(messages):
                    user_msg = messages[i]
                    ai_msg = messages[i + 1]
                    
                    history.append({
                        "question": user_msg.content,
                        "answer": ai_msg.content,
                        "timestamp": datetime.now().isoformat()
                    })
            
            return history
            
        except Exception as e:
            logger.error(f"Failed to get conversation history: {e}")
            return []
    
    def validate_system_state(self) -> Dict[str, bool]:
        """Validate the current state of the RAG system"""
        return {
            "chain_initialized": self.chain is not None,
            "memory_initialized": self.memory is not None,
            "vector_store_available": self.vector_store_manager.vector_store is not None,
            "graph_service_available": self.graph_service is not None,
            "graph_data_available": self.graph_service.has_data() if self.graph_service else False
        }
    
    def get_system_info(self) -> Dict[str, Any]:
        """Get comprehensive system information"""
        state = self.validate_system_state()
        
        info = {
            "system_state": state,
            "configuration": {
                "chunk_size": settings.document.chunk_size,
                "search_k": settings.retrieval.search_k,
                "rerank_k": settings.retrieval.rerank_k,
                "graph_enabled": settings.graph.enable_graph_processing,
                "conversation_window": settings.memory.conversation_window_size
            },
            "last_query_info": {
                "chunks_retrieved": len(self.last_retrieved_chunks),
                "has_recent_chunks": len(self.last_retrieved_chunks) > 0
            }
        }
        
        # Add graph statistics if available
        if state["graph_service_available"]:
            info["graph_stats"] = self.get_graph_stats()
        
        return info