from typing import List, Dict, Any, Optional
import logging
import time
from langchain.schema import Document

from config import Config
from services import ModelManager, RAGService, ServiceException
from graph_rag_service import GraphRAGService
from graph_database import SQLiteGraphDatabase
from graph_extractor import LLMGraphExtractor
from models import SourceDocument

logger = logging.getLogger(__name__)


class HybridRAGService:
    """Hybrid service that combines traditional RAG with Graph RAG"""
    
    def __init__(self, config: Config, model_manager: ModelManager, rag_service: RAGService):
        self.config = config
        self.model_manager = model_manager
        self.traditional_rag = rag_service
        
        # Initialize Graph RAG components
        self.graph_database = SQLiteGraphDatabase(
            db_path=f"{config.PERSIST_DIRECTORY}/graph.db"
        )
        self.graph_extractor = LLMGraphExtractor(config, model_manager)
        self.graph_rag = GraphRAGService(
            config, model_manager, self.graph_database, self.graph_extractor
        )
        
        self.graph_enabled = False
    
    def build_graph_from_documents(self, documents: List[Document]) -> bool:
        """Build graph from processed documents"""
        try:
            logger.info("Building knowledge graph from documents...")
            success = self.graph_rag.build_graph(documents)
            
            if success:
                self.graph_enabled = True
                logger.info("Knowledge graph built successfully")
            else:
                logger.warning("Failed to build knowledge graph")
            
            return success
            
        except Exception as e:
            logger.error(f"Failed to build graph: {e}")
            return False
    
    def query(self, question: str, use_enhanced_query: bool = False, 
              use_graph: bool = True, chat_history: List = None) -> Dict[str, Any]:
        """Query using hybrid approach - combines traditional RAG and Graph RAG"""
        start_time = time.time()
        
        try:
            # Always get traditional RAG response
            traditional_result = self.traditional_rag.query(
                question=question,
                use_enhanced_query=use_enhanced_query,
                chat_history=chat_history
            )
            
            # If graph is enabled and requested, also get graph response
            graph_result = None
            if use_graph and self.graph_enabled:
                try:
                    graph_result = self.graph_rag.query_graph(question, max_nodes=10)
                except Exception as e:
                    logger.warning(f"Graph query failed, using traditional RAG only: {e}")
            
            # Combine results
            final_answer = self._combine_responses(
                traditional_result, graph_result, question
            )
            
            # Combine source documents
            source_documents = traditional_result.get("source_documents", [])
            if graph_result and graph_result.source_chunks:
                # Add graph sources
                for chunk_id in list(graph_result.source_chunks)[:5]:  # Limit to 5
                    source_documents.append(SourceDocument(
                        content=f"Graph knowledge: {chunk_id}",
                        metadata={"source": "knowledge_graph", "chunk_id": chunk_id},
                        score=0.9
                    ))
            
            processing_time = time.time() - start_time
            
            return {
                "answer": final_answer,
                "source_documents": source_documents,
                "generated_question": traditional_result.get("generated_question"),
                "enhanced_query": use_enhanced_query,
                "processing_time": processing_time,
                "graph_used": graph_result is not None,
                "graph_nodes_found": len(graph_result.relevant_nodes) if graph_result else 0,
                "traditional_sources": len(traditional_result.get("source_documents", [])),
                "tokens_used": traditional_result.get("tokens_used")
            }
            
        except Exception as e:
            logger.error(f"Hybrid query failed: {e}")
            raise ServiceException(f"Hybrid query failed: {e}")
    
    def _combine_responses(self, traditional_result: Dict, graph_result, question: str) -> str:
        """Combine traditional RAG and Graph RAG responses intelligently"""
        traditional_answer = traditional_result.get("answer", "")
        
        if not graph_result:
            return traditional_answer
        
        # Use LLM to synthesize both responses
        try:
            llm = self.model_manager.get_llm()
            
            synthesis_prompt = f"""Anda adalah ahli dalam menggabungkan informasi dari berbagai sumber untuk memberikan jawaban yang komprehensif.

Pertanyaan: {question}

Jawaban dari RAG tradisional:
{traditional_answer}

Jawaban dari Knowledge Graph:
{graph_result.answer}

Tugas Anda:
1. Gabungkan kedua jawaban untuk memberikan respons yang paling lengkap dan akurat
2. Jika ada konflik informasi, prioritaskan yang lebih spesifik dan detail
3. Tambahkan hubungan atau konteks dari knowledge graph jika memperkaya jawaban
4. Pastikan jawaban tetap koheren dan mudah dipahami
5. Jika informasi saling melengkapi, integrasikan dengan baik

Jawaban terpadu:"""

            response = llm.invoke(synthesis_prompt)
            return response.content.strip()
            
        except Exception as e:
            logger.warning(f"Failed to synthesize responses: {e}")
            # Fallback: return traditional answer with graph supplement
            if graph_result.answer and graph_result.answer != traditional_answer:
                return f"{traditional_answer}\n\nInformasi tambahan dari knowledge graph:\n{graph_result.answer}"
            return traditional_answer
    
    def get_graph_stats(self) -> Dict[str, Any]:
        """Get graph statistics"""
        if self.graph_enabled:
            return self.graph_rag.get_graph_stats()
        return {"status": "disabled", "message": "Graph RAG not enabled"}
    
    def clear_memory(self):
        """Clear memory from traditional RAG"""
        self.traditional_rag.clear_memory()