import time
import logging
from typing import List, Dict, Any, Optional
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferWindowMemory
from langchain.prompts import PromptTemplate
from langchain.schema import Document

from config import Config
from interfaces import RAGServiceInterface, ModelManagerInterface, VectorStoreInterface
from models import SearchType, SourceDocument, GraphEntity, GraphRelationship
from exceptions import RAGException
from graph_service import SemanticGraphService

logger = logging.getLogger(__name__)

class RAGService(RAGServiceInterface):
    def __init__(self, config: Config, model_manager: ModelManagerInterface, vector_store_manager: VectorStoreInterface):
        self.config = config
        self.model_manager = model_manager
        self.vector_store_manager = vector_store_manager
        self.graph_service = SemanticGraphService(config) if config.ENABLE_GRAPH_PROCESSING else None
        
        vector_store_manager.graph_service = self.graph_service
        
        self.last_retrieved_chunks = []
        self.chain = None
        self.memory = None

    def setup_chain(self) -> None:
        try:
            self.memory = ConversationBufferWindowMemory(
                memory_key="chat_history",
                return_messages=True,
                output_key="answer",
                k=5 
            )
            
            qa_prompt = PromptTemplate(
                template=self.config.QA_TEMPLATE,
                input_variables=["context", "question"]
            )
            
            self.chain = ConversationalRetrievalChain.from_llm(
                llm=self.model_manager.get_llm(),
                retriever=self.vector_store_manager.get_retriever(),
                memory=self.memory,
                return_source_documents=True,
                return_generated_question=True,
                combine_docs_chain_kwargs={"prompt": qa_prompt},
                verbose=True
            )
            
            if self.graph_service:
                self.graph_service.load_graph_data()
            
            logger.info("RAG chain setup complete")
            
        except Exception as e:
            logger.error(f"Failed to setup RAG chain: {e}")
            raise RAGException(f"Failed to setup RAG chain: {e}")
    
    def query(self, question: str, search_type: SearchType = SearchType.VECTOR, use_enhanced_query: bool = False, chat_history: List = None) -> Dict[str, Any]:
        if not self.chain:
            raise RAGException("RAG chain not initialized")
        
        start_time = time.time()
        
        try:
            if search_type == SearchType.GRAPH:
                return self._query_graph(question, use_enhanced_query, chat_history, start_time)
            elif search_type == SearchType.HYBRID:
                return self._query_hybrid(question, use_enhanced_query, chat_history, start_time)
            else:
                return self._query_vector(question, use_enhanced_query, chat_history, start_time)
                
        except Exception as e:
            logger.error(f"Query processing failed: {e}")
            raise RAGException(f"Query processing failed: {e}")
    
    def clear_memory(self) -> None:
        if self.memory:
            self.memory.clear()
            logger.info("Memory cleared")
    
    def get_last_chunks(self) -> List[Document]:
        return self.last_retrieved_chunks
    
    def get_graph_stats(self) -> Dict[str, Any]:
        if not self.graph_service:
            return {"has_graph": False}
        
        stats = self.graph_service.get_graph_stats()
        stats["has_graph"] = True
        return stats
    
    def visualize_graph(self, filename: str = "graph_visualization.html") -> str:
        if not self.graph_service:
            return ""
        
        return self.graph_service.visualize_graph(filename)
    
    def _query_vector(self, question: str, use_enhanced_query: bool, chat_history: List, start_time: float) -> Dict[str, Any]:
        processed_question = question
        if use_enhanced_query:
            processed_question = self._enhance_query(question, chat_history)
            logger.info(f"Enhanced query: {processed_question}")

        result = self.chain({"question": processed_question})
        self.last_retrieved_chunks = result.get("source_documents", [])

        source_documents = [
            SourceDocument(
                content=doc.page_content,
                metadata=doc.metadata,
                score=doc.metadata.get("score"),
                rerank_score=doc.metadata.get("rerank_score")
            )
            for doc in self.last_retrieved_chunks
        ]
        
        processing_time = time.time() - start_time
        
        return {
            "answer": result["answer"],
            "source_documents": source_documents,
            "generated_question": result.get("generated_question"),
            "enhanced_query": use_enhanced_query,
            "search_type_used": SearchType.VECTOR,
            "processing_time": processing_time,
            "graph_entities": [],
            "graph_relationships": []
        }
    
    def _query_graph(self, question: str, use_enhanced_query: bool, chat_history: List, start_time: float) -> Dict[str, Any]:
        if not self.graph_service or not self.graph_service.has_graph_data():
            logger.warning("Graph search requested but no graph data available, falling back to vector search")
            return self._query_vector(question, use_enhanced_query, chat_history, start_time)
        
        processed_question = question
        if use_enhanced_query:
            processed_question = self._enhance_query(question, chat_history)
        
        graph_result = self.graph_service.search_graph(processed_question)
        
        graph_prompt = PromptTemplate(
            template=self.config.GRAPH_QA_TEMPLATE,
            input_variables=["graph_context", "question"]
        )
        
        llm = self.model_manager.get_llm()
        formatted_prompt = graph_prompt.format(
            graph_context=graph_result["context"],
            question=processed_question
        )
        
        response = llm.invoke(formatted_prompt)
        answer = response.content
        
        graph_entities = [
            GraphEntity(
                id=entity["id"],
                type=entity["type"],
                properties=entity.get("properties", {})
            )
            for entity in graph_result.get("relevant_entities", [])
        ]
        
        graph_relationships = [
            GraphRelationship(
                source=rel["source"],
                target=rel["target"],
                type=rel["type"],
                properties=rel.get("properties", {})
            )
            for rel in graph_result.get("relevant_relationships", [])
        ]
        
        processing_time = time.time() - start_time
        
        return {
            "answer": answer,
            "source_documents": [],
            "generated_question": processed_question if use_enhanced_query else None,
            "enhanced_query": use_enhanced_query,
            "search_type_used": SearchType.GRAPH,
            "processing_time": processing_time,
            "graph_entities": graph_entities,
            "graph_relationships": graph_relationships
        }
    
    def _query_hybrid(self, question: str, use_enhanced_query: bool, chat_history: List, start_time: float) -> Dict[str, Any]:
        vector_result = self._query_vector(question, use_enhanced_query, chat_history, start_time)
        
        graph_entities = []
        graph_relationships = []
        
        if self.graph_service and self.graph_service.has_graph_data():
            processed_question = question
            if use_enhanced_query:
                processed_question = self._enhance_query(question, chat_history)
            
            graph_result = self.graph_service.search_graph(processed_question)
            
            graph_entities = [
                GraphEntity(
                    id=entity["id"],
                    type=entity["type"],
                    properties=entity.get("properties", {})
                )
                for entity in graph_result.get("relevant_entities", [])
            ]
            
            graph_relationships = [
                GraphRelationship(
                    source=rel["source"],
                    target=rel["target"],
                    type=rel["type"],
                    properties=rel.get("properties", {})
                )
                for rel in graph_result.get("relevant_relationships", [])
            ]
            
            if graph_result["context"] and graph_result["context"] != "Tidak ditemukan informasi graph yang relevan dengan pertanyaan.":
                enhanced_prompt = f"""
                Konteks dari pencarian vektor: {vector_result['answer']}
                
                Informasi tambahan dari graph dengan semantic search dan traversal: {graph_result['context']}
                
                Entry points yang ditemukan: {graph_result.get('entry_points', [])}
                
                Pertanyaan: {question}
                
                Berikan jawaban yang menggabungkan informasi dari kedua sumber di atas dengan fokus pada hubungan semantik:
                """
                
                llm = self.model_manager.get_llm()
                enhanced_response = llm.invoke(enhanced_prompt)
                vector_result["answer"] = enhanced_response.content
        
        vector_result.update({
            "search_type_used": SearchType.HYBRID,
            "graph_entities": graph_entities,
            "graph_relationships": graph_relationships,
            "processing_time": time.time() - start_time
        })
        
        return vector_result

    def _enhance_query(self, query: str, chat_history: List = None) -> str:
        try:
            llm = self.model_manager.get_llm()
            history_context = self._format_chat_history(chat_history)
            
            enhancement_prompt = PromptTemplate(
                template=self.config.QUERY_ENHANCEMENT_TEMPLATE,
                input_variables=["query", "chat_history"]
            )
            
            formatted_prompt = enhancement_prompt.format(
                query=query, 
                chat_history=history_context
            )
            response = llm.invoke(formatted_prompt)
            
            enhanced_query = response.content.strip()
            return enhanced_query if enhanced_query else query
            
        except Exception as e:
            logger.error(f"Query enhancement failed: {e}")
            return query

    def _format_chat_history(self, chat_history: List = None) -> str:
        if not chat_history:
            return "Tidak ada riwayat percakapan sebelumnya."

        recent_history = chat_history[-6:] if len(chat_history) > 6 else chat_history
        
        formatted_history = []
        for msg in recent_history:
            role = "Pengguna" if msg.role == "user" else "Asisten"
            content = msg.content[:300] + "..." if len(msg.content) > 300 else msg.content
            formatted_history.append(f"{role}: {content}")
        
        return "\n".join(formatted_history) if formatted_history else "Tidak ada riwayat percakapan sebelumnya."