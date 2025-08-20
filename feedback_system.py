import logging
import json
import pickle
from typing import List, Dict, Any, Optional
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, asdict
from langchain.schema import Document
from langchain.prompts import PromptTemplate

from config import Config
from interfaces import RAGServiceInterface
from models import QueryRequest, SearchType, FeedbackRequest

logger = logging.getLogger(__name__)


@dataclass
class FeedbackEntry:
    """Single feedback entry"""
    query: str
    response: str
    relevance_score: int
    quality_score: int
    search_type: str
    response_time: Optional[float] = None
    comments: Optional[str] = None
    user_id: Optional[str] = None
    timestamp: datetime = None
    feedback_id: str = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()
        if self.feedback_id is None:
            self.feedback_id = f"fb_{int(self.timestamp.timestamp())}"


class FeedbackStorage:
    """Storage system for feedback data"""
    
    def __init__(self, storage_directory: str = "data/feedback"):
        self.storage_dir = Path(storage_directory)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.feedback_file = self.storage_dir / "feedback_entries.json"
        self.learned_data_file = self.storage_dir / "learned_documents.pkl"
        
    def store_feedback(self, feedback: FeedbackEntry) -> str:
        """Store a feedback entry"""
        try:
            # Load existing feedback
            existing_feedback = self.load_all_feedback()
            
            # Add new feedback
            existing_feedback.append(feedback)
            
            # Save updated feedback
            with open(self.feedback_file, 'w', encoding='utf-8') as f:
                feedback_data = [asdict(entry) for entry in existing_feedback]
                # Convert datetime to string for JSON serialization
                for entry in feedback_data:
                    entry['timestamp'] = entry['timestamp'].isoformat()
                json.dump(feedback_data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Stored feedback entry: {feedback.feedback_id}")
            return feedback.feedback_id
            
        except Exception as e:
            logger.error(f"Failed to store feedback: {e}")
            raise
    
    def load_all_feedback(self) -> List[FeedbackEntry]:
        """Load all feedback entries"""
        if not self.feedback_file.exists():
            return []
        
        try:
            with open(self.feedback_file, 'r', encoding='utf-8') as f:
                feedback_data = json.load(f)
            
            feedback_entries = []
            for data in feedback_data:
                # Convert timestamp string back to datetime
                if isinstance(data['timestamp'], str):
                    data['timestamp'] = datetime.fromisoformat(data['timestamp'])
                feedback_entries.append(FeedbackEntry(**data))
            
            return feedback_entries
            
        except Exception as e:
            logger.error(f"Failed to load feedback: {e}")
            return []
    
    def get_high_quality_feedback(self, min_relevance: int = 4, min_quality: int = 4) -> List[FeedbackEntry]:
        """Get high-quality feedback entries for learning"""
        all_feedback = self.load_all_feedback()
        return [
            fb for fb in all_feedback 
            if fb.relevance_score >= min_relevance and fb.quality_score >= min_quality
        ]
    
    def clear_all_feedback(self) -> int:
        """Clear all feedback entries"""
        count = len(self.load_all_feedback())
        
        if self.feedback_file.exists():
            self.feedback_file.unlink()
        
        if self.learned_data_file.exists():
            self.learned_data_file.unlink()
        
        logger.info(f"Cleared {count} feedback entries")
        return count
    
    def get_feedback_stats(self) -> Dict[str, Any]:
        """Get feedback statistics"""
        feedback_entries = self.load_all_feedback()
        
        if not feedback_entries:
            return {
                "total_feedback": 0,
                "average_relevance": 0.0,
                "average_quality": 0.0,
                "search_type_distribution": {},
                "feedback_learning_enabled": True
            }
        
        total = len(feedback_entries)
        avg_relevance = sum(fb.relevance_score for fb in feedback_entries) / total
        avg_quality = sum(fb.quality_score for fb in feedback_entries) / total
        
        # Search type distribution
        search_types = {}
        for fb in feedback_entries:
            search_types[fb.search_type] = search_types.get(fb.search_type, 0) + 1
        
        return {
            "total_feedback": total,
            "average_relevance": round(avg_relevance, 2),
            "average_quality": round(avg_quality, 2),
            "search_type_distribution": search_types,
            "feedback_learning_enabled": True
        }


class FeedbackLearner:
    """Learn from feedback and enhance queries/retrieval"""
    
    def __init__(self, rag_service: RAGServiceInterface, config: Config):
        self.rag_service = rag_service
        self.config = config
        self.storage = FeedbackStorage()
        
    def create_learned_documents_from_feedback(self) -> List[Document]:
        """Create synthetic documents from high-quality feedback"""
        high_quality_feedback = self.storage.get_high_quality_feedback()
        
        if not high_quality_feedback:
            return []
        
        learned_docs = []
        for feedback in high_quality_feedback:
            # Create a synthetic document from Q&A pair
            content = f"Question: {feedback.query}\n\nAnswer: {feedback.response}"
            
            # Create metadata
            metadata = {
                "source": "feedback_learning",
                "feedback_id": feedback.feedback_id,
                "relevance_score": feedback.relevance_score,
                "quality_score": feedback.quality_score,
                "search_type": feedback.search_type,
                "timestamp": feedback.timestamp.isoformat(),
                "document_type": "learned_qa",
                "content_hash": f"fb_{feedback.feedback_id}"
            }
            
            if feedback.comments:
                content += f"\n\nAdditional Context: {feedback.comments}"
                
            doc = Document(page_content=content, metadata=metadata)
            learned_docs.append(doc)
        
        logger.info(f"Created {len(learned_docs)} learned documents from feedback")
        return learned_docs
    
    def enhance_query_with_feedback_context(self, query: str, chat_history: List = None) -> str:
        """Enhance query using similar feedback entries"""
        try:
            # Find similar feedback entries
            similar_feedback = self._find_similar_feedback(query)
            
            if not similar_feedback:
                return query  # No similar feedback found
            
            # Create enhancement context from similar feedback
            feedback_context = self._create_feedback_context(similar_feedback)
            
            # Use LLM to enhance query
            enhanced_query = self._enhance_query_with_llm(query, feedback_context, chat_history)
            
            logger.info(f"Enhanced query using {len(similar_feedback)} feedback entries")
            return enhanced_query
            
        except Exception as e:
            logger.error(f"Query enhancement with feedback failed: {e}")
            return query
    
    def _find_similar_feedback(self, query: str, max_results: int = 3) -> List[FeedbackEntry]:
        """Find feedback entries similar to the current query"""
        high_quality_feedback = self.storage.get_high_quality_feedback()
        
        if not high_quality_feedback:
            return []
        
        # Simple similarity based on keyword overlap
        # In production, you might use embeddings for better similarity
        query_words = set(query.lower().split())
        
        scored_feedback = []
        for feedback in high_quality_feedback:
            feedback_words = set(feedback.query.lower().split())
            similarity = len(query_words.intersection(feedback_words)) / len(query_words.union(feedback_words))
            if similarity > 0.2:  # Threshold for similarity
                scored_feedback.append((feedback, similarity))
        
        # Sort by similarity and return top results
        scored_feedback.sort(key=lambda x: x[1], reverse=True)
        return [fb for fb, score in scored_feedback[:max_results]]
    
    def _create_feedback_context(self, feedback_entries: List[FeedbackEntry]) -> str:
        """Create context string from feedback entries"""
        context_parts = ["Similar queries and high-quality answers from previous interactions:"]
        
        for i, feedback in enumerate(feedback_entries, 1):
            context_parts.append(f"\n{i}. Query: {feedback.query}")
            context_parts.append(f"   Answer: {feedback.response[:200]}...")  # Truncate long responses
            if feedback.comments:
                context_parts.append(f"   Context: {feedback.comments}")
        
        return "\n".join(context_parts)
    
    def _enhance_query_with_llm(self, query: str, feedback_context: str, chat_history: List = None) -> str:
        """Use LLM to enhance query with feedback context"""
        try:
            llm = self.rag_service.model_manager.get_llm()
            
            # Format chat history
            history_context = ""
            if chat_history:
                recent_history = chat_history[-3:]  # Last 3 exchanges
                history_parts = []
                for msg in recent_history:
                    role = "User" if msg.role == "user" else "Assistant"
                    content = msg.content[:150] + "..." if len(msg.content) > 150 else msg.content
                    history_parts.append(f"{role}: {content}")
                history_context = "\n".join(history_parts)
            
            enhancement_template = """You are an expert in query reformulation for legal document search systems.
Based on the conversation history, similar high-quality Q&A pairs, and the current question, 
reformulate the question to be more specific and effective for document retrieval.

Conversation History:
{chat_history}

Similar High-Quality Q&A Examples:
{feedback_context}

Current Question: {query}

Provide a reformulated question that:
1. Incorporates insights from similar high-quality interactions
2. Uses more specific legal terminology when appropriate
3. Maintains the original intent of the question
4. Is optimized for document search and retrieval
5. Considers the conversation context

Enhanced Question:"""

            prompt = PromptTemplate(
                template=enhancement_template,
                input_variables=["query", "feedback_context", "chat_history"]
            )
            
            formatted_prompt = prompt.format(
                query=query,
                feedback_context=feedback_context,
                chat_history=history_context or "No previous conversation"
            )
            
            response = llm.invoke(formatted_prompt)
            enhanced_query = response.content.strip()
            
            # Validate enhanced query
            if enhanced_query and len(enhanced_query) > 10:
                return enhanced_query
            else:
                return query
                
        except Exception as e:
            logger.error(f"LLM-based query enhancement failed: {e}")
            return query


class EnhancedRAGService:
    """RAG Service with feedback learning capabilities"""
    
    def __init__(self, base_rag_service: RAGServiceInterface, config: Config):
        self.base_rag_service = base_rag_service
        self.config = config
        self.feedback_learner = FeedbackLearner(base_rag_service, config)
        self.storage = FeedbackStorage()
        
        # Add learned documents to vector store
        self._initialize_learned_documents()
    
    def _initialize_learned_documents(self):
        """Initialize vector store with learned documents from feedback"""
        try:
            learned_docs = self.feedback_learner.create_learned_documents_from_feedback()
            
            if learned_docs:
                # Add learned documents to vector store
                success = self.base_rag_service.vector_store_manager.add_documents(learned_docs)
                if success:
                    logger.info(f"Added {len(learned_docs)} learned documents to vector store")
                else:
                    logger.warning("Failed to add learned documents to vector store")
            
        except Exception as e:
            logger.error(f"Failed to initialize learned documents: {e}")
    
    def query_with_feedback_learning(self, question: str, search_type: SearchType = SearchType.VECTOR, 
                                   use_enhanced_query: bool = True, chat_history: List = None) -> Dict[str, Any]:
        """Query with feedback-enhanced processing"""
        start_time = datetime.now()
        
        try:
            # Enhance query using feedback if enabled
            processed_question = question
            feedback_learning_applied = False
            feedback_entries_used = 0
            
            if use_enhanced_query:
                enhanced_question = self.feedback_learner.enhance_query_with_feedback_context(
                    question, chat_history
                )
                if enhanced_question != question:
                    processed_question = enhanced_question
                    feedback_learning_applied = True
                    # Count similar feedback entries used
                    similar_feedback = self.feedback_learner._find_similar_feedback(question)
                    feedback_entries_used = len(similar_feedback)
                    
                    logger.info(f"Query enhanced with feedback: '{question}' -> '{processed_question}'")
            
            # Use the base RAG service for actual query processing
            # Fixed: Use get_retriever() and then retrieve documents, don't call invoke on retriever
            result = self.base_rag_service.query(
                processed_question, 
                search_type=search_type,
                use_enhanced_query=False,  # We already enhanced it
                chat_history=chat_history
            )
            
            # Add feedback learning metadata
            processing_time = (datetime.now() - start_time).total_seconds()
            
            # Count learned documents in results
            documents_learned = 0
            if 'source_documents' in result:
                for doc in result['source_documents']:
                    if hasattr(doc, 'metadata') and doc.metadata.get('source') == 'feedback_learning':
                        documents_learned += 1
            
            # Create enhanced result
            enhanced_result = {
                **result,
                'feedback_learning_applied': feedback_learning_applied,
                'feedback_entries_used': feedback_entries_used,
                'documents_learned': documents_learned,
                'query_with_feedback_time': processing_time,
                'original_question': question,
                'processed_question': processed_question
            }
            
            return enhanced_result
            
        except Exception as e:
            logger.error(f"Feedback-enhanced query failed: {e}")
            # Fallback to base service
            return self.base_rag_service.query(question, search_type=search_type, chat_history=chat_history)
    
    def store_feedback(self, feedback_request: FeedbackRequest) -> str:
        """Store feedback for learning"""
        try:
            feedback_entry = FeedbackEntry(
                query=feedback_request.query,
                response=feedback_request.response,
                relevance_score=feedback_request.relevance_score,
                quality_score=feedback_request.quality_score,
                search_type=feedback_request.search_type or "vector",
                response_time=feedback_request.response_time,
                comments=feedback_request.comments,
                user_id=feedback_request.user_id
            )
            
            feedback_id = self.storage.store_feedback(feedback_entry)
            
            # If high-quality feedback, update learned documents
            if feedback_entry.relevance_score >= 4 and feedback_entry.quality_score >= 4:
                self._update_learned_documents()
            
            return feedback_id
            
        except Exception as e:
            logger.error(f"Failed to store feedback: {e}")
            raise
    
    def _update_learned_documents(self):
        """Update vector store with new learned documents"""
        try:
            # Get new learned documents
            learned_docs = self.feedback_learner.create_learned_documents_from_feedback()
            
            if learned_docs:
                # Filter out documents already in vector store
                existing_hashes = set()
                try:
                    # Get existing learned documents
                    if hasattr(self.base_rag_service.vector_store_manager, 'vector_store'):
                        collection = self.base_rag_service.vector_store_manager.vector_store.get()
                        for metadata in collection.get('metadatas', []):
                            if metadata and metadata.get('source') == 'feedback_learning':
                                existing_hashes.add(metadata.get('content_hash', ''))
                except Exception as e:
                    logger.warning(f"Could not get existing learned documents: {e}")
                
                # Add only new documents
                new_docs = [
                    doc for doc in learned_docs 
                    if doc.metadata.get('content_hash') not in existing_hashes
                ]
                
                if new_docs:
                    success = self.base_rag_service.vector_store_manager.add_documents(new_docs)
                    if success:
                        logger.info(f"Added {len(new_docs)} new learned documents")
            
        except Exception as e:
            logger.error(f"Failed to update learned documents: {e}")
    
    def get_feedback_stats(self) -> Dict[str, Any]:
        """Get feedback statistics"""
        return self.storage.get_feedback_stats()
    
    def clear_feedback_history(self) -> int:
        """Clear all feedback history"""
        return self.storage.clear_all_feedback()
    
    # Delegate other methods to base service
    def __getattr__(self, name):
        return getattr(self.base_rag_service, name)