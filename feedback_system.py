import json
import logging
import time
from typing import List, Dict, Any, Optional
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass
from langchain.schema import Document

logger = logging.getLogger(__name__)

@dataclass
class FeedbackEntry:
    query: str
    response: str
    relevance_score: int  # 1-5
    quality_score: int    # 1-5
    response_time: Optional[float] = None
    search_type: Optional[str] = None
    timestamp: Optional[datetime] = None
    comments: Optional[str] = None
    user_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "query": self.query,
            "response": self.response,
            "relevance_score": self.relevance_score,
            "quality_score": self.quality_score,
            "response_time": self.response_time,
            "search_type": self.search_type,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "comments": self.comments,
            "user_id": self.user_id
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'FeedbackEntry':
        timestamp = None
        if data.get('timestamp'):
            timestamp = datetime.fromisoformat(data['timestamp'])
        
        return cls(
            query=data['query'],
            response=data['response'],
            relevance_score=data['relevance_score'],
            quality_score=data['quality_score'],
            response_time=data.get('response_time'),
            search_type=data.get('search_type'),
            timestamp=timestamp,
            comments=data.get('comments'),
            user_id=data.get('user_id')
        )


class FeedbackStorage:
    def __init__(self, storage_path: str = "data/feedback.jsonl"):
        self.storage_path = Path(storage_path)
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
    
    def store_feedback(self, feedback: FeedbackEntry) -> bool:
        try:
            with open(self.storage_path, 'a', encoding='utf-8') as f:
                json.dump(feedback.to_dict(), f, ensure_ascii=False)
                f.write('\n')
            logger.info(f"Stored feedback for query: {feedback.query[:50]}...")
            return True
        except Exception as e:
            logger.error(f"Failed to store feedback: {e}")
            return False
    
    def load_all_feedback(self) -> List[FeedbackEntry]:
        if not self.storage_path.exists():
            return []
        
        feedback_entries = []
        try:
            with open(self.storage_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        data = json.loads(line)
                        feedback_entries.append(FeedbackEntry.from_dict(data))
            logger.info(f"Loaded {len(feedback_entries)} feedback entries")
        except Exception as e:
            logger.error(f"Failed to load feedback: {e}")
        
        return feedback_entries


class FeedbackLearner:
    def __init__(self, min_similarity_threshold: float = 0.2, max_adjustment: float = 0.3):
        self.min_similarity_threshold = min_similarity_threshold
        self.max_adjustment = max_adjustment
    
    def calculate_query_similarity(self, query1: str, query2: str) -> float:
        """Calculate similarity between two queries using Jaccard similarity"""
        words1 = set(query1.lower().split())
        words2 = set(query2.lower().split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = len(words1 & words2)
        union = len(words1 | words2)
        
        return intersection / union if union > 0 else 0.0
    
    def calculate_document_relevance(self, document: Document, query_words: set) -> float:
        """Calculate how relevant a document is to query words"""
        doc_words = set(document.page_content.lower().split())
        
        if not doc_words or not query_words:
            return 0.0
        
        intersection = len(doc_words & query_words)
        return intersection / len(query_words) if query_words else 0.0
    
    def apply_feedback_learning(self, 
                              documents: List[Document], 
                              current_query: str, 
                              feedback_history: List[FeedbackEntry]) -> List[Document]:
        """Apply feedback learning to adjust document relevance scores"""
        
        if not feedback_history:
            # Set default scores if no feedback history
            for doc in documents:
                doc.metadata['relevance_score'] = 1.0
            return documents
        
        current_query_words = set(current_query.lower().split())
        
        for doc in documents:
            applicable_feedback = []
            
            # Find relevant feedback for this document
            for feedback in feedback_history:
                # Calculate similarity between current query and feedback query
                query_similarity = self.calculate_query_similarity(current_query, feedback.query)
                
                # Calculate document relevance to feedback query
                feedback_words = set(feedback.query.lower().split())
                doc_relevance = self.calculate_document_relevance(doc, feedback_words)
                
                # Include feedback if both similarities exceed threshold
                if query_similarity >= self.min_similarity_threshold and doc_relevance > 0.1:
                    applicable_feedback.append(feedback)
            
            # Calculate adjustment based on applicable feedback
            if applicable_feedback:
                # Calculate average satisfaction score (0-1 scale)
                total_score = sum(
                    (fb.relevance_score + fb.quality_score) / 10.0  # Normalize to 0-1
                    for fb in applicable_feedback
                )
                avg_satisfaction = total_score / len(applicable_feedback)
                
                # Calculate adjustment (-max_adjustment to +max_adjustment)
                adjustment = (avg_satisfaction - 0.5) * 2 * self.max_adjustment
                
                # Apply adjustment
                base_score = doc.metadata.get('relevance_score', 1.0)
                doc.metadata['relevance_score'] = base_score * (1 + adjustment)
                doc.metadata['feedback_applied'] = len(applicable_feedback)
                
                logger.debug(f"Document adjusted by {adjustment:.3f} based on {len(applicable_feedback)} feedback entries")
            else:
                # Set default score if no applicable feedback
                doc.metadata['relevance_score'] = doc.metadata.get('relevance_score', 1.0)
                doc.metadata['feedback_applied'] = 0
        
        return documents
    
    def get_feedback_stats(self, feedback_history: List[FeedbackEntry]) -> Dict[str, Any]:
        """Get statistics about feedback history"""
        if not feedback_history:
            return {
                "total_feedback": 0,
                "average_relevance": 0.0,
                "average_quality": 0.0,
                "search_type_distribution": {},
                "recent_feedback": []
            }
        
        total = len(feedback_history)
        avg_relevance = sum(f.relevance_score for f in feedback_history) / total
        avg_quality = sum(f.quality_score for f in feedback_history) / total
        
        search_type_counts = {}
        for feedback in feedback_history:
            search_type = feedback.search_type or "unknown"
            search_type_counts[search_type] = search_type_counts.get(search_type, 0) + 1
        
        return {
            "total_feedback": total,
            "average_relevance": round(avg_relevance, 2),
            "average_quality": round(avg_quality, 2),
            "search_type_distribution": search_type_counts,
            "recent_feedback": [fb.to_dict() for fb in feedback_history[-5:]] if total >= 5 else [fb.to_dict() for fb in feedback_history]
        }


class EnhancedRAGService:
    """Enhanced RAG Service with feedback learning capabilities"""
    
    def __init__(self, base_rag_service, config):
        self.base_rag_service = base_rag_service
        self.config = config
        self.feedback_storage = FeedbackStorage()
        self.feedback_learner = FeedbackLearner()
        self._feedback_cache = []
        self._cache_last_updated = 0
        self.cache_ttl = 300  # 5 minutes cache TTL
    
    def _get_feedback_history(self) -> List[FeedbackEntry]:
        """Get feedback history with caching for performance"""
        current_time = time.time()
        
        if (current_time - self._cache_last_updated) > self.cache_ttl:
            self._feedback_cache = self.feedback_storage.load_all_feedback()
            self._cache_last_updated = current_time
        
        return self._feedback_cache
    
    def query_with_feedback_learning(self, 
                                   question: str, 
                                   search_type: str = "vector",
                                   use_enhanced_query: bool = False,
                                   chat_history: List = None) -> Dict[str, Any]:
        """Execute query with feedback learning applied to document selection"""
        
        start_time = time.time()
        
        try:
            # Get feedback history
            feedback_history = self._get_feedback_history()
            
            # First, get documents using standard retrieval
            if hasattr(self.base_rag_service.vector_store_manager, 'vector_store'):
                retriever = self.base_rag_service.vector_store_manager.get_retriever()
                
                # For vector/hybrid search, apply feedback learning
                if search_type in ["vector", "hybrid"]:
                    # Get more documents initially for better selection
                    extended_retriever = self.base_rag_service.vector_store_manager.vector_store.as_retriever(
                        search_kwargs={"k": self.config.SEARCH_K * 2}  # Get more candidates
                    )
                    candidate_docs = extended_retriever.invoke(question)
                    
                    # Apply feedback learning
                    learned_docs = self.feedback_learner.apply_feedback_learning(
                        candidate_docs, question, feedback_history
                    )
                    
                    # Sort by relevance score and take top k
                    learned_docs.sort(key=lambda x: x.metadata.get('relevance_score', 1.0), reverse=True)
                    selected_docs = learned_docs[:self.config.RERANK_K]
                    
                    # Temporarily replace the retriever's documents for this query
                    original_method = retriever.invoke
                    retriever.invoke = lambda q: selected_docs
                    
                    # Execute the query with learned document selection
                    result = self.base_rag_service.query(
                        question=question,
                        search_type=search_type,
                        use_enhanced_query=use_enhanced_query,
                        chat_history=chat_history
                    )
                    
                    # Restore original retriever method
                    retriever.invoke = original_method
                    
                    # Add feedback learning metadata
                    result['feedback_learning_applied'] = True
                    result['feedback_entries_used'] = len(feedback_history)
                    result['documents_learned'] = len([d for d in selected_docs if d.metadata.get('feedback_applied', 0) > 0])
                    
                else:
                    # For graph search, use standard query
                    result = self.base_rag_service.query(
                        question=question,
                        search_type=search_type,
                        use_enhanced_query=use_enhanced_query,
                        chat_history=chat_history
                    )
                    result['feedback_learning_applied'] = False
            else:
                # Fallback to standard query if vector store not available
                result = self.base_rag_service.query(
                    question=question,
                    search_type=search_type,
                    use_enhanced_query=use_enhanced_query,
                    chat_history=chat_history
                )
                result['feedback_learning_applied'] = False
            
            result['query_with_feedback_time'] = time.time() - start_time
            return result
            
        except Exception as e:
            logger.error(f"Feedback-enhanced query failed: {e}")
            # Fallback to standard query
            return self.base_rag_service.query(
                question=question,
                search_type=search_type,
                use_enhanced_query=use_enhanced_query,
                chat_history=chat_history
            )
    
    def store_feedback(self, 
                      query: str, 
                      response: str, 
                      relevance_score: int, 
                      quality_score: int,
                      response_time: Optional[float] = None,
                      search_type: Optional[str] = None,
                      comments: Optional[str] = None,
                      user_id: Optional[str] = None) -> bool:
        """Store user feedback for learning"""
        
        feedback = FeedbackEntry(
            query=query,
            response=response,
            relevance_score=relevance_score,
            quality_score=quality_score,
            response_time=response_time,
            search_type=search_type,
            timestamp=datetime.now(),
            comments=comments,
            user_id=user_id
        )
        
        success = self.feedback_storage.store_feedback(feedback)
        
        if success:
            # Invalidate cache to include new feedback
            self._cache_last_updated = 0
            logger.info(f"Feedback stored successfully for query: {query[:50]}...")
        
        return success
    
    def get_feedback_stats(self) -> Dict[str, Any]:
        """Get feedback statistics"""
        feedback_history = self._get_feedback_history()
        return self.feedback_learner.get_feedback_stats(feedback_history)
    
    def clear_feedback_history(self) -> bool:
        """Clear all feedback history (use with caution)"""
        try:
            if self.feedback_storage.storage_path.exists():
                self.feedback_storage.storage_path.unlink()
            self._feedback_cache = []
            self._cache_last_updated = 0
            logger.info("Feedback history cleared")
            return True
        except Exception as e:
            logger.error(f"Failed to clear feedback history: {e}")
            return False