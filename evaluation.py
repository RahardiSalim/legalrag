"""
Fixed RAGAS Evaluation Module for RAG System
Addresses OpenAI dependency and adds proper ground truth handling
"""

import logging
import time
from typing import List, Dict, Any, Optional
from datetime import datetime
from pathlib import Path
import pandas as pd
from datasets import Dataset
import os

# Import RAGAS components
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_recall,
    context_precision,
)

# Import LangChain wrappers for local models
from langchain.llms.base import LLM
from langchain.embeddings.base import Embeddings

# Try new import first, fallback to old one
try:
    from langchain_ollama import OllamaLLM
    OLLAMA_AVAILABLE = True
except ImportError:
    try:
        from langchain_community.llms import Ollama as OllamaLLM
        OLLAMA_AVAILABLE = True
    except ImportError:
        OLLAMA_AVAILABLE = False

# Import Gemini for evaluation
try:
    from langchain_google_genai import ChatGoogleGenerativeAI
    GEMINI_AVAILABLE = True
except ImportError:
    try:
        from langchain_community.llms import GooglePalm
        GEMINI_AVAILABLE = True
    except ImportError:
        GEMINI_AVAILABLE = False

from models import QueryRequest, SearchType
from interfaces import RAGServiceInterface

logger = logging.getLogger(__name__)


class LocalLLMWrapper(LLM):
    """Wrapper to make Ollama compatible with RAGAS"""
    
    def __init__(self, ollama_llm):
        super().__init__()
        self._ollama_llm = ollama_llm
    
    def _call(self, prompt: str, stop: Optional[List[str]] = None, run_manager=None, **kwargs) -> str:
        try:
            # Use invoke method for ChatOllama
            if hasattr(self._ollama_llm, 'invoke'):
                response = self._ollama_llm.invoke(prompt)
                return response.content if hasattr(response, 'content') else str(response)
            else:
                # Fallback for regular Ollama
                return self._ollama_llm(prompt)
        except Exception as e:
            logger.error(f"Error in LocalLLMWrapper: {e}")
            return "Error generating response"
    
    @property
    def _llm_type(self) -> str:
        return "local_ollama"


class RAGASEvaluator:
    """
    Fixed RAGAS evaluation framework for RAG pipeline assessment
    """
    
    def __init__(self, rag_service: RAGServiceInterface, config):
        self.rag_service = rag_service
        self.config = config
        self.evaluation_results = []
        self.evaluation_history = []
        
        # Initialize local models for RAGAS
        self.evaluator_llm = self._setup_local_llm()
        self.evaluator_embeddings = self._setup_local_embeddings()
        
    def _setup_local_llm(self):
        """Setup LLM for RAGAS evaluation - can use Gemini or Ollama"""
        
        # Check if Gemini API key is available
        gemini_api_key = os.getenv('GOOGLE_API_KEY') or os.getenv('GEMINI_API_KEY')
        use_gemini = gemini_api_key is not None and GEMINI_AVAILABLE
        
        if use_gemini:
            logger.info("Using Gemini for RAGAS evaluation")
            try:
                # Use Gemini for evaluation (more reliable for structured evaluation tasks)
                gemini_llm = ChatGoogleGenerativeAI(
                    model="gemini-2.5-flash",
                    temperature=0.1,
                    google_api_key=gemini_api_key,
                    convert_system_message_to_human=True  # For compatibility
                )
                
                # Test Gemini connection
                try:
                    test_response = gemini_llm.invoke("Test connection")
                    logger.info(f"Gemini LLM initialized successfully for RAGAS evaluation")
                except Exception as e:
                    logger.warning(f"Gemini test failed: {e}")
                
                return gemini_llm
                
            except Exception as e:
                logger.error(f"Failed to initialize Gemini, falling back to Ollama: {e}")
                use_gemini = False
        
        # Fallback to Ollama if Gemini is not available
        if not use_gemini:
            logger.info("Using Ollama for RAGAS evaluation")
            try:
                if not OLLAMA_AVAILABLE:
                    raise ImportError("Ollama LLM not available. Please install langchain-ollama: pip install -U langchain-ollama")
                
                # Create Ollama instance for evaluation
                ollama_llm = OllamaLLM(
                    model=self.config.LLM_MODEL,
                    base_url=self.config.OLLAMA_BASE_URL,
                    temperature=0.1,  # Lower temperature for evaluation consistency
                )
                
                # Wrap it for RAGAS compatibility
                wrapped_llm = LocalLLMWrapper(ollama_llm)
                logger.info(f"Ollama LLM initialized for RAGAS: {self.config.LLM_MODEL}")
                return wrapped_llm
                
            except Exception as e:
                logger.error(f"Failed to setup Ollama LLM for RAGAS: {e}")
                raise
    
    def _setup_local_embeddings(self):
        """Setup local embeddings for RAGAS evaluation"""
        try:
            # Use the same embeddings from model manager
            embeddings = self.rag_service.model_manager.get_embeddings()
            logger.info("Local embeddings initialized for RAGAS")
            return embeddings
            
        except Exception as e:
            logger.error(f"Failed to setup local embeddings for RAGAS: {e}")
            raise
        
    def prepare_evaluation_dataset(
        self,
        questions: List[str],
        ground_truths: Optional[List[List[str]]] = None,
        search_type: SearchType = SearchType.VECTOR,
        use_enhanced_query: bool = False,
        save_intermediate: bool = True
    ) -> Dataset:
        """
        Prepare evaluation dataset by running queries through RAG pipeline
        
        Args:
            questions: List of evaluation questions
            ground_truths: Optional list of ground truth answers (required for context_recall)
            search_type: Type of search to use
            use_enhanced_query: Whether to use query enhancement
            save_intermediate: Whether to save intermediate results to CSV
            
        Returns:
            Dataset ready for RAGAS evaluation
        """
        logger.info(f"Preparing evaluation dataset with {len(questions)} questions")
        
        answers = []
        contexts = []
        processing_times = []
        
        # Prepare intermediate results for CSV saving
        intermediate_results = []
        
        for i, question in enumerate(questions):
            try:
                # Run query through RAG pipeline
                start_time = time.time()
                result = self.rag_service.query(
                    question=question,
                    search_type=search_type,
                    use_enhanced_query=use_enhanced_query,
                    chat_history=[]
                )
                processing_time = time.time() - start_time
                
                # Extract answer
                answer = result["answer"]
                answers.append(answer)
                
                # Extract contexts from source documents
                source_docs = result.get("source_documents", [])
                context_list = [doc.content for doc in source_docs] if source_docs else []
                contexts.append(context_list)
                
                processing_times.append(processing_time)
                
                # Store intermediate result
                intermediate_result = {
                    "question_id": i + 1,
                    "question": question,
                    "answer": answer,
                    "num_contexts": len(context_list),
                    "processing_time": processing_time,
                    "search_type": search_type.value,
                    "enhanced_query": use_enhanced_query,
                    "timestamp": datetime.now().isoformat()
                }
                
                # Add ground truth if available
                if ground_truths and i < len(ground_truths):
                    intermediate_result["ground_truth"] = ground_truths[i][0] if ground_truths[i] else ""
                
                # Add source information
                if source_docs:
                    sources = [doc.metadata.get("source", "Unknown") for doc in source_docs[:3]]
                    intermediate_result["top_sources"] = "; ".join(sources)
                
                intermediate_results.append(intermediate_result)
                
                logger.info(f"Processed question {i+1}/{len(questions)}: {question[:50]}... in {processing_time:.2f}s")
                
            except Exception as e:
                logger.error(f"Failed to process question '{question}': {e}")
                answers.append("Error: Could not generate answer")
                contexts.append([])
                processing_times.append(0.0)
                
                # Add error to intermediate results
                intermediate_results.append({
                    "question_id": i + 1,
                    "question": question,
                    "answer": "Error: Could not generate answer",
                    "error": str(e),
                    "processing_time": 0.0,
                    "search_type": search_type.value,
                    "enhanced_query": use_enhanced_query,
                    "timestamp": datetime.now().isoformat()
                })
        
        # Save intermediate results if requested
        if save_intermediate and intermediate_results:
            self._save_intermediate_results(intermediate_results, search_type)
        
        # Prepare data dictionary for RAGAS
        data = {
            "question": questions,
            "answer": answers,
            "contexts": contexts,
            "processing_time": processing_times,
        }
        
        # Add ground truths if provided - FIXED: Use correct format for newer RAGAS
        if ground_truths:
            if len(ground_truths) != len(questions):
                logger.warning(f"Ground truths count ({len(ground_truths)}) doesn't match questions ({len(questions)})")
                # Pad or truncate ground truths to match questions
                if len(ground_truths) < len(questions):
                    ground_truths.extend([["No ground truth provided"]] * (len(questions) - len(ground_truths)))
                else:
                    ground_truths = ground_truths[:len(questions)]
            
            # Convert List[List[str]] to List[str] for newer RAGAS versions
            ground_truth_strings = []
            for gt in ground_truths:
                if isinstance(gt, list):
                    ground_truth_strings.append(gt[0] if gt else "No ground truth provided")
                else:
                    ground_truth_strings.append(str(gt))
            
            data["ground_truth"] = ground_truth_strings
            logger.info("Ground truths added to dataset as strings")
        
        # Store processing times for analysis
        self.last_processing_times = processing_times
        self.last_intermediate_results = intermediate_results
        
        # Convert to dataset
        dataset = Dataset.from_dict(data)
        logger.info(f"Evaluation dataset prepared with {len(dataset)} samples")
        
        return dataset
    
    def _save_intermediate_results(self, results: List[Dict], search_type: SearchType):
        """Save intermediate results to CSV with pipe delimiter"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"intermediate_results_{search_type.value}_{timestamp}.csv"
            
            df = pd.DataFrame(results)
            df.to_csv(filename, index=False, encoding='utf-8', sep='|')
            logger.info(f"Intermediate results saved to {filename}")
            
        except Exception as e:
            logger.error(f"Failed to save intermediate results: {e}")
    
    def evaluate_pipeline(
        self,
        dataset: Dataset,
        metrics_to_use: Optional[List] = None
    ) -> Dict[str, Any]:
        """
        Evaluate RAG pipeline using RAGAS metrics with local models
        
        Args:
            dataset: Prepared evaluation dataset
            metrics_to_use: List of metrics to evaluate (defaults to all available)
            
        Returns:
            Dictionary containing evaluation results
        """
        logger.info("Starting RAGAS evaluation with local models")
        
        # Determine which metrics to use
        if metrics_to_use is None:
            metrics_to_use = [faithfulness, answer_relevancy]
            
            # Add context_recall only if ground truths are available
            if "ground_truth" in dataset.column_names:
                metrics_to_use.append(context_recall)
                logger.info("Ground truths available - including context_recall metric")
            
            # Add context_precision (doesn't require ground truth)
            metrics_to_use.append(context_precision)
        
        try:
            # Run evaluation with local models
            start_time = time.time()
            
            logger.info("Running RAGAS evaluation...")
            result = evaluate(
                dataset=dataset,
                metrics=metrics_to_use,
                llm=self.evaluator_llm,
                embeddings=self.evaluator_embeddings,
                # Disable any OpenAI fallbacks
                raise_exceptions=False
            )
            
            evaluation_time = time.time() - start_time
            
            # Convert to pandas for easier analysis
            df = result.to_pandas()
            
            # Calculate aggregate metrics
            evaluation_summary = {
                "timestamp": datetime.now().isoformat(),
                "num_samples": len(df),
                "evaluation_time": evaluation_time,
                "metrics": {}
            }
            
            # Add individual metric scores
            for metric in metrics_to_use:
                metric_name = metric.__class__.__name__
                if metric_name in df.columns:
                    scores = df[metric_name].dropna()
                    if len(scores) > 0:
                        evaluation_summary["metrics"][metric_name] = {
                            "mean": float(scores.mean()),
                            "min": float(scores.min()),
                            "max": float(scores.max()),
                            "std": float(scores.std()),
                            "count": len(scores)
                        }
                    else:
                        evaluation_summary["metrics"][metric_name] = {
                            "error": "No valid scores generated"
                        }
            
            # Add processing time stats if available
            if hasattr(self, 'last_processing_times'):
                evaluation_summary["processing_times"] = {
                    "mean": sum(self.last_processing_times) / len(self.last_processing_times),
                    "total": sum(self.last_processing_times)
                }
            
            # Store results
            self.evaluation_results.append(evaluation_summary)
            self.last_evaluation_df = df
            
            logger.info(f"Evaluation completed in {evaluation_time:.2f}s")
            
            return evaluation_summary
            
        except Exception as e:
            logger.error(f"RAGAS evaluation failed: {e}")
            # Save what we have so far
            if hasattr(self, 'last_intermediate_results'):
                self._save_intermediate_results(self.last_intermediate_results, SearchType.VECTOR)
            raise
    
    def evaluate_with_test_set(
        self,
        test_questions: List[str],
        test_ground_truths: Optional[List[List[str]]] = None,
        search_type: SearchType = SearchType.VECTOR,
        use_enhanced_query: bool = False
    ) -> Dict[str, Any]:
        """
        Convenience method to prepare dataset and evaluate in one step
        
        Args:
            test_questions: List of test questions
            test_ground_truths: Optional ground truth answers
            search_type: Search type to use
            use_enhanced_query: Whether to use query enhancement
            
        Returns:
            Evaluation results dictionary
        """
        # Prepare dataset with intermediate saving
        dataset = self.prepare_evaluation_dataset(
            questions=test_questions,
            ground_truths=test_ground_truths,
            search_type=search_type,
            use_enhanced_query=use_enhanced_query,
            save_intermediate=True
        )
        
        # Evaluate
        return self.evaluate_pipeline(dataset)
    
    def get_detailed_results(self) -> Optional[pd.DataFrame]:
        """
        Get detailed results from last evaluation as DataFrame
        
        Returns:
            DataFrame with detailed evaluation results or None
        """
        if hasattr(self, 'last_evaluation_df'):
            return self.last_evaluation_df
        return None
    
    def save_evaluation_results(self, filepath: Optional[str] = None) -> str:
        """
        Save evaluation results to file with pipe delimiter
        
        Args:
            filepath: Optional filepath (defaults to evaluation_results_{timestamp}.csv)
            
        Returns:
            Path where results were saved
        """
        if not hasattr(self, 'last_evaluation_df'):
            raise ValueError("No evaluation results to save")
        
        if filepath is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = f"evaluation_results_{timestamp}.csv"
        
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        # Save with UTF-8 encoding and pipe delimiter for Indonesian text
        self.last_evaluation_df.to_csv(filepath, index=False, encoding='utf-8', sep='|')
        logger.info(f"Evaluation results saved to {filepath}")
        
        return str(filepath)
    
    def get_evaluation_summary(self) -> Dict[str, Any]:
        """
        Get summary of all evaluations performed
        
        Returns:
            Dictionary with evaluation history and statistics
        """
        if not self.evaluation_results:
            return {"message": "No evaluations performed yet"}
        
        latest = self.evaluation_results[-1]
        
        summary = {
            "total_evaluations": len(self.evaluation_results),
            "latest_evaluation": latest,
            "metric_trends": {}
        }
        
        # Calculate trends if multiple evaluations
        if len(self.evaluation_results) > 1:
            for metric_name in latest["metrics"].keys():
                scores = [
                    eval_result["metrics"].get(metric_name, {}).get("mean", 0)
                    for eval_result in self.evaluation_results
                ]
                summary["metric_trends"][metric_name] = {
                    "scores": scores,
                    "improvement": scores[-1] - scores[0] if len(scores) > 1 else 0
                }
        
        return summary
    
    def compare_search_types(
        self,
        test_questions: List[str],
        test_ground_truths: Optional[List[List[str]]] = None
    ) -> Dict[str, Any]:
        """
        Compare performance across different search types
        
        Args:
            test_questions: Questions to test
            test_ground_truths: Optional ground truths
            
        Returns:
            Comparison results dictionary
        """
        comparison_results = {}
        
        search_types_to_test = [SearchType.VECTOR]
        
        # Add other search types if available
        if hasattr(self.rag_service, 'graph_service') and self.rag_service.graph_service:
            if self.rag_service.graph_service.has_data():
                search_types_to_test.extend([SearchType.GRAPH, SearchType.HYBRID])
        
        for search_type in search_types_to_test:
            logger.info(f"Evaluating search type: {search_type.value}")
            
            try:
                result = self.evaluate_with_test_set(
                    test_questions=test_questions,
                    test_ground_truths=test_ground_truths,
                    search_type=search_type,
                    use_enhanced_query=False
                )
                comparison_results[search_type.value] = result
            except Exception as e:
                logger.error(f"Failed to evaluate {search_type.value}: {e}")
                comparison_results[search_type.value] = {"error": str(e)}
        
        return comparison_results