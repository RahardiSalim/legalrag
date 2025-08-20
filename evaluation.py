"""
RAGAS Evaluation Module for RAG System
Provides evaluation metrics for RAG pipeline performance
"""

import logging
import time
from typing import List, Dict, Any, Optional
from datetime import datetime
from pathlib import Path
import pandas as pd
from datasets import Dataset

from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_recall,
    context_precision,
)

from models import QueryRequest, SearchType
from interfaces import RAGServiceInterface

logger = logging.getLogger(__name__)


class RAGASEvaluator:
    """
    RAGAS evaluation framework for RAG pipeline assessment
    """
    
    def __init__(self, rag_service: RAGServiceInterface, config):
        self.rag_service = rag_service
        self.config = config
        self.evaluation_results = []
        self.evaluation_history = []
        
    def prepare_evaluation_dataset(
        self,
        questions: List[str],
        ground_truths: Optional[List[List[str]]] = None,
        search_type: SearchType = SearchType.VECTOR,
        use_enhanced_query: bool = False
    ) -> Dataset:
        """
        Prepare evaluation dataset by running queries through RAG pipeline
        
        Args:
            questions: List of evaluation questions
            ground_truths: Optional list of ground truth answers (required for context_recall)
            search_type: Type of search to use
            use_enhanced_query: Whether to use query enhancement
            
        Returns:
            Dataset ready for RAGAS evaluation
        """
        logger.info(f"Preparing evaluation dataset with {len(questions)} questions")
        
        answers = []
        contexts = []
        processing_times = []
        
        for question in questions:
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
                answers.append(result["answer"])
                
                # Extract contexts from source documents
                source_docs = result.get("source_documents", [])
                context_list = [doc.content for doc in source_docs] if source_docs else []
                contexts.append(context_list)
                
                processing_times.append(processing_time)
                
                logger.debug(f"Processed question: {question[:50]}... in {processing_time:.2f}s")
                
            except Exception as e:
                logger.error(f"Failed to process question '{question}': {e}")
                answers.append("Error: Could not generate answer")
                contexts.append([])
                processing_times.append(0.0)
        
        # Prepare data dictionary
        data = {
            "question": questions,
            "answer": answers,
            "contexts": contexts,
            "processing_time": processing_times,
        }
        
        # Add ground truths if provided
        if ground_truths:
            if len(ground_truths) != len(questions):
                logger.warning(f"Ground truths count ({len(ground_truths)}) doesn't match questions ({len(questions)})")
                ground_truths = ground_truths[:len(questions)] if len(ground_truths) > len(questions) else \
                               ground_truths + [["No ground truth provided"]] * (len(questions) - len(ground_truths))
            data["ground_truths"] = ground_truths
        
        # Store processing times for analysis
        self.last_processing_times = processing_times
        
        # Convert to dataset
        dataset = Dataset.from_dict(data)
        logger.info(f"Evaluation dataset prepared with {len(dataset)} samples")
        
        return dataset
    
    def evaluate_pipeline(
        self,
        dataset: Dataset,
        metrics_to_use: Optional[List] = None
    ) -> Dict[str, Any]:
        """
        Evaluate RAG pipeline using RAGAS metrics
        
        Args:
            dataset: Prepared evaluation dataset
            metrics_to_use: List of metrics to evaluate (defaults to all available)
            
        Returns:
            Dictionary containing evaluation results
        """
        logger.info("Starting RAGAS evaluation")
        
        # Determine which metrics to use
        if metrics_to_use is None:
            metrics_to_use = [context_precision, faithfulness, answer_relevancy]
            
            # Add context_recall only if ground truths are available
            if "ground_truths" in dataset.column_names:
                metrics_to_use.append(context_recall)
                logger.info("Ground truths available - including context_recall metric")
        
        try:
            # Run evaluation
            start_time = time.time()
            result = evaluate(
                dataset=dataset,
                metrics=metrics_to_use,
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
                    evaluation_summary["metrics"][metric_name] = {
                        "mean": float(scores.mean()),
                        "min": float(scores.min()),
                        "max": float(scores.max()),
                        "std": float(scores.std())
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
        # Prepare dataset
        dataset = self.prepare_evaluation_dataset(
            questions=test_questions,
            ground_truths=test_ground_truths,
            search_type=search_type,
            use_enhanced_query=use_enhanced_query
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
        Save evaluation results to file
        
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
        
        self.last_evaluation_df.to_csv(filepath, index=False)
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


# class EvaluationDataGenerator:
#     """
#     Helper class to generate test data for evaluation
#     """
    
#     def __init__(self, rag_service: RAGServiceInterface, config):
#         self.rag_service = rag_service
#         self.config = config
    
#     def generate_test_questions_from_documents(
#         self,
#         num_questions: int = 10,
#         focus_areas: Optional[List[str]] = None
#     ) -> List[str]:
#         """
#         Generate test questions based on indexed documents
        
#         Args:
#             num_questions: Number of questions to generate
#             focus_areas: Optional list of topics to focus on
            
#         Returns:
#             List of generated test questions
#         """
#         # This is a placeholder - in production, you might use an LLM to generate questions
#         # based on the actual document content
        
#         default_questions = [
#             "Apa saja ketentuan utama dalam dokumen ini?",
#             "Bagaimana prosedur yang dijelaskan dalam dokumen?",
#             "Apa sanksi yang disebutkan dalam dokumen?",
#             "Siapa pihak yang bertanggung jawab menurut dokumen?",
#             "Apa definisi istilah penting dalam dokumen?",
#             "Bagaimana mekanisme pengawasan yang diatur?",
#             "Apa hak dan kewajiban yang disebutkan?",
#             "Bagaimana proses penyelesaian sengketa?",
#             "Apa persyaratan yang harus dipenuhi?",
#             "Bagaimana struktur organisasi yang dijelaskan?"
#         ]
        
#         return default_questions[:num_questions]
    
#     def create_benchmark_dataset(
#         self,
#         questions: List[str],
#         ground_truths: List[List[str]],
#         metadata: Optional[Dict[str, Any]] = None
#     ) -> Dict[str, Any]:
#         """
#         Create a benchmark dataset for consistent evaluation
        
#         Args:
#             questions: List of benchmark questions
#             ground_truths: List of ground truth answers
#             metadata: Optional metadata about the dataset
            
#         Returns:
#             Benchmark dataset dictionary
#         """
#         benchmark = {
#             "questions": questions,
#             "ground_truths": ground_truths,
#             "metadata": metadata or {},
#             "created_at": datetime.now().isoformat(),
#             "version": "1.0"
#         }
        
#         # Save to file for reuse
#         timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
#         filepath = Path(f"benchmarks/benchmark_{timestamp}.json")
#         filepath.parent.mkdir(parents=True, exist_ok=True)
        
#         import json
#         with open(filepath, 'w', encoding='utf-8') as f:
#             json.dump(benchmark, f, ensure_ascii=False, indent=2)
        
#         logger.info(f"Benchmark dataset saved to {filepath}")
        
#         return benchmark