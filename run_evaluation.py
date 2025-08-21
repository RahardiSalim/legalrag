import pandas as pd
import logging
import asyncio
from typing import List, Tuple
import os
from pathlib import Path

# Import necessary classes from your project
from config import Config
from model_manager import ModelManager
from vector_store_manager import VectorStoreManager
from rag_service import RAGService
from evaluation import RAGASEvaluator
from models import SearchType

# Configure logging to see the progress
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# --- Configuration ---
GROUND_TRUTH_FILE_PATH = "D:/RAG/q&a_slik_reporting.csv"  # IMPORTANT: Change this to your file path
QUESTION_COLUMN_NAME = "Pertanyaan"  # IMPORTANT: Change to your question column name
GROUND_TRUTH_COLUMN_NAME = "Jawaban"  # IMPORTANT: Change to your ground truth column name
RESULTS_OUTPUT_PATH = "evaluation_results_vector_search.csv"

# CSV Delimiter Configuration
CSV_DELIMITER = "|"  # Change this to match your CSV format (common options: ",", ";", "|", "\t")

# Evaluation Settings
LIMIT_QUESTIONS = 2  # Set to None to use all questions, or set a number to limit for testing
SAVE_INTERMEDIATE_RESULTS = True  # Save results progressively to avoid loss

# Gemini API Configuration (Optional - for better evaluation accuracy)
# Set your Gemini API key as environment variable or uncomment and set directly:
os.environ['GOOGLE_API_KEY'] = 'AIzaSyAX9QYngTF_XC2QIOQ1-oJc-Ic10yPxd-E'
# If not set, evaluation will use Ollama instead


def load_ground_truth_data(filepath: str) -> Tuple[List[str], List[List[str]]]:
    """
    Loads questions and ground truths from a CSV file with robust error handling.
    
    Args:
        filepath: Path to the CSV file
        
    Returns:
        Tuple of (questions, ground_truths) where ground_truths is List[List[str]]
    """
    try:
        # Check if file exists
        if not os.path.exists(filepath):
            logger.error(f"File not found: {filepath}")
            return [], []
        
        # Try different encodings
        encodings_to_try = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']
        df = None
        
        for encoding in encodings_to_try:
            try:
                logger.info(f"Attempting to read CSV with encoding: {encoding}")
                df = pd.read_csv(filepath, delimiter=CSV_DELIMITER, encoding=encoding)
                logger.info(f"Successfully loaded CSV with encoding: {encoding}")
                break
            except UnicodeDecodeError:
                logger.warning(f"Failed to read with encoding: {encoding}")
                continue
            except Exception as e:
                logger.warning(f"Error with encoding {encoding}: {e}")
                continue
        
        if df is None:
            logger.error("Could not read CSV file with any encoding")
            return [], []
        
        logger.info(f"CSV loaded successfully. Shape: {df.shape}")
        logger.info(f"Columns: {list(df.columns)}")
        
        # Check if required columns exist
        if QUESTION_COLUMN_NAME not in df.columns:
            logger.error(f"Question column '{QUESTION_COLUMN_NAME}' not found. Available columns: {list(df.columns)}")
            return [], []
            
        if GROUND_TRUTH_COLUMN_NAME not in df.columns:
            logger.error(f"Ground truth column '{GROUND_TRUTH_COLUMN_NAME}' not found. Available columns: {list(df.columns)}")
            return [], []
        
        # Log sample of data for verification
        logger.info("Sample data:")
        for i, row in df.head(2).iterrows():
            logger.info(f"  Row {i}: Q='{str(row[QUESTION_COLUMN_NAME])[:50]}...', A='{str(row[GROUND_TRUTH_COLUMN_NAME])[:50]}...'")
        
        # Apply limit if specified
        if LIMIT_QUESTIONS is not None and LIMIT_QUESTIONS > 0:
            original_len = len(df)
            df = df.head(LIMIT_QUESTIONS)
            logger.info(f"Limited dataset from {original_len} to {len(df)} questions for testing")
        
        # Clean and filter data
        initial_count = len(df)
        
        # Remove rows with missing values in essential columns
        df = df.dropna(subset=[QUESTION_COLUMN_NAME, GROUND_TRUTH_COLUMN_NAME])
        after_dropna = len(df)
        
        if after_dropna < initial_count:
            logger.warning(f"Dropped {initial_count - after_dropna} rows with missing values")
        
        # Remove empty strings
        df = df[df[QUESTION_COLUMN_NAME].str.strip().str.len() > 0]
        df = df[df[GROUND_TRUTH_COLUMN_NAME].str.strip().str.len() > 0]
        final_count = len(df)
        
        if final_count < after_dropna:
            logger.warning(f"Dropped {after_dropna - final_count} rows with empty strings")
        
        # Extract questions and ground truths
        questions = df[QUESTION_COLUMN_NAME].astype(str).str.strip().tolist()
        
        # For newer RAGAS versions, ground truths should be List[List[str]] initially
        # but will be converted to List[str] in the evaluation module
        raw_ground_truths = df[GROUND_TRUTH_COLUMN_NAME].astype(str).str.strip().tolist()
        ground_truths = [[gt] for gt in raw_ground_truths]
        
        logger.info(f"Successfully loaded {len(questions)} questions and ground truths from {filepath}")
        
        # Save a backup copy of processed data
        if SAVE_INTERMEDIATE_RESULTS:
            try:
                processed_df = pd.DataFrame({
                    'question': questions,
                    'ground_truth': raw_ground_truths  # Save as strings for readability
                })
                backup_path = f"processed_evaluation_data_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv"
                processed_df.to_csv(backup_path, index=False, encoding='utf-8', sep='|')
                logger.info(f"Saved processed evaluation data to {backup_path}")
            except Exception as e:
                logger.warning(f"Could not save backup copy: {e}")
        
        return questions, ground_truths
        
    except Exception as e:
        logger.error(f"Error loading ground truth data: {e}")
        logger.error(f"Please check the file path: {filepath}")
        logger.error(f"And verify column names: '{QUESTION_COLUMN_NAME}', '{GROUND_TRUTH_COLUMN_NAME}'")
        return [], []


async def main():
    """Main function to run the RAG evaluation with improved error handling."""
    logger.info("🚀 Starting RAG evaluation for Vector Search...")

    try:
        # 1. Load the evaluation dataset
        logger.info(f"Loading evaluation data from: {GROUND_TRUTH_FILE_PATH}")
        test_questions, test_ground_truths = load_ground_truth_data(GROUND_TRUTH_FILE_PATH)
        
        if not test_questions:
            logger.error("No data to evaluate. Please check your file path and column names.")
            logger.error(f"File path: {GROUND_TRUTH_FILE_PATH}")
            logger.error(f"Question column: {QUESTION_COLUMN_NAME}")
            logger.error(f"Ground truth column: {GROUND_TRUTH_COLUMN_NAME}")
            return

        logger.info(f"Loaded {len(test_questions)} questions for evaluation")

        # 2. Initialize the RAG system components
        logger.info("Initializing RAG system components...")
        try:
            config = Config()
            logger.info("✅ Config initialized")
            
            model_manager = ModelManager(config)
            logger.info("✅ Model manager initialized")
            
            vector_store_manager = VectorStoreManager(config, model_manager)
            logger.info("✅ Vector store manager initialized")
            
            # IMPORTANT: Ensure the vector store is loaded before evaluation
            logger.info("Loading vector store...")
            if not vector_store_manager.load_store():
                logger.error("❌ Failed to load vector store. Cannot proceed with evaluation.")
                logger.error("Please make sure you have indexed documents in the vector store first.")
                return
            
            logger.info("✅ Vector store loaded successfully")
            
            rag_service = RAGService(config, model_manager, vector_store_manager)
            rag_service.setup_chain()
            logger.info("✅ RAG Service initialized successfully")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize RAG service: {e}")
            logger.error("Make sure all required models are available and Ollama is running.")
            return

        # 3. Instantiate the evaluator with local models
        logger.info("Initializing RAGAS evaluator with local models...")
        try:
            evaluator = RAGASEvaluator(rag_service, config)
            logger.info("✅ RAGAS Evaluator initialized with local models")
        except Exception as e:
            logger.error(f"❌ Failed to initialize RAGAS evaluator: {e}")
            return

        # 4. Run the evaluation using the vector search approach
        logger.info(f"Running evaluation with {len(test_questions)} questions using VECTOR search...")
        logger.info("This may take several minutes depending on the number of questions...")
        
        try:
            evaluation_summary = evaluator.evaluate_with_test_set(
                test_questions=test_questions,
                test_ground_truths=test_ground_truths,
                search_type=SearchType.VECTOR,
                use_enhanced_query=False
            )
            
            # 5. Print the summary results
            logger.info("\n" + "="*50)
            logger.info("RAGAS EVALUATION SUMMARY")
            logger.info("="*50)
            
            print(f"\n📊 Evaluation Results:")
            print(f"   • Total Questions: {evaluation_summary.get('num_samples', 0)}")
            print(f"   • Evaluation Time: {evaluation_summary.get('evaluation_time', 0):.2f}s")
            
            if 'processing_times' in evaluation_summary:
                pt = evaluation_summary['processing_times']
                print(f"   • Average Processing Time: {pt.get('mean', 0):.2f}s per question")
                print(f"   • Total Processing Time: {pt.get('total', 0):.2f}s")
            
            print(f"\n📈 Metric Scores:")
            for metric_name, scores in evaluation_summary.get('metrics', {}).items():
                if 'error' in scores:
                    print(f"   • {metric_name}: ❌ {scores['error']}")
                else:
                    print(f"   • {metric_name}:")
                    print(f"     - Mean: {scores.get('mean', 0):.4f}")
                    print(f"     - Min:  {scores.get('min', 0):.4f}")
                    print(f"     - Max:  {scores.get('max', 0):.4f}")
                    print(f"     - Std:  {scores.get('std', 0):.4f}")
            
            logger.info("="*50 + "\n")

            # 6. Save the detailed results to a CSV file
            logger.info("Saving detailed evaluation results...")
            try:
                saved_path = evaluator.save_evaluation_results(RESULTS_OUTPUT_PATH)
                logger.info(f"✅ Detailed results saved to: {saved_path}")
                
                # Also save the summary as JSON for easy reference
                import json
                summary_path = RESULTS_OUTPUT_PATH.replace('.csv', '_summary.json')
                with open(summary_path, 'w', encoding='utf-8') as f:
                    json.dump(evaluation_summary, f, indent=2, ensure_ascii=False)
                logger.info(f"✅ Summary saved to: {summary_path}")
                
            except Exception as e:
                logger.error(f"❌ Failed to save results: {e}")
        
        except Exception as e:
            logger.error(f"❌ Evaluation failed: {e}")
            logger.error("Check if your models are properly loaded and Ollama is running.")
            
            # Try to save any intermediate results that were generated
            if hasattr(evaluator, 'last_intermediate_results'):
                try:
                    logger.info("Attempting to save intermediate results...")
                    import pandas as pd
                    df_intermediate = pd.DataFrame(evaluator.last_intermediate_results)
                    emergency_path = f"emergency_results_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv"
                    df_intermediate.to_csv(emergency_path, index=False, encoding='utf-8')
                    logger.info(f"✅ Emergency results saved to: {emergency_path}")
                except Exception as save_error:
                    logger.error(f"Could not save emergency results: {save_error}")
            
            return

        # 7. Optional: Run comparison across different search types if available
        logger.info("Evaluation completed successfully!")
        
        # Check if user wants to compare search types
        if hasattr(rag_service, 'graph_service') and rag_service.graph_service:
            if rag_service.graph_service.has_data():
                logger.info("Graph data detected. You can run comparison across search types by uncommenting the code below.")
                # Uncomment the following lines to compare search types:
                # logger.info("Running comparison across search types...")
                # comparison_results = evaluator.compare_search_types(test_questions, test_ground_truths)
                # logger.info("Search type comparison completed!")
        
    except KeyboardInterrupt:
        logger.warning("Evaluation interrupted by user. Attempting to save any progress...")
        if 'evaluator' in locals() and hasattr(evaluator, 'last_intermediate_results'):
            try:
                import pandas as pd
                df_intermediate = pd.DataFrame(evaluator.last_intermediate_results)
                interrupted_path = f"interrupted_results_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv"
                df_intermediate.to_csv(interrupted_path, index=False, encoding='utf-8', sep='|')
                logger.info(f"Saved interrupted results to: {interrupted_path}")
            except Exception as e:
                logger.error(f"Could not save interrupted results: {e}")
                
    except Exception as e:
        logger.error(f"Unexpected error during evaluation: {e}")
        logger.error("Please check your configuration and ensure all dependencies are properly installed.")


if __name__ == "__main__":
    print("\n" + "="*60)
    print("RAG SYSTEM EVALUATION SCRIPT")
    print("="*60)
    print(f"Ground Truth File: {GROUND_TRUTH_FILE_PATH}")
    print(f"Question Column: {QUESTION_COLUMN_NAME}")
    print(f"Answer Column: {GROUND_TRUTH_COLUMN_NAME}")
    print(f"CSV Delimiter: '{CSV_DELIMITER}'")
    print(f"Question Limit: {LIMIT_QUESTIONS if LIMIT_QUESTIONS else 'All questions'}")
    print(f"Output Path: {RESULTS_OUTPUT_PATH}")
    print("="*60 + "\n")
    
    # Verify file exists before starting
    if not os.path.exists(GROUND_TRUTH_FILE_PATH):
        print(f"ERROR: Ground truth file not found at: {GROUND_TRUTH_FILE_PATH}")
        print("Please update the GROUND_TRUTH_FILE_PATH variable in the script.")
        exit(1)
    
    asyncio.run(main())