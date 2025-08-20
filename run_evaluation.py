import pandas as pd
import logging
import asyncio
from typing import List

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

def load_ground_truth_data(filepath: str) -> (List[str], List[List[str]]):
    """Loads questions and ground truths from a CSV file."""
    try:
        df = pd.read_csv(filepath, delimiter='|')  # For example, using semicolon as the delimiter
        df = df.iloc[:5]
        # Ensure the columns exist
        if QUESTION_COLUMN_NAME not in df.columns or GROUND_TRUTH_COLUMN_NAME not in df.columns:
            raise ValueError(f"CSV must contain '{QUESTION_COLUMN_NAME}' and '{GROUND_TRUTH_COLUMN_NAME}' columns.")
            
        # Drop rows with missing values in essential columns
        df.dropna(subset=[QUESTION_COLUMN_NAME, GROUND_TRUTH_COLUMN_NAME], inplace=True)

        questions = df[QUESTION_COLUMN_NAME].tolist()
        
        # RAGAS expects ground truths as a list of lists of strings
        # e.g., [["answer1"], ["answer2"], ...]
        ground_truths = [[str(ans)] for ans in df[GROUND_TRUTH_COLUMN_NAME].tolist()]
        
        logger.info(f"Loaded {len(questions)} questions from {filepath}")
        return questions, ground_truths
    except FileNotFoundError:
        logger.error(f"Evaluation data file not found at: {filepath}")
        return [], []
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        return [], []

async def main():
    """Main function to run the RAG evaluation."""
    logger.info("🚀 Starting RAG evaluation for Vector Search...")

    # 1. Load the evaluation dataset
    test_questions, test_ground_truths = load_ground_truth_data(GROUND_TRUTH_FILE_PATH)
    if not test_questions:
        logger.error("No data to evaluate. Exiting.")
        return

    # 2. Initialize the RAG system components
    try:
        config = Config()
        model_manager = ModelManager(config)
        vector_store_manager = VectorStoreManager(config, model_manager)
        
        # IMPORTANT: Ensure the vector store is loaded before evaluation
        if not vector_store_manager.load_store():
            logger.error("Failed to load vector store. Cannot proceed with evaluation.")
            return
            
        rag_service = RAGService(config, model_manager, vector_store_manager)
        rag_service.setup_chain()
        logger.info("✅ RAG Service initialized successfully.")
    except Exception as e:
        logger.error(f"Failed to initialize RAG service: {e}")
        return

    # 3. Instantiate the evaluator
    evaluator = RAGASEvaluator(rag_service, config)
    logger.info("✅ RAGAS Evaluator initialized.")

    # 4. Run the evaluation using the vector search approach
    logger.info(f"Running evaluation with {len(test_questions)} questions on VECTOR search type...")
    try:
        evaluation_summary = evaluator.evaluate_with_test_set(
            test_questions=test_questions,
            test_ground_truths=test_ground_truths,
            search_type=SearchType.VECTOR,  # Specify vector-only approach
            use_enhanced_query=False
        )
        
        # 5. Print the summary results
        logger.info("\n--- RAGAS Evaluation Summary ---")
        print(evaluation_summary)
        logger.info("----------------------------------\n")

        # 6. Save the detailed results to a CSV file
        logger.info("Saving detailed evaluation results...")
        saved_path = evaluator.save_evaluation_results(RESULTS_OUTPUT_PATH)
        logger.info(f"✅ Detailed results saved to: {saved_path}")
        
    except Exception as e:
        logger.error(f"An error occurred during evaluation: {e}")


if __name__ == "__main__":
    asyncio.run(main())