import logging
import sys
from bm25_retriever import BM25MultiHopRetriever
import time

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_retriever(question: str, n_iterations: int = 5, n_queries: int = 5, n_docs: int = 2):
    """Test the BM25 retriever with a given question."""
    try:
        logger.info(f"Testing question: {question}")
        logger.info(f"Parameters: n_iterations={n_iterations}, n_queries={n_queries}, n_docs={n_docs}")
        
        # Initialize retriever
        retriever = BM25MultiHopRetriever(
            n_iterations=n_iterations,
            n_queries=n_queries,
            n_docs=n_docs
        )
        
        # Get answer
        answer, context, visited_pages = retriever.retrieve(question)
        
        # Print results
        logger.info("\nResults:")
        logger.info(f"Answer: {answer}")
        logger.info(f"\nNumber of documents retrieved: {len(context)}")
        logger.info(f"Number of unique pages visited: {len(visited_pages)}")
        logger.info("\nVisited pages:")
        for page in visited_pages:
            logger.info(f"- {page}")
        
        return answer, context, visited_pages
        
    except Exception as e:
        logger.error(f"Error processing question: {str(e)}", exc_info=True)
        return None, [], set()

def main():
    """Run example questions through the retriever."""
    # Test questions from different domains
    questions = [
        "What was the relationship between Alan Turing and Christopher Morcom, and how did it influence Turing's later work?",
        "How did the development of the printing press influence the Protestant Reformation?",
        "What role did the Zimmermann Telegram play in bringing the United States into World War I?",
        "How did Marie Curie's discovery of radium influence the development of cancer treatments?",
        "What was the connection between the Space Race and the development of integrated circuits?"
    ]
    
    success = 0
    total = len(questions)
    
    for i, question in enumerate(questions, 1):
        logger.info("\n" + "="*80)
        logger.info(f"Processing question {i}/{total}")
        
        start_time = time.time()
        answer, context, visited_pages = test_retriever(question)
        end_time = time.time()
        
        if answer:
            success += 1
            logger.info(f"Time taken: {end_time - start_time:.2f} seconds")
        else:
            logger.error("Failed to process question")
        
        logger.info("="*80 + "\n")
        
        # Add delay between questions to avoid rate limits
        if i < total:
            time.sleep(2)
    
    logger.info(f"\nProcessing complete: {success}/{total} questions processed successfully")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("\nProcessing interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}", exc_info=True)
        sys.exit(1)
