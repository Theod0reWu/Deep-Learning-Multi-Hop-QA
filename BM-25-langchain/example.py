from bm25_retriever import BM25MultiHopRetriever
import logging
import os
from dotenv import load_dotenv

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_retriever(question: str):
    """Test the retriever with a given question."""
    logger.info(f"\nTesting question: {question}")
    logger.info("-" * 80)
    
    # Get answer and context
    answer, context_history, visited_pages = retriever.retrieve(question)
    
    # Print results
    print("\nQuestion:", question)
    print("\nAnswer:", answer)
    print("\nVisited Pages:", visited_pages)
    print("\nContext History:")
    for i, context in enumerate(context_history, 1):
        print(f"\nHop {i}:")
        print(context[:200] + "...")  # Print first 200 chars of each context
    
    print("\nWorkflow visualization saved to:", os.path.abspath("retrieval_workflow.png"))
    print("-" * 80)

def main():
    # Initialize the retriever
    global retriever
    retriever = BM25MultiHopRetriever()
    
    # Test cases
    test_questions = [
        "What was the impact of Alan Turing's work on the development of artificial intelligence?",
        "How did the invention of the printing press influence the Protestant Reformation?",
        "What role did the Rosetta Stone play in understanding ancient Egyptian hieroglyphs?",
        "How did Marie Curie's discoveries influence our understanding of radioactivity?",
    ]
    
    # Run tests
    for question in test_questions:
        test_retriever(question)
        print("\n")  # Add spacing between tests

if __name__ == "__main__":
    main()
