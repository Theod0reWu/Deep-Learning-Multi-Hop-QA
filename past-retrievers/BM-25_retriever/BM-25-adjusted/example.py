import sys
import os

# Add the current directory to the Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from bm25_retriever import BM25MultiHopRetriever
import logging
from dotenv import load_dotenv

print("Imported BM25MultiHopRetriever from:", BM25MultiHopRetriever.__module__)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def print_results(question: str, answer: str, state):
    """Print retrieval results in a formatted way."""
    print("\n" + "-" * 80)
    
    # Print results
    print("\nQuestion:", question)
    print(f"\nNumber of iterations: {state.current_iteration}")
    print("\nAnswer:", answer)
    
    print("\nDocument Lists:")
    print("\nDocuments in context:")
    for doc in state.context_docs:
        url = state.page_urls.get(doc, f"https://en.wikipedia.org/wiki/{doc.replace(' ', '_')}")
        print(f"- {doc}: {url}")
    
    print("\nAll documents processed:")
    # Check for duplicates
    from collections import Counter
    all_docs = Counter(state.processed_docs)
    duplicates = {doc: count for doc, count in all_docs.items() if count > 1}
    
    for doc in state.processed_docs:
        url = state.page_urls.get(doc, f"https://en.wikipedia.org/wiki/{doc.replace(' ', '_')}")
        print(f"- {doc}: {url}")
    
    if duplicates:
        print("\nWARNING: Found duplicate documents in processed_docs:")
        for doc, count in duplicates.items():
            print(f"- {doc}: appeared {count} times")
    else:
        print("\nNo duplicates found in processed_docs")
    
    print("\nQueries made:", state.queries_made)
    
    print("\nContext History:")
    for i, context in enumerate(state.context_history, 1):
        print(f"\nHop {i}:")
        print(context[:200] + "...")
    
    print("-" * 80)

def test_retriever(question: str, n_iterations: int = 5):
    """Test the retriever with a given question and number of iterations."""
    logger.info(f"\nTesting question: {question}")
    logger.info(f"Number of iterations: {n_iterations}")
    logger.info("-" * 80)
    
    # Get answer and context
    answer, state = retriever.retrieve(question, n_iterations)
    
    # Print results
    print_results(question, answer, state)

def main():
    # Initialize the retriever
    global retriever
    retriever = BM25MultiHopRetriever(
        n_iterations=5,  # Maximum iterations allowed
        n_queries=5,     # Number of queries per iteration
        n_docs=1         # Number of documents to add per iteration
    )
    
    # Test cases
    test_cases = [
        ("What was the impact of Alan Turing's work on the development of artificial intelligence?", 5),
        ("How did Marie Curie's discoveries influence our understanding of radioactivity?", 3),
        # ("What role did the Medici family play in the Italian Renaissance?", 4)
    ]
    
    # Run tests
    for question, n_iterations in test_cases:
        test_retriever(question, n_iterations)
        print("\n")

if __name__ == "__main__":
    main()
