import os
import logging
from bm25_retriever import BM25MultiHopRetriever

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def print_results(question: str, answer: str, state, target_docs: int):
    """Print retrieval results in a formatted way."""
    print("\n" + "-" * 80 + "\n")
    print(f"Question: {question}\n")
    print(f"Target context documents: {target_docs}\n")
    print(f"Answer: {answer}\n")
    
    print("Document Lists:\n")
    print("Documents in context:")
    for hop_num, title, _ in state.hops:
        url = f"https://en.wikipedia.org/wiki/{title.replace(' ', '_')}"
        print(f"Hop {hop_num}: {title}")
        print(f"URL: {url}\n")

def test_retriever(question: str, target_docs: int):
    """Test the retriever with a given question and target number of context documents."""
    logger.info(f"\nTesting question: {question}")
    logger.info(f"Target context documents: {target_docs}")
    logger.info("-" * 80)
    
    # Get answer and context
    answer, state = retriever.retrieve(question, target_docs)
    
    # Print results
    print_results(question, answer, state, target_docs)

def main():
    """Run example queries with the BM25 multi-hop retriever."""
    # Initialize retriever with Gemini API key from environment
    gemini_api_key = os.getenv('GEMINI_API_KEY')
    if not gemini_api_key:
        print("Please set GEMINI_API_KEY environment variable")
        return
        
    global retriever
    retriever = BM25MultiHopRetriever(
        gemini_api_key=gemini_api_key,
        n_iterations=5,
        n_queries=3,
        n_docs=1,  # Only get top document per iteration
        doc_content_chars_max=2000
    )
    
    # Test cases with questions and their target number of documents
    test_cases = [
        ("What was the impact of Alan Turing's work on the development of artificial intelligence?", 5),
        ("How did Marie Curie's discoveries influence our understanding of radioactivity?", 3),
        ("What role did the Medici family play in the Italian Renaissance?", 4)
    ]
    
    # Run each test case
    for question, target_docs in test_cases:
        print("\nTesting question:", question)
        print("Target context documents:", target_docs)
        print("-" * 80)
        
        # Get answer and state
        answer, state = retriever.retrieve(question, target_docs)
        
        # Calculate token usage
        total_tokens_processed = sum(len(doc) for _, _, doc in state.context_docs)
        context_tokens = min(10000, total_tokens_processed)  # Cap at 10k tokens
        api_tokens = len(question) + len(answer)
        
        print("\nTotal tokens processed:", total_tokens_processed)
        print("Tokens in context:", context_tokens)
        print("Total tokens used in API calls:", api_tokens)
        print("\n" + "-" * 80 + "\n")
        print("Question:", question)
        print("\nTarget context documents:", target_docs)
        print("\nAnswer:", answer)
        print("\nDocument Lists:\n")
        print("Documents in context:")
        for hop_num, title, _ in sorted(state.context_docs):
            print(f"Hop {hop_num}: {title}")
            print(f"URL: https://en.wikipedia.org/wiki/{title.replace(' ', '_')}")
        print("\n")

if __name__ == "__main__":
    main()
