import os
import logging
from bm25_retriever import BM25MultiHopRetriever

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_retriever(question: str, target_docs: int):
    """Test the retriever with a given question."""
    logger.info(f"\nTesting question: {question}")
    logger.info(f"Target context documents: {target_docs}")
    logger.info("-" * 80)
    
    # Get answer and state
    answer, context_history, visited_pages, processed_docs, context_docs, query_history = retriever.retrieve(question, target_docs)
    
    # Calculate token usage
    total_tokens_processed = sum(len(content) for _, content in processed_docs)
    context_tokens = sum(len(entry) for entry in context_history)  # Ensure full content length is counted
    
    # Calculate Gemini model tokens: initial question + queries + final answer
    gemini_tokens = len(question)  # Initial question
    for _, query, _ in query_history:  # All generated queries
        gemini_tokens += len(query)
    gemini_tokens += len(answer)  # Final answer
    gemini_tokens += context_tokens  # Add context tokens to Gemini model calls
    
    print("\nToken Usage Statistics:")
    print(f"Total tokens processed (Wikipedia): {total_tokens_processed:,}")
    print(f"Tokens in context: {context_tokens:,}")
    print(f"Tokens used in Gemini model calls: {gemini_tokens:,}")
    print(f"Number of API calls: {len(query_history) + 2}")  # queries + initial question + final answer
    print("\n" + "-" * 80 + "\n")
    print("Question:", question)
    print("\nTarget context documents:", target_docs)
    print("\nAnswer:", answer)
    
    # Print processed documents
    print("\nAll Processed Documents:")
    for title, _ in processed_docs:
        print(f"- {title}")
    
    # Print context documents with queries
    print("\nDocuments in context and their queries:")
    # Create a mapping of iteration to query
    query_map = {iteration: (query, doc_title) for iteration, query, doc_title in query_history}
    
    for iteration, title, _ in sorted(context_docs):
        print(f"\nHop {iteration}:")
        print(f"Document: {title}")
        print(f"URL: https://en.wikipedia.org/wiki/{title.replace(' ', '_')}")
        if iteration in query_map:
            query, _ = query_map[iteration]
            print(f"Query used: {query}")
    print("\n")

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
    )
    
    # Test cases with questions and their target number of documents
    test_cases = [
        ("What was the impact of Alan Turing's work on the development of artificial intelligence?", 5),
        # ("How did Marie Curie's discoveries influence our understanding of radioactivity?", 3),
    ]
    
    # Run each test case
    for question, target_docs in test_cases:
        test_retriever(question, target_docs)

if __name__ == "__main__":
    main()