import argparse
import logging
from bm25_retriever import BM25MultiHopRetriever

def main():
    parser = argparse.ArgumentParser(description='BM25 Multi-hop Retriever Example')
    parser.add_argument('--steps', type=int, default=3, help='Number of retrieval steps')
    parser.add_argument('--queries_per_step', type=int, default=2, help='Queries per step')
    parser.add_argument('--docs_per_query', type=int, default=2, help='Documents per query')
    args = parser.parse_args()

    # Example question
    question = "What was the relationship between Alan Turing and Christopher Morcom, and how did it influence Turing's later work?"

    print(f"\nProcessing question: {question}")
    print(f"Using {args.steps} retrieval steps...")

    try:
        # Initialize retriever (will get API key from environment)
        retriever = BM25MultiHopRetriever()
        
        # Perform retrieval
        answer, context_history, visited_pages = retriever.retrieve(
            question=question,
            num_iterations=args.steps,
            queries_per_iteration=args.queries_per_step,
            docs_per_query=args.docs_per_query
        )

        # Print results
        print("\nRetrieval Settings:")
        print(f"Steps: {args.steps}")
        print(f"Queries per step: {args.queries_per_step}")
        print(f"Docs per query: {args.docs_per_query}\n")

        print("Visited Wikipedia pages:")
        for page in visited_pages:
            print(f"- {page}")

        print(f"\nTotal documents retrieved: {len(context_history)}")
        print(f"Total unique pages visited: {len(visited_pages)}")
        
        print("\nAnswer:")
        print(answer)

    except Exception as e:
        logging.error(f"Error: {str(e)}")
        return 1

    return 0

if __name__ == "__main__":
    exit(main())
