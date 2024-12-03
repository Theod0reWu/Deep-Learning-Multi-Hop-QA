import argparse
import logging
from bm25_scratch import BM25MultiHopRetriever

# Define test questions with their settings
# TEST_CASES = [
#     {
#         "question": "What was the relationship between Alan Turing and Christopher Morcom?",
#         "target_docs": 3,
#         "queries_per_iteration": 1,
#         "docs_per_query": 2,
#         "score_threshold": 0.8,
#         "max_tokens": 10000
#     },
#     {
#         "question": "How did Albert Einstein's influence work on special relativity",
#         "target_docs": 6,
#         "queries_per_iteration": 1,
#         "docs_per_query": 3,
#         "score_threshold": 0.5,
#         "max_tokens": 25000
#     },
#     # Add more test cases here
# ]
TEST_CASES = [
    {
        "question": "During the year that Serbia became an independent republic after separation from Montenegro, who won the Nobel Prize for Literature?",
        "target_docs": 3,
        "queries_per_iteration": 1,
        "docs_per_query": 2,
        "score_threshold": 0.8,
        "max_tokens": 100000
    },
    {
        "question": "What is the name of the river in the city where Ikea's headquarters are?",
        "target_docs": 6,
        "queries_per_iteration": 1,
        "docs_per_query": 3,
        "score_threshold": 0.5,
        "max_tokens": 100000
    },
    {
        "question": "A disease that had millions of dollars raised for on April 20, 1992, was first recognized by the Center for Disease Control and Prevention (CDC) in what year?",
        "target_docs": 3,
        "queries_per_iteration": 1,
        "docs_per_query": 3,
        "score_threshold": 0.5,
        "max_tokens": 100000
    },
    {
        "question": " what is the difference between the population of Seattle, WA, and Portland, OR",
        "target_docs": 3,
        "queries_per_iteration": 1,
        "docs_per_query": 3,
        "score_threshold": 0.5,
        "max_tokens": 100000
    },
    # Add more test cases here
]

def process_question(retriever, test_case, case_number):
    """Process a single test case"""
    print(f"\n{'='*80}")
    print(f"Test Case {case_number}")
    print(f"{'='*80}")
    
    print(f"\nProcessing question: {test_case['question']}")
    print(f"Settings:")
    print(f"- Target documents: {test_case['target_docs']}")
    print(f"- Queries per iteration: {test_case['queries_per_iteration']}")
    print(f"- Docs per query: {test_case['docs_per_query']}")
    print(f"- Score threshold: {test_case['score_threshold']}")
    print(f"- Max tokens: {test_case['max_tokens']}")

    try:
        # Perform retrieval
        answer, _, visited_pages, context_docs, processed_docs, generated_queries = retriever.retrieve(
            question=test_case['question'],
            num_iterations=test_case['target_docs'],
            queries_per_iteration=test_case['queries_per_iteration'],
            docs_per_query=test_case['docs_per_query'],
            relative_score_threshold=test_case['score_threshold'],
            max_tokens=test_case['max_tokens']
        )

        print("\nGenerated Queries:")
        for query in generated_queries:
            print(f"{query}")

        print("\nDocuments Used in Final Context:")
        for title, _ in context_docs:
            wiki_url = f"https://en.wikipedia.org/wiki/{title.replace(' ', '_')}"
            print(f"- {title}: {wiki_url}")

        print("\nAll Retrieved Documents:")
        for title, _ in processed_docs:
            wiki_url = f"https://en.wikipedia.org/wiki/{title.replace(' ', '_')}"
            print(f"- {title}: {wiki_url}")

        print("\nFinal Answer:")
        print(answer)

        print("\nRaw Context (500 tokens per document):")
        print("-" * 80)
        for title, content in context_docs:
            print(f"\n### Document: {title} ###")
            print(content[:500] + "..." if len(content) > 500 else content)
            print("-" * 80)

        print("\nContext Used:")
        for i, doc in enumerate(context_docs, 1):
            print(f"Document {i}: {doc[0]}")

        print(f"\nRetrieval Statistics:")
        print(f"Documents in final context: {len(context_docs)}")
        print(f"Total documents retrieved: {len(set([title for title, _ in processed_docs]))}")
        print(f"Total unique pages visited: {len(visited_pages)}")
        print(f"Total queries generated: {len(generated_queries)}")
        
        print(f"\nCache Performance:")
        print(f"Cache hits: {retriever.cache_hits}")
        print(f"Cache misses: {retriever.cache_misses}")
        hit_rate = retriever.cache_hits / (retriever.cache_hits + retriever.cache_misses) * 100 if (retriever.cache_hits + retriever.cache_misses) > 0 else 0
        print(f"Cache hit rate: {hit_rate:.1f}%")

        print(f"\nToken Usage:")
        print(f"Total input tokens: {retriever.gemini_tokens_in}")
        print(f"Total output tokens: {retriever.gemini_tokens_out}")
        print(f"Total tokens: {retriever.gemini_tokens_in + retriever.gemini_tokens_out}")

    except Exception as e:
        print(f"Error processing question: {str(e)}")

def main():
    parser = argparse.ArgumentParser(description='BM25 Multi-hop Retriever Example')
    parser.add_argument('--case', type=int, help='Run specific test case number (1-based index)')
    parser.add_argument('--all', action='store_true', help='Run all test cases')
    args = parser.parse_args()

    # Initialize retriever (will get API key from environment)
    retriever = BM25MultiHopRetriever()

    if args.case:
        if 1 <= args.case <= len(TEST_CASES):
            process_question(retriever, TEST_CASES[args.case-1], args.case)
        else:
            print(f"Invalid test case number. Please choose between 1 and {len(TEST_CASES)}")
    elif args.all:
        for i, test_case in enumerate(TEST_CASES, 1):
            process_question(retriever, test_case, i)
    else:
        print("Please specify either --case NUMBER to run a specific test case or --all to run all test cases")

if __name__ == "__main__":
    exit(main())
