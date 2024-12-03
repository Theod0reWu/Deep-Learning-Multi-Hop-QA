from bm25_retriever import BM25MultiHopRetriever

def test_step1():
    """Test basic retrieval with short iteration."""
    print("\nStep 1: Testing basic retrieval...")
    question = "What was Alan Turing's early education?"
    
    retriever = BM25MultiHopRetriever()
    answer, context, pages = retriever.retrieve(
        question=question,
        num_iterations=2,
        queries_per_iteration=2,
        docs_per_query=1
    )
    
    print(f"\nQuestion: {question}")
    print(f"\nVisited {len(pages)} pages:")
    for page in pages:
        print(f"- {page}")
    print(f"\nAnswer:\n{answer}")

def test_step2():
    """Test multi-hop retrieval with more complex question."""
    print("\nStep 2: Testing multi-hop retrieval...")
    question = "What was the relationship between Alan Turing and Christopher Morcom, and how did it influence Turing's later work?"
    
    retriever = BM25MultiHopRetriever()
    answer, context, pages = retriever.retrieve(
        question=question,
        num_iterations=3,
        queries_per_iteration=2,
        docs_per_query=2
    )
    
    print(f"\nQuestion: {question}")
    print(f"\nVisited {len(pages)} pages:")
    for page in pages:
        print(f"- {page}")
    print(f"\nAnswer:\n{answer}")

if __name__ == "__main__":
    print("Starting BM25 Multi-hop Retriever Tests...")
    test_step1()
    test_step2()
    print("\nTests completed.")
