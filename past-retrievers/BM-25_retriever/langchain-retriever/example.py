from bm25_retriever import BM25WikiRetriever
from langchain.chains import RetrievalQA
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import os
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    # Load environment variables
    load_dotenv()
    
    # Initialize components
    retriever = BM25WikiRetriever(
        n_docs=5,                    # Number of documents to retrieve per query
        load_max_docs=10,            # Maximum documents to load from Wikipedia
        doc_content_chars_max=2000,  # Maximum characters per document
        max_processed_docs=15,       # Maximum number of processed documents to keep
        gemini_api_key=os.getenv('GEMINI_API_KEY')
    )
    
    # Initialize LLM
    llm = ChatGoogleGenerativeAI(
        model="gemini-pro",
        google_api_key=os.getenv('GEMINI_API_KEY'),
        temperature=0.7
    )
    
    # Create QA chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",  # Simple document combining strategy
        retriever=retriever,
        return_source_documents=True,
        verbose=True
    )
    
    # Test cases
    test_cases = [
        "What was the impact of Alan Turing's work on the development of artificial intelligence?",
        "How did Marie Curie's discoveries influence our understanding of radioactivity?",
    ]
    
    # Run tests
    for question in test_cases:
        logger.info(f"\nTesting question: {question}")
        logger.info("-" * 80)
        
        # Get answer
        result = qa_chain({"query": question})
        
        # Print results
        print("\nQuestion:", question)
        print("\nAnswer:", result["result"])
        print("\nSources:")
        for doc in result["source_documents"]:
            print(f"\n- {doc.metadata.get('title')}: {doc.metadata.get('source', 'No URL available')}")
        
        print("\nProcessed documents:")
        for title in retriever.processed_docs:
            print(f"- {title}")
        
        print("-" * 80)

if __name__ == "__main__":
    main()
