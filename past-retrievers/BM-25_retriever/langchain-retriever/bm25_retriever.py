from typing import List, Dict, Any, Optional
from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document
from langchain_core.callbacks.manager import CallbackManagerForRetrieverRun
from langchain.retrievers import WikipediaRetriever
from langchain_community.utilities import WikipediaAPIWrapper
from rank_bm25 import BM25Okapi
import nltk
from langchain.chains import LLMChain
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('langchain_bm25_retriever')

class BM25WikiRetriever(BaseRetriever):
    """A LangChain retriever that combines Wikipedia search with BM25 ranking."""
    
    def __init__(
        self,
        n_docs: int = 5,
        lang: str = "en",
        load_max_docs: int = 5,
        doc_content_chars_max: Optional[int] = 2000,
        gemini_api_key: Optional[str] = None,
    ):
        """Initialize the retriever.
        
        Args:
            n_docs: Number of documents to retrieve per query
            lang: Wikipedia language
            load_max_docs: Maximum number of documents to load from Wikipedia
            doc_content_chars_max: Maximum characters per document
            gemini_api_key: API key for Google's Gemini model
        """
        super().__init__()
        
        # Initialize components
        self.wiki_retriever = WikipediaRetriever(
            lang=lang,
            load_max_docs=load_max_docs,
            doc_content_chars_max=doc_content_chars_max,
            wiki_client=WikipediaAPIWrapper(lang=lang)
        )
        
        self.n_docs = n_docs
        self.processed_docs: Dict[str, Document] = {}
        
        # Initialize NLTK
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt', quiet=True)
        self.word_tokenize = nltk.tokenize.word_tokenize
        
        # Set up query generation chain
        llm = ChatGoogleGenerativeAI(
            model="gemini-pro",
            google_api_key=gemini_api_key,
            temperature=0.7
        )
        
        query_prompt = PromptTemplate(
            input_variables=["context", "question"],
            template="""Given this context: {context}
            And this question: {question}
            Generate a concise search query (max 5 words) to find relevant Wikipedia articles."""
        )
        
        self.query_chain = LLMChain(
            llm=llm,
            prompt=query_prompt,
            memory=ConversationBufferMemory(),
            verbose=True
        )
    
    def _tokenize(self, text: str) -> List[str]:
        """Tokenize text for BM25."""
        return self.word_tokenize(text.lower())
    
    def _rank_documents(self, docs: List[Document], query: str) -> List[Document]:
        """Rank documents using BM25."""
        if not docs:
            return []
        
        # Tokenize documents and query
        tokenized_docs = [self._tokenize(doc.page_content) for doc in docs]
        tokenized_query = self._tokenize(query)
        
        # Create and use BM25
        bm25 = BM25Okapi(tokenized_docs)
        doc_scores = bm25.get_scores(tokenized_query)
        
        # Sort documents by score
        scored_docs = list(zip(docs, doc_scores))
        scored_docs.sort(key=lambda x: x[1], reverse=True)
        
        return [doc for doc, _ in scored_docs[:self.n_docs]]
    
    def _get_next_query(self, current_context: str, question: str) -> str:
        """Generate next search query based on current context."""
        try:
            result = self.query_chain.predict(context=current_context, question=question)
            return result.strip()
        except Exception as e:
            logger.error(f"Error generating query: {e}")
            return question  # Fallback to original question
    
    async def _aget_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        """Asynchronously get relevant documents (required by BaseRetriever)."""
        return await super()._aget_relevant_documents(query, run_manager=run_manager)
    
    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        """Get relevant documents using Wikipedia search and BM25 ranking."""
        try:
            # Get documents from Wikipedia
            docs = self.wiki_retriever.get_relevant_documents(query)
            
            # Update processed docs
            for doc in docs:
                if doc.metadata.get("title") not in self.processed_docs:
                    self.processed_docs[doc.metadata.get("title")] = doc
            
            # Rank documents using BM25
            ranked_docs = self._rank_documents(docs, query)
            
            # Log results
            logger.info(f"Retrieved {len(docs)} documents, ranked top {len(ranked_docs)}")
            for doc in ranked_docs:
                logger.info(f"Title: {doc.metadata.get('title')}")
            
            return ranked_docs
            
        except Exception as e:
            logger.error(f"Error retrieving documents: {e}")
            return []
