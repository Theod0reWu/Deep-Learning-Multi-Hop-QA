from typing import Dict, List, Tuple, Any, Optional
import os
from dotenv import load_dotenv
import google.generativeai as genai
import wikipediaapi
import time
from rank_bm25 import BM25Okapi
import logging
import nltk
from dataclasses import dataclass, field

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('bm25_retriever')

@dataclass
class State:
    """The state of our multi-hop retrieval workflow."""
    question: str = field()
    context_history: List[str] = field(default_factory=list)
    visited_pages: set = field(default_factory=set)
    page_urls: Dict[str, str] = field(default_factory=dict)
    current_iteration: int = 0
    current_query: str = ""
    documents: List[Tuple[str, str]] = field(default_factory=list)
    answer: str = ""
    target_docs: int = 0
    processed_docs: List[str] = field(default_factory=list)  # All docs processed
    context_docs: List[str] = field(default_factory=list)    # Docs used in context
    queries_made: List[str] = field(default_factory=list)    # Queries used

class BM25MultiHopRetriever:
    def __init__(self, gemini_api_key: Optional[str] = None, n_iterations: int = 5, n_queries: int = 5, n_docs: int = 1):
        """Initialize the BM25 Multi-hop Retriever.
        
        Args:
            gemini_api_key: Optional API key for Gemini. If None, will try to load from environment
            n_iterations: Maximum number of retrieval iterations
            n_queries: Number of queries per iteration
            n_docs: Number of documents to retrieve per query (default: 1)
        """
        # Load environment variables
        load_dotenv()
        
        # Set up API keys and configurations
        self.gemini_api_key = gemini_api_key or os.getenv('GEMINI_API_KEY')
        if not self.gemini_api_key:
            raise ValueError("Gemini API key must be provided or set in environment as GEMINI_API_KEY")
        
        genai.configure(api_key=self.gemini_api_key)
        
        # Initialize Gemini model with configuration
        generation_config = genai.types.GenerationConfig(
            temperature=0.7,
            top_p=1,
            top_k=1,
            max_output_tokens=2048,
        )
        
        self.model = genai.GenerativeModel(
            model_name="gemini-pro",
            generation_config=generation_config
        )
        
        # Set up Wikipedia API
        self.wiki = wikipediaapi.Wikipedia(
            language='en',
            extract_format=wikipediaapi.ExtractFormat.WIKI,
            user_agent='BM25MultiHopRetriever/1.0'
        )
        
        # Initialize page cache
        self.page_cache = {}
        
        # Store configuration
        self.n_iterations = n_iterations
        self.n_queries = n_queries
        self.n_docs = n_docs
        
        # Initialize rate limiting
        self.last_api_call = 0
        self.min_delay = 1.0  # Minimum delay between API calls in seconds
        
        # Initialize NLTK
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt', quiet=True)
        self.word_tokenize = nltk.tokenize.word_tokenize

    def _get_wiki_page(self, title: str) -> Tuple[Optional[wikipediaapi.WikipediaPage], bool]:
        """Get Wikipedia page, using cache if available."""
        if title in self.page_cache:
            logger.info(f"Using cached page: {title}")
            return self.page_cache[title], False
        
        page = self.wiki.page(title)
        if page.exists():
            self.page_cache[title] = page
            logger.info(f"Fetched new page: {title}")
            return page, True
        return None, True

    def _tokenize(self, text: str) -> List[str]:
        """Tokenize text for BM25."""
        return self.word_tokenize(text.lower())

    def _call_gemini_with_backoff(self, prompt: str) -> str:
        """Call Gemini API with exponential backoff."""
        current_time = time.time()
        if current_time - self.last_api_call < self.min_delay:
            time.sleep(self.min_delay - (current_time - self.last_api_call))
        
        response = self.model.generate_content(prompt)
        self.last_api_call = time.time()
        return response.text.strip()

    def _generate_next_query(self, state: State) -> State:
        """Generate a search query using Gemini API."""
        prompt = f"""Based on this question and context, generate a VERY concise search query (max 3-4 words) to find additional relevant information.

Question: {state.question}

Previous queries:
{' | '.join(state.queries_made[-2:]) if state.queries_made else 'None'}

Current context summary:
{' | '.join(state.context_docs[-2:]) if state.context_docs else 'None'}

Generate only the search query, no explanation."""
        
        try:
            query = self._call_gemini_with_backoff(prompt)
            query_words = query.split()
            if len(query_words) > 4:
                query = ' '.join(query_words[:4])
            
            state.current_query = query
            state.queries_made.append(query)
            
        except Exception as e:
            logger.error(f"Error generating search query: {str(e)}")
            query_words = state.question.split()[:4]
            state.current_query = ' '.join(query_words)
            state.queries_made.append(state.current_query)
        
        return state

    def _search_wikipedia(self, state: State) -> State:
        """Search Wikipedia and return relevant pages."""
        try:
            # First try direct page lookup
            page, is_new = self._get_wiki_page(state.current_query)
            if page:
                logger.info(f"Found direct match for query: {state.current_query}")
                pages = [page]
            else:
                # Get related pages through links
                search_page, is_new = self._get_wiki_page(state.current_query.replace(" ", "_"))
                if search_page and search_page.exists():
                    pages = [search_page]
                    # Get linked pages
                    links = list(search_page.links.keys())[:5]  # Limit to first 5 links
                    for link in links:
                        if link not in state.visited_pages:
                            link_page, is_new = self._get_wiki_page(link)
                            if link_page and link_page.exists():
                                pages.append(link_page)
                else:
                    pages = []
            
            # Update state with documents
            state.documents = []
            for page in pages:
                if page.title not in state.visited_pages:
                    content = page.text
                    if content:
                        state.documents.append((page.title, content[:2000]))  # Limit content size
                        if page.fullurl:  # Check if URL exists
                            state.page_urls[page.title] = str(page.fullurl)  # Ensure URL is string
                        
            return state
            
        except Exception as e:
            logger.error(f"Error searching Wikipedia: {str(e)}")
            return state

    def _rank_documents(self, state: State) -> State:
        """Rank documents using BM25."""
        if not state.documents:
            return state
            
        try:
            # Add all retrieved documents to processed list
            for title, _ in state.documents:
                if title not in state.processed_docs:
                    state.processed_docs.append(title)
            
            # Tokenize documents
            tokenized_docs = [self._tokenize(doc[1]) for doc in state.documents]
            
            # Create BM25 model
            bm25 = BM25Okapi(tokenized_docs)
            
            # Get document scores
            query_tokens = self._tokenize(state.question)
            doc_scores = bm25.get_scores(query_tokens)
            
            # Sort documents by score
            ranked_docs = [(state.documents[i][0], state.documents[i][1], score) 
                          for i, score in enumerate(doc_scores)]
            ranked_docs.sort(key=lambda x: x[2], reverse=True)
            
            # Add top document to context if new
            docs_added = 0
            for title, content, _ in ranked_docs:
                if (title not in state.visited_pages and 
                    title not in state.context_docs):
                    state.visited_pages.add(title)
                    state.context_docs.append(title)
                    state.context_history.append(
                        f"\nIteration {state.current_iteration}, Query: {state.current_query}\n"
                        f"From {title}:\n{content[:2000]}"
                    )
                    docs_added += 1
                    if docs_added >= self.n_docs:  # Only add specified number of docs per iteration
                        break
            
            return state
            
        except Exception as e:
            logger.error(f"Error ranking documents: {str(e)}")
            return state

    def _generate_answer(self, state: State) -> State:
        """Generate final answer using collected context."""
        try:
            if len(state.context_history) > 1:
                context = "\n\n".join(state.context_history)
                
                prompt = f"""Based on the following context, provide a direct and concise answer to the question. Do not include citations or references.

Question: {state.question}

Context:
{context}

Answer the question directly and concisely."""
                
                state.answer = self._call_gemini_with_backoff(prompt)
            else:
                state.answer = "I could not find relevant information to answer this question."
            
            return state
            
        except Exception as e:
            logger.error(f"Error generating answer: {str(e)}")
            state.answer = "Sorry, I was unable to generate an answer due to an error."
            return state

    def retrieve(self, question: str, target_docs: int) -> Tuple[str, State]:
        """
        Perform multi-hop retrieval to answer the question.
        
        Args:
            question: The question to answer
            target_docs: Target number of documents to collect
            
        Returns:
            Tuple containing the final answer and the final state
        """
        state = State(
            question=question,
            target_docs=target_docs
        )
        
        while state.current_iteration < target_docs:  # Use target_docs as iteration limit
            # Generate search query
            state = self._generate_next_query(state)
            
            # Search and rank documents
            state = self._search_wikipedia(state)
            state = self._rank_documents(state)
            
            # Update iteration counter
            state.current_iteration += 1
        
        # Generate final answer
        state = self._generate_answer(state)
        
        return state.answer, state
