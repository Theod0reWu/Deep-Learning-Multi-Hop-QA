import wikipediaapi
import google.generativeai as genai
from rank_bm25 import BM25Okapi
from typing import List, Set, Dict, Tuple
import os
import logging
import requests
from dotenv import load_dotenv
import time

# Set up logging
logging.basicConfig(level=logging.INFO)

class BM25MultiHopRetriever:
    def __init__(self, gemini_api_key=None):
        """Initialize the BM25 Multi-hop Retriever."""
        self.logger = logging.getLogger('bm25_retriever')
        self.logger.setLevel(logging.INFO)
        
        # Initialize NLTK and download required data
        try:
            import nltk
            try:
                nltk.data.find('tokenizers/punkt')
            except LookupError:
                nltk.download('punkt', quiet=True)
            self.word_tokenize = nltk.tokenize.word_tokenize
        except ImportError:
            self.logger.warning("NLTK not found. Installing required packages...")
            import subprocess
            import sys
            subprocess.check_call([sys.executable, "-m", "pip", "install", "nltk"])
            import nltk
            nltk.download('punkt', quiet=True)
            self.word_tokenize = nltk.tokenize.word_tokenize

        # Configure Wikipedia API
        self.wiki = wikipediaapi.Wikipedia(
            user_agent='BM25MultiHopRetriever/1.0 (https://github.com/Theod0reWu/Deep-Learning-Multi-Hop-QA; jasplevy@gmail.com)',
            language='en'
        )
        
        # Configure Gemini API
        self.logger.info("Configuring Gemini API...")
        if gemini_api_key:
            genai.configure(api_key=gemini_api_key)
        else:
            load_dotenv()
            genai.configure(api_key=os.getenv('GEMINI_API_KEY'))
            
        self.model = genai.GenerativeModel('gemini-pro')
        self.logger.info("Gemini API configured successfully")
        
        # Rate limiting
        self.last_api_call = 0
        self.min_delay = 1.0  # Minimum delay between API calls in seconds
    
    def _search_wikipedia(self, query):
        """Search Wikipedia and return page objects."""
        try:
            # First try direct page lookup
            direct_page = self.wiki.page(query)
            if direct_page.exists():
                self.logger.info(f"Found direct match for query: {query}")
                return [direct_page]

            # Then try search API
            search_url = 'https://en.wikipedia.org/w/api.php'
            search_params = {
                'action': 'query',
                'list': 'search',
                'srsearch': query,
                'format': 'json',
                'srlimit': 5
            }
            headers = {'User-Agent': 'BM25MultiHopRetriever/1.0'}
            
            response = requests.get(search_url, params=search_params, headers=headers)
            response.raise_for_status()
            
            search_results = response.json()
            if 'query' in search_results and 'search' in search_results['query']:
                pages = []
                for result in search_results['query']['search']:
                    try:
                        page = self.wiki.page(result['title'])
                        if page.exists():
                            pages.append(page)
                            self.logger.info(f"Found page: {page.title}")
                    except Exception as e:
                        self.logger.warning(f"Error fetching page {result['title']}: {str(e)}")
                        continue
                return pages
            return []
            
        except Exception as e:
            self.logger.error(f"Error searching Wikipedia: {str(e)}")
            return []
    
    def _tokenize(self, text: str) -> List[str]:
        """Tokenize text for BM25."""
        return self.word_tokenize(text.lower())
    
    def _get_wiki_content(self, title: str) -> Tuple[str, List[str]]:
        """Get Wikipedia page content and links."""
        try:
            page = self.wiki.page(title)
            if not page.exists():
                self.logger.warning(f"Wikipedia page not found: {title}")
                return "", []
            return page.text, [link for link in page.links.keys()]
        except Exception as e:
            self.logger.error(f"Error fetching Wikipedia content for {title}: {str(e)}")
            return "", []
    
    def _rank_documents(self, query: str, documents: List[Tuple[str, str]]) -> List[Tuple[str, str]]:
        """Rank documents using BM25."""
        if not documents:
            return []
            
        try:
            # Tokenize documents
            tokenized_docs = [self._tokenize(doc[1]) for doc in documents]
            
            # Create BM25 instance
            bm25 = BM25Okapi(tokenized_docs)
            
            # Get scores
            tokenized_query = self._tokenize(query)
            doc_scores = bm25.get_scores(tokenized_query)
            
            # Sort documents by score
            ranked_docs = [(documents[i][0], documents[i][1], score) 
                          for i, score in enumerate(doc_scores)]
            ranked_docs.sort(key=lambda x: x[2], reverse=True)
            
            return [(doc[0], doc[1]) for doc in ranked_docs]
        except Exception as e:
            self.logger.error(f"Error ranking documents: {str(e)}")
            return []
    
    def _generate_search_query(self, context, question):
        """Generate a search query using Gemini API with rate limiting."""
        try:
            # Rate limiting
            current_time = time.time()
            time_since_last = current_time - self.last_api_call
            if time_since_last < self.min_delay:
                time.sleep(self.min_delay - time_since_last)
            
            prompt = f"Given this context: {context}\nAnd this question: {question}\nGenerate a concise search query (max 5 words) to find relevant Wikipedia articles."
            response = self.model.generate_content(prompt)
            
            self.last_api_call = time.time()
            return response.text.strip()
            
        except Exception as e:
            self.logger.error(f"Error generating search query: {str(e)}")
            # Fallback to basic query from question
            return ' '.join(question.split()[:5])

    def _generate_answer(self, question: str, context_history: List[str]) -> str:
        """Generate final answer using collected context."""
        try:
            # Rate limiting
            current_time = time.time()
            time_since_last = current_time - self.last_api_call
            if time_since_last < self.min_delay:
                time.sleep(self.min_delay - time_since_last)
            
            # Combine context with newlines between documents
            context = "\n\n".join(context_history)
            
            prompt = f"""Based on the following context, answer the question. Include only information that is supported by the context.

Question: {question}

Context:
{context}

Answer:"""
            
            response = self.model.generate_content(prompt)
            self.last_api_call = time.time()
            return response.text.strip()
            
        except Exception as e:
            self.logger.error(f"Error generating answer: {str(e)}")
            return "Sorry, I was unable to generate an answer due to an error."
    
    def retrieve(self, 
                question: str, 
                num_iterations: int = 3, 
                queries_per_iteration: int = 2, 
                docs_per_query: int = 2) -> Tuple[str, List[str], Set[str]]:
        """
        Perform multi-hop retrieval.
        
        Returns:
            Tuple containing:
            - Final answer
            - List of context documents
            - Set of visited page titles
        """
        self.context_history = [question]
        self.visited_pages = set()
        
        try:
            for iteration in range(num_iterations):
                self.logger.info(f"Starting iteration {iteration + 1}/{num_iterations}")
                current_context = " ".join(self.context_history[-3:])  # Use last 3 context pieces
                
                for query_num in range(queries_per_iteration):
                    self.logger.info(f"Processing query {query_num + 1}/{queries_per_iteration}")
                    
                    # Generate and execute search
                    search_query = self._generate_search_query(current_context, question)
                    self.logger.info(f"Generated search query: {search_query}")
                    
                    search_results = self._search_wikipedia(search_query)
                    self.logger.info(f"Found {len(search_results)} search results")
                    
                    if not search_results:
                        continue
                        
                    # Collect and rank documents
                    candidates = []
                    for page in search_results:
                        if page.title not in self.visited_pages:
                            content = page.text
                            if content:
                                candidates.append((page.title, content[:5000]))  # Limit content size
                                # Add linked pages as candidates
                                for link in list(page.links.keys())[:3]:  # Limit to first 3 links
                                    if link not in self.visited_pages:
                                        link_page = self.wiki.page(link)
                                        if link_page.exists():
                                            candidates.append((link, link_page.text[:5000]))
                    
                    if candidates:
                        # Rank and add top documents
                        ranked_docs = self._rank_documents(search_query, candidates)
                        for title, content in ranked_docs[:docs_per_query]:
                            if title not in self.visited_pages:
                                self.logger.info(f"Adding document: {title}")
                                self.visited_pages.add(title)
                                # Add a truncated version to context
                                self.context_history.append(f"From {title}:\n{content[:2000]}")
            
            # Generate final answer
            if len(self.context_history) > 1:  # Only if we found some content
                answer = self._generate_answer(question, self.context_history)
            else:
                answer = "I could not find relevant information to answer this question."
            
            return answer, self.context_history, self.visited_pages
            
        except Exception as e:
            self.logger.error(f"Error in retrieve method: {str(e)}")
            return "An error occurred during retrieval.", self.context_history, self.visited_pages
