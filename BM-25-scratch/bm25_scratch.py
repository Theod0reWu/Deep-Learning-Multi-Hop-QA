import wikipediaapi
import google.generativeai as genai
from rank_bm25 import BM25Okapi
from typing import List, Set, Tuple, Optional
import os
import logging
import requests
from dotenv import load_dotenv
import time
import nltk
from nltk.tokenize import word_tokenize

# Set up logging
logging.basicConfig(level=logging.INFO)

class BM25MultiHopRetriever:
    def __init__(self, gemini_api_key=None):
        """Initialize the BM25 Multi-hop Retriever."""
        self.logger = logging.getLogger('bm25_retriever')
        self.logger.setLevel(logging.INFO)
        
        # Document cache
        self.doc_cache = {}  # title -> (title, full_text) mapping
        self.cache_hits = 0  # Track cache hits
        self.cache_misses = 0  # Track cache misses
        
        # Context tracking
        self.context_history = []  # Track context history
        self.visited_pages = set()  # Track visited pages
        self.context_docs = []  # Track documents in context
        self.context_titles = set()  # Track titles in context
        self.processed_docs = []  # Track all processed documents
        
        # Token tracking for Gemini API
        self.gemini_tokens_in = 0
        self.gemini_tokens_out = 0
        
        # Initialize NLTK
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt', quiet=True)
        self.word_tokenize = word_tokenize
        
        # Configure Wikipedia API
        self.wiki = wikipediaapi.Wikipedia(
            user_agent='BM25MultiHopRetriever/1.0 (https://github.com/YourRepo/)',
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

    def _get_wiki_page(self, title: str) -> Optional[Tuple[str, str]]:
        """Get Wikipedia page content, using cache if available.
        
        Args:
            title: The title of the Wikipedia page
            
        Returns:
            Optional[Tuple[str, str]]: A tuple of (title, content) if found, None if page doesn't exist
        """
        if title in self.doc_cache:
            self.cache_hits += 1
            self.logger.info(f"Cache hit for '{title}' (Total hits: {self.cache_hits})")
            return self.doc_cache[title]
        
        self.cache_misses += 1
        self.logger.info(f"Cache miss for '{title}' (Total misses: {self.cache_misses})")
        
        page = self.wiki.page(title)
        if not page.exists():
            return None
        
        # Cache the page content (truncated to 5000 chars to save memory)
        content = (title, page.text[:5000])
        self.doc_cache[title] = content
        return content

    def _search_wikipedia(self, query: str):
        """Search Wikipedia and return page objects."""
        try:
            # Direct page lookup
            direct_page = self.wiki.page(query)
            if direct_page.exists():
                self.logger.info(f"Found direct match for query: {query}")
                return [direct_page]

            # Search API
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
                return pages
            return []
        except Exception as e:
            self.logger.error(f"Error searching Wikipedia: {str(e)}")
            return []

    def _tokenize(self, text: str) -> List[str]:
        """Tokenize text for BM25."""
        return self.word_tokenize(text.lower())

    def _rank_documents(self, query: str, documents: List[Tuple[str, str]]) -> List[Tuple[str, str, float]]:
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
            
            return ranked_docs
        except Exception as e:
            self.logger.error(f"Error ranking documents: {str(e)}")
            return []

    def _generate_search_query(self, context: str, question: str, iteration: int, previous_queries: List[str]) -> str:
        """Generate a search query using Gemini API with rate limiting."""
        try:
            # Rate limiting
            current_time = time.time()
            time_since_last = current_time - self.last_api_call
            if time_since_last < self.min_delay:
                time.sleep(self.min_delay - time_since_last)

            # Configure safety settings
            safety_settings = {
                "HARM_CATEGORY_HARASSMENT": "BLOCK_NONE",
                "HARM_CATEGORY_HATE_SPEECH": "BLOCK_NONE",
                "HARM_CATEGORY_SEXUALLY_EXPLICIT": "BLOCK_NONE",
                "HARM_CATEGORY_DANGEROUS_CONTENT": "BLOCK_NONE"
            }
            
            # For first iteration (iteration 0), generate initial subquestion
            if iteration == 0:
                prompt = f"""Given this multi-step question: "{question}"

What would be the most relevant Wikipedia article title we need to look up first?

Guidelines:
1. Consider the main entities or subjects in the question
2. Format your response as you would expect a Wikipedia article title
3. Choose the most specific and relevant article that would contain the key information
4. Use standard Wikipedia title capitalization

Examples:
Question: "What is the name of the river in the city where Ikea's headquarters are?"
Response: IKEA

Question: "Which US president was in office when the Panama Canal was completed?"
Response: Panama Canal

Question: "What was the GDP of the country where the 1992 Winter Olympics were held?"
Response: 1992 Winter Olympics

Your response should be just the article title, nothing else.

Response:"""
                
                # Track input tokens
                self.gemini_tokens_in += len(self.word_tokenize(prompt))
                
                response = self.model.generate_content(prompt, safety_settings=safety_settings)
                response_text = response.text.strip()
                
                # Track output tokens
                self.gemini_tokens_out += len(self.word_tokenize(response_text))
                
                return response_text
            
            # For subsequent iterations, generate new queries based on context
            prompt = f"""Given this context and question, determine the next Wikipedia article we need to look up.

Original Question: "{question}"

Context So Far:
{context}

Previous Articles Queried: {', '.join(previous_queries)}

Guidelines:
1. Based on what we've learned from the context, what's the next key piece of information we need?
2. Format your response as a Wikipedia article title that would contain this information
3. Make sure this is DIFFERENT from previous queries: {', '.join(previous_queries)}
4. Use standard Wikipedia title capitalization

Examples of good article titles:
- For population data: "Portland, Oregon" or "Seattle, Washington"
- For company info: "Microsoft Corporation" or "Apple Inc."
- For historical events: "American Civil War" or "World War II"

Your response should be just the article title that would contain our next needed piece of information.

Response:"""
            
            # Track input tokens
            self.gemini_tokens_in += len(self.word_tokenize(prompt))
            
            response = self.model.generate_content(prompt, safety_settings=safety_settings)
            
            self.last_api_call = time.time()
            generated_query = response.text.strip()
            
            # Track output tokens
            self.gemini_tokens_out += len(self.word_tokenize(generated_query))
            
            # Ensure query is not too similar to previous queries
            if generated_query in previous_queries:
                prompt += "\n\nIMPORTANT: We've already looked up that article. We need a DIFFERENT Wikipedia article that would have the next piece of information we need. Previous articles: " + ', '.join(previous_queries)
                
                # Track additional input tokens
                self.gemini_tokens_in += len(self.word_tokenize(prompt))
                
                response = self.model.generate_content(prompt, safety_settings=safety_settings)
                generated_query = response.text.strip()
                
                # Track additional output tokens
                self.gemini_tokens_out += len(self.word_tokenize(generated_query))
            
            return generated_query
            
        except Exception as e:
            self.logger.error(f"Error generating search query: {str(e)}")
            # Extract key terms from question for fallback
            words = question.split()
            if iteration == 0:
                return ' '.join(words[:5])
            else:
                start_idx = (iteration * 3) % max(5, len(words))
                return ' '.join(words[start_idx:start_idx + 5])

    def _generate_answer(self, question: str, context_history: List[str]) -> str:
        """Generate final answer using collected context."""
        try:
            # Rate limiting
            current_time = time.time()
            time_since_last = current_time - self.last_api_call
            if time_since_last < self.min_delay:
                time.sleep(self.min_delay - time_since_last)
        
            # Combine context
            context = "\n\n".join(context_history)
            prompt = f"""Based on the following context, answer the question. Include only information that is supported by the context.

Question: {question}

Context:
{context}

Answer:"""
        
            # Track input tokens for final answer generation
            self.gemini_tokens_in += len(self.word_tokenize(prompt))
            
            try:
                response = self.model.generate_content(prompt,
                    safety_settings={
                        "HARM_CATEGORY_SEXUALLY_EXPLICIT": "BLOCK_NONE",
                        "HARM_CATEGORY_HATE_SPEECH": "BLOCK_ONLY_HIGH",
                        "HARM_CATEGORY_HARASSMENT": "BLOCK_ONLY_HIGH",
                        "HARM_CATEGORY_DANGEROUS_CONTENT": "BLOCK_ONLY_HIGH"
                    })
                self.last_api_call = time.time()
            
                answer = ""
                if hasattr(response, 'text'):
                    answer = response.text.strip()
                elif hasattr(response, 'parts') and response.parts:
                    answer = response.parts[0].text.strip()
                else:
                    answer = "Unable to generate response due to content safety filters. Please try rephrasing the question."
                
                # Track output tokens for final answer
                self.gemini_tokens_out += len(self.word_tokenize(answer))
                
                return answer
                
            except Exception as e:
                self.logger.error(f"Error in Gemini API call: {str(e)}")
                return "Error generating response. Please try rephrasing the question."
            
        except Exception as e:
            self.logger.error(f"Error generating answer: {str(e)}")
            return "Sorry, I was unable to generate an answer due to an error."

    def retrieve(self, 
                question: str, 
                num_iterations: int = 3,
                queries_per_iteration: int = 1,
                docs_per_query: int = 1,
                relative_score_threshold: float = 0.8,
                max_tokens: int = 2000):
        """
        Perform multi-hop retrieval.
        
        Args:
            question: The question to answer
            num_iterations: Target number of documents to collect (will stop early if reached)
            queries_per_iteration: Number of queries per iteration (default: 1)
            docs_per_query: Maximum documents to consider per query (default: 1)
            relative_score_threshold: Take documents scoring above (top_score * threshold) (default: 0.8)
            max_tokens: Maximum number of tokens to keep from each retrieved article (default: 2000)
        """
        self.context_history = [question]
        self.visited_pages = set()  # Track all visited pages
        self.context_docs = []  # Documents actually used in context
        self.context_titles = set()  # Track titles of documents in context
        self.processed_docs = []  # All documents processed, including those not used
        generated_queries = []  # Track all queries generated
        query_texts = []  # Track actual query texts without iteration numbers

        try:
            iteration = 0
            while len(self.context_docs) < num_iterations and iteration < num_iterations * 2:  # Allow up to 2x iterations to find enough docs
                self.logger.info(f"Starting iteration {iteration + 1}, documents collected: {len(self.context_docs)}/{num_iterations}")
                current_context = "\n\n".join(self.context_history)
                
                # Generate search query
                search_query = self._generate_search_query(current_context, question, iteration, query_texts)
                self.logger.info(f"Generated search query: {search_query}")
                
                # Track queries
                query_texts.append(search_query)
                generated_queries.append(f"Iteration {iteration + 1}: {search_query}")
                
                # Search Wikipedia
                search_results = self._search_wikipedia(search_query)
                if not search_results:
                    iteration += 1
                    continue
                
                # Process search results and update cache
                candidates = []
                for page in search_results:
                    wiki_content = self._get_wiki_page(page.title)
                    if wiki_content:
                        candidates.append(wiki_content)
                        # Add to processed_docs if not already there
                        if wiki_content not in self.processed_docs:
                            self.processed_docs.append(wiki_content)
                
                # Filter out visited pages and pages already in context
                unvisited_candidates = [doc for doc in candidates 
                                      if doc[0] not in self.visited_pages 
                                      and doc[0] not in self.context_titles]
                if not unvisited_candidates:
                    iteration += 1
                    continue
                
                # Rank unvisited documents
                ranked_docs = self._rank_documents(search_query, unvisited_candidates)
                if not ranked_docs:
                    iteration += 1
                    continue
                
                # Get the top score
                top_score = ranked_docs[0][2]
                score_threshold = top_score * relative_score_threshold
                
                # Take documents above the threshold
                docs_to_add = min(docs_per_query, num_iterations - len(self.context_docs))
                docs_added = 0
                
                for title, content, score in ranked_docs:
                    # Triple check: not visited, not in context, and meets score threshold
                    if (title not in self.visited_pages 
                        and title not in self.context_titles 
                        and score >= score_threshold):
                        
                        context_content = content[:max_tokens]  # Truncate to max_tokens
                        self.context_history.append(f"From {title}:\n{context_content}")
                        self.context_docs.append((title, context_content))
                        self.context_titles.add(title)  # Track title in context
                        self.visited_pages.add(title)  # Mark as visited
                        docs_added += 1
                        
                        self.logger.info(f"Added document: {title} (score: {score:.3f})")
                        
                        # Break if we've added enough docs
                        if docs_added >= docs_to_add or len(self.context_docs) >= num_iterations:
                            break
                
                iteration += 1
            
            self.logger.info(f"Retrieval complete. Collected {len(self.context_docs)} documents in {iteration} iterations")
            self.logger.info(f"Cache statistics - Hits: {self.cache_hits}, Misses: {self.cache_misses}, Hit rate: {self.cache_hits/(self.cache_hits + self.cache_misses):.2%}")
            self.logger.info(f"Gemini API usage - Input tokens: {self.gemini_tokens_in}, Output tokens: {self.gemini_tokens_out}")
            
            # Generate final answer
            answer = self._generate_answer(question, self.context_history)
            
            return answer, self.context_history, self.visited_pages, self.context_docs, self.processed_docs, generated_queries
            
        except Exception as e:
            self.logger.error(f"Error in retrieval process: {str(e)}")
            return "Error during retrieval", [], set(), [], [], []
