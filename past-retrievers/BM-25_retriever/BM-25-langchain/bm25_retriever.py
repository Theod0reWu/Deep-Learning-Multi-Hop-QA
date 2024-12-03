from typing import Dict, List, Tuple, Any, Optional, Annotated
import os
import time
import logging
import nltk
import networkx as nx
import requests
import wikipediaapi
import google.generativeai as genai
from typing import List, Dict, Tuple, Optional
from rank_bm25 import BM25Okapi
from dataclasses import dataclass, field
import random
from langchain import hub
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from operator import itemgetter
import operator
from typing import TypedDict, Sequence, Union
import matplotlib.pyplot as plt
from langchain_core.messages import HumanMessage

# Set up logging
logging.basicConfig(level=logging.INFO)

class State:
    """State object to track retrieval progress."""
    def __init__(self, question: str, target_docs: int):
        self.question = question
        self.target_docs = target_docs
        self.current_iteration = 1  # Start at 1
        self.current_query = question
        self.context_docs = []  # List of (hop_num, title, content) tuples for documents in current context
        self.processed_docs = []  # List of (title, content) tuples for all documents seen
        self.visited_pages = set()  # Set of page titles we've seen
        self.queries_made = []  # List of (iteration, query) tuples to track query history
        self.hops = []  # List of (hop_num, title, content) tuples in order of retrieval
        self.answer = ""

class BM25MultiHopRetriever:
    def __init__(self, gemini_api_key: Optional[str] = None, n_iterations: int = 5, n_queries: int = 5, n_docs: int = 2, doc_content_chars_max: int = 2000):
        """Initialize the BM25 Multi-hop Retriever with LangChain integration.
        
        Args:
            gemini_api_key: Optional API key for Google's Gemini model
            n_iterations: Maximum number of retrieval iterations
            n_queries: Number of queries to generate per iteration
            n_docs: Number of documents to retrieve per query
            doc_content_chars_max: Maximum number of characters to keep from each document
        """
        # Initialize logger
        self.logger = logging.getLogger('bm25_retriever_langchain')
        
        self.n_iterations = n_iterations
        self.n_queries = n_queries
        self.n_docs = n_docs
        self.doc_content_chars_max = doc_content_chars_max
        self.page_cache = {}
        
        # Initialize Wikipedia API
        self.wiki_api = wikipediaapi.Wikipedia(
            language='en',
            user_agent='BM25MultiHopRetriever/1.0 (https://github.com/Theod0reWu/Deep-Learning-Multi-Hop-QA) (Wikipedia-API/0.6.0; https://github.com/martin-majlis/Wikipedia-API/)',
            extract_format=wikipediaapi.ExtractFormat.WIKI
        )
        
        # Initialize visualization graph
        self.graph = nx.DiGraph()
        
        # Configure Gemini
        self.logger.info("Configuring Gemini API...")
        if gemini_api_key:
            genai.configure(api_key=gemini_api_key)
        
        generation_config = genai.types.GenerationConfig(
            temperature=0.7,
            top_p=1,
            top_k=1,
            max_output_tokens=50
        )
        
        safety_settings = [
            {
                "category": "HARM_CATEGORY_HARASSMENT",
                "threshold": "BLOCK_MEDIUM_AND_ABOVE"
            },
            {
                "category": "HARM_CATEGORY_HATE_SPEECH",
                "threshold": "BLOCK_MEDIUM_AND_ABOVE"
            },
            {
                "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                "threshold": "BLOCK_MEDIUM_AND_ABOVE"
            },
            {
                "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                "threshold": "BLOCK_MEDIUM_AND_ABOVE"
            },
        ]
        
        self.model = genai.GenerativeModel(
            model_name="gemini-pro",
            generation_config=generation_config,
            safety_settings=safety_settings
        )
        self.logger.info("Gemini API configured successfully")
        
        # Initialize NLTK
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt', quiet=True)
        self.word_tokenize = nltk.tokenize.word_tokenize

        # Rate limiting
        self.last_api_call = 0
        self.min_delay = 1.0  # Minimum delay between API calls in seconds

    def _create_graph(self) -> nx.DiGraph:
        """Create the workflow graph for visualization."""
        graph = nx.DiGraph()
        
        # Add nodes for each step with attributes
        graph.add_node("start", type="entry", label="Start")
        graph.add_node("search", type="operation", label="Search Wikipedia")
        graph.add_node("rank", type="operation", label="Rank Documents")
        graph.add_node("generate_query", type="operation", label="Generate Next Query")
        graph.add_node("synthesize", type="operation", label="Synthesize Answer")
        graph.add_node("end", type="terminal", label="End")
        
        # Add edges with labels
        edges = [
            ("start", "search", "initial_query"),
            ("search", "rank", "documents"),
            ("rank", "generate_query", "ranked_docs"),
            ("generate_query", "search", "next_query"),
            ("generate_query", "synthesize", "final_iteration"),
            ("synthesize", "end", "answer")
        ]
        
        for src, dst, label in edges:
            graph.add_edge(src, dst, label=label)
        
        return graph

    def _get_wiki_page(self, title: str) -> Tuple[Optional[wikipediaapi.WikipediaPage], bool]:
        """Get Wikipedia page, using cache if available."""
        if title in self.page_cache:
            self.logger.info(f"Using cached page: {title}")
            return self.page_cache[title], False
        
        retries = 5
        delay = 1.0
        for attempt in range(retries):
            page = self.wiki_api.page(title)
            if page.exists():
                # Truncate content to doc_content_chars_max
                content = page.text[:self.doc_content_chars_max]
                self.page_cache[title] = (page, content)
                self.logger.info(f"Fetched new page: {title}")
                return page, True
            else:
                self.logger.warning(f"Page not found: {title}. Retrying in {delay} seconds...")
                time.sleep(delay)
                delay *= 2  # Exponential backoff
        
        self.logger.error(f"Failed to retrieve page: {title} after {retries} attempts.")
        return None, True

    def _tokenize(self, text: str) -> List[str]:
        """Tokenize text for BM25."""
        return self.word_tokenize(text.lower())

    def _rank_documents(self, state: State, current_hop: int) -> State:
        """Rank documents using BM25."""
        node_name = f"rank_{state.current_iteration}"
        ranked_name = f"ranked_{state.current_iteration}"
        
        # Add dynamic nodes and edges to graph
        self.graph.add_node(node_name, type="operation", label=f"Rank {state.current_iteration}")
        self.graph.add_node(ranked_name, type="data", label=f"Ranked {state.current_iteration}")
        self.graph.add_edge(node_name, ranked_name, label=f"rank_{state.current_iteration}")
        
        if not state.context_docs:
            return state
            
        try:
            # Extract content for ranking
            docs_for_ranking = []
            titles = []
            for _, title, content in state.context_docs:
                if title not in state.visited_pages:
                    docs_for_ranking.append(content)
                    titles.append(title)
            
            if not docs_for_ranking:
                return state
            
            # Tokenize documents
            tokenized_docs = [self._tokenize(doc) for doc in docs_for_ranking]
            
            # Create BM25 instance
            bm25 = BM25Okapi(tokenized_docs)
            
            # Get scores
            tokenized_query = self._tokenize(state.current_query)
            doc_scores = bm25.get_scores(tokenized_query)
            
            # Sort documents by score
            ranked_pairs = list(zip(titles, docs_for_ranking, doc_scores))
            ranked_pairs.sort(key=lambda x: x[2], reverse=True)
            
            # Add only the top document to context
            if ranked_pairs:
                title, content, score = ranked_pairs[0]
                if title not in state.visited_pages:
                    state.visited_pages.add(title)
                    state.context_docs.append((current_hop, title, content[:self.doc_content_chars_max]))
                    state.processed_docs.append((title, content[:self.doc_content_chars_max]))
                    self.logger.info(f"Added document to context: {title}")
                else:
                    self.logger.info(f"Top document {title} already visited")
            else:
                self.logger.info("No documents to rank in this iteration")
            
            return state
            
        except Exception as e:
            self.logger.error(f"Error ranking documents: {str(e)}")
            return state

    def _call_gemini_with_backoff(self, prompt: str, max_retries: int = 5, initial_delay: float = 1.0) -> str:
        """Call Gemini API with exponential backoff."""
        delay = initial_delay
        attempt = 0
        
        while attempt < max_retries:
            try:
                # Ensure minimum delay between calls
                current_time = time.time()
                time_since_last_call = current_time - self.last_api_call
                if time_since_last_call < self.min_delay:
                    time.sleep(self.min_delay - time_since_last_call)
                
                response = self.model.generate_content(prompt)
                self.last_api_call = time.time()
                
                if response.text:
                    return response.text.strip()
                    
            except Exception as e:
                attempt += 1
                if "rate limit exceeded" in str(e).lower():
                    wait_time = delay * (2 ** attempt)  # Exponential backoff
                    self.logger.warning(f"Rate limit hit, waiting {wait_time:.1f} seconds before retry {attempt}/{max_retries}")
                    time.sleep(wait_time)
                    continue
                elif attempt == max_retries:
                    self.logger.error(f"Failed to call Gemini API after {max_retries} attempts: {str(e)}")
                    raise
                else:
                    self.logger.warning(f"Gemini API error (attempt {attempt}/{max_retries}): {str(e)}")
                    time.sleep(delay)
        
        # Generate a simple query if all retries failed
        return self._generate_fallback_query(prompt)

    def _generate_fallback_query(self, prompt: str) -> str:
        """Generate a simple fallback query when API fails."""
        # Extract key terms from the original question
        words = self._tokenize(prompt.lower())
        # Remove common words and take 3-4 most relevant terms
        stopwords = {'what', 'how', 'why', 'when', 'where', 'who', 'is', 'are', 'was', 'were', 'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
        keywords = [word for word in words if word not in stopwords]
        return ' '.join(keywords[:4])

    def _search_wikipedia(self, state: State) -> State:
        """Search Wikipedia and return page objects."""
        node_name = f"search_{state.current_iteration}"
        docs_name = f"docs_{state.current_iteration}"
        
        # Add dynamic nodes and edges to graph
        self.graph.add_node(node_name, type="operation", label=f"Search {state.current_iteration}")
        self.graph.add_node(docs_name, type="data", label=f"Documents {state.current_iteration}")
        self.graph.add_edge(node_name, docs_name, label=state.current_query)
        
        try:
            # First try direct page lookup
            page = self.wiki_api.page(state.current_query)
            if page.exists():
                self.logger.info(f"Found direct match for query: {state.current_query}")
                pages = [page]
            else:
                # Then try search API
                search_url = 'https://en.wikipedia.org/w/api.php'
                search_params = {
                    'action': 'query',
                    'list': 'search',
                    'srsearch': state.current_query,
                    'format': 'json',
                    'srlimit': self.n_docs
                }
                headers = {'User-Agent': 'BM25MultiHopRetriever/1.0 (https://github.com/Theod0reWu/Deep-Learning-Multi-Hop-QA)'}
                
                response = requests.get(search_url, params=search_params, headers=headers)
                response.raise_for_status()
                
                search_results = response.json()
                pages = []
                if 'query' in search_results and 'search' in search_results['query']:
                    for result in search_results['query']['search']:
                        page = self.wiki_api.page(result['title'])
                        if page.exists():
                            pages.append(page)
                            self.logger.info(f"Found page: {page.title}")
            
            # Update state with documents
            state.context_docs = []
            for page in pages:
                if page.title not in state.visited_pages:
                    content = page.text
                    if content:
                        state.context_docs.append((state.current_iteration, page.title, content[:self.doc_content_chars_max]))
                        # Add linked pages (but don't add them to context_docs yet)
                        for link in list(page.links.keys())[:3]:  # Limit to first 3 links
                            if link not in state.visited_pages:
                                link_page = self.wiki_api.page(link)
                                if link_page.exists():
                                    state.context_docs.append((state.current_iteration, link, link_page.text[:self.doc_content_chars_max]))
            
            return state
            
        except Exception as e:
            self.logger.error(f"Error searching Wikipedia: {str(e)}")
            return state

    def _generate_next_query(self, state: State) -> State:
        """Generate a search query using Gemini API with rate limiting."""
        node_name = f"query_{state.current_iteration}"
        
        # Add dynamic node and edge to graph
        self.graph.add_node(node_name, type="operation", label=f"Generate Query {state.current_iteration}")
        self.graph.add_edge(node_name, f"search_{state.current_iteration+1}", label=f"generate_{state.current_iteration}")
        
        try:
            # Use a random selection of context pieces for query generation
            context_pieces = state.hops[-5:]  # Consider the last 5 entries
            random.shuffle(context_pieces)
            context = "\n\n".join([f"{hop[1]}: {hop[2]}" for hop in context_pieces[:3]])  # Randomly pick 3
            
            prompt = (f"Given this context: {context}\n"
                      f"And this question: {state.question}\n"
                      "Generate a concise search query (max 5 words) to find relevant Wikipedia articles. "
                      "Try to focus on different aspects or angles of the topic.")
            
            query = self._call_gemini_with_backoff(prompt)
            state.current_query = query
            state.queries_made.append((state.current_iteration, query))
            
            return state
            
        except Exception as e:
            self.logger.error(f"Error generating query: {str(e)}")
            # Fallback to using question words if query generation fails
            query_words = state.question.split()[:5]
            state.current_query = ' '.join(query_words)
            state.queries_made.append((state.current_iteration, state.current_query))
            return state

    def _generate_answer(self, state: State) -> State:
        """Generate final answer using collected context."""
        try:
            if len(state.hops) > 1:
                context = "\n\n".join([f"{hop[1]}: {hop[2]}" for hop in state.hops])
                
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
            self.logger.error(f"Error generating answer: {str(e)}")
            state.answer = "Sorry, I was unable to generate an answer due to an error."
            return state

    def _save_visualization(self):
        """Save visualization with improved layout."""
        plt.figure(figsize=(12, 8))
        pos = nx.spring_layout(self.graph, k=2, iterations=50)
        
        # Draw nodes with different colors based on type
        node_colors = []
        for node in self.graph.nodes():
            if self.graph.nodes[node].get('type') == 'entry':
                node_colors.append('lightgreen')
            elif self.graph.nodes[node].get('type') == 'operation':
                node_colors.append('lightblue')
            elif self.graph.nodes[node].get('type') == 'data':
                node_colors.append('lightyellow')
            elif self.graph.nodes[node].get('type') == 'terminal':
                node_colors.append('lightcoral')
            else:
                node_colors.append('gray')
        
        # Draw the graph
        nx.draw(self.graph, pos, 
                node_color=node_colors,
                node_size=2000, 
                font_size=8,
                font_weight='bold',
                with_labels=True,
                labels={node: self.graph.nodes[node].get('label', node) for node in self.graph.nodes()},
                edge_color='gray',
                arrows=True,
                arrowsize=20)
        
        # Add edge labels
        edge_labels = nx.get_edge_attributes(self.graph, 'label')
        nx.draw_networkx_edge_labels(self.graph, pos, edge_labels=edge_labels, font_size=6)
        
        plt.title("Multi-Hop Retrieval Workflow")
        plt.axis('off')
        plt.tight_layout()
        plt.savefig("retrieval_workflow.png", dpi=300, bbox_inches='tight')
        plt.close()

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
        
        # Set initial query to the question
        state.current_query = question
        
        while len(state.context_docs) < target_docs and state.current_iteration <= self.n_iterations:
            # Generate search query based on current context
            state = self._generate_next_query(state)
            self.logger.info(f"\nIteration {state.current_iteration}")
            self.logger.info(f"Query: {state.current_query}")
            
            # Skip if we've seen this query before
            if len(state.queries_made) > 1 and state.current_query in [q[1] for q in state.queries_made[:-1]]:
                continue
            
            # Search Wikipedia and get documents
            state = self._search_wikipedia(state)
            
            # Track the current iteration for documents found in this round
            current_hop = state.current_iteration
            
            # Rank documents and add top one to context
            state = self._rank_documents(state, current_hop)
            
            # Break if no new documents were found
            if len(state.context_docs) == len(state.visited_pages) - 1:
                self.logger.info(f"No new documents found after {state.current_iteration} iterations")
                break
            
            # Update iteration counter for next round
            state.current_iteration += 1
        
        # Generate final answer
        state = self._generate_answer(state)
        
        # Log the retrieval path
        self.logger.info("\nRetrieval Path:")
        for hop_num, title, _ in sorted(state.context_docs):
            url = f"https://en.wikipedia.org/wiki/{title.replace(' ', '_')}"
            self.logger.info(f"Hop {hop_num}: {title} ({url})")
        
        self.logger.info("\nAll Processed Documents:")
        for title, _ in state.processed_docs:
            url = f"https://en.wikipedia.org/wiki/{title.replace(' ', '_')}"
            self.logger.info(f"{title} ({url})")
        
        self.logger.info("\nQuery History:")
        for iteration, query in state.queries_made:
            self.logger.info(f"Iteration {iteration}: {query}")
        
        return state.answer, state
