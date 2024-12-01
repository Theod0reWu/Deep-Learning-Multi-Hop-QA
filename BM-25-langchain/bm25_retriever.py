from typing import Dict, List, Tuple, Any, Annotated
import os
from dotenv import load_dotenv
import google.generativeai as genai
import wikipediaapi
import requests
import time
from rank_bm25 import BM25Okapi
import logging
import nltk
from langchain import hub
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from operator import itemgetter
import operator
from typing import TypedDict, Sequence, Union
import networkx as nx
import matplotlib.pyplot as plt
from langchain_core.messages import HumanMessage
from dataclasses import dataclass, field

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('bm25_retriever_langgraph')

@dataclass
class State:
    """The state of our multi-hop retrieval workflow."""
    question: str
    context_history: List[str] = field(default_factory=list)
    visited_pages: set = field(default_factory=set)
    current_iteration: int = 0
    current_query: str = ""
    documents: List[Tuple[str, str]] = field(default_factory=list)
    answer: str = ""

class BM25MultiHopRetriever:
    def __init__(self, gemini_api_key=None):
        """Initialize the BM25 Multi-hop Retriever with LangChain integration."""
        self.logger = logger
        
        # Initialize NLTK
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt', quiet=True)
        self.word_tokenize = nltk.tokenize.word_tokenize

        # Configure Wikipedia API
        self.wiki = wikipediaapi.Wikipedia(
            user_agent='BM25MultiHopRetriever/1.0 (https://github.com/Theod0reWu/Deep-Learning-Multi-Hop-QA)',
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

        # Create the workflow graph
        self.graph = self._create_graph()
        
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
            direct_page = self.wiki.page(state.current_query)
            if direct_page.exists():
                self.logger.info(f"Found direct match for query: {state.current_query}")
                pages = [direct_page]
            else:
                # Then try search API
                search_url = 'https://en.wikipedia.org/w/api.php'
                search_params = {
                    'action': 'query',
                    'list': 'search',
                    'srsearch': state.current_query,
                    'format': 'json',
                    'srlimit': 5
                }
                headers = {'User-Agent': 'BM25MultiHopRetriever/1.0'}
                
                response = requests.get(search_url, params=search_params, headers=headers)
                response.raise_for_status()
                
                search_results = response.json()
                pages = []
                if 'query' in search_results and 'search' in search_results['query']:
                    for result in search_results['query']['search']:
                        try:
                            page = self.wiki.page(result['title'])
                            if page.exists():
                                pages.append(page)
                                self.logger.info(f"Found page: {page.title}")
                        except Exception as e:
                            self.logger.warning(f"Error fetching page {result['title']}: {str(e)}")
            
            # Update state with documents
            state.documents = []
            for page in pages:
                if page.title not in state.visited_pages:
                    content = page.text
                    if content:
                        state.documents.append((page.title, content[:5000]))  # Limit content size
                        # Add linked pages
                        for link in list(page.links.keys())[:3]:  # Limit to first 3 links
                            if link not in state.visited_pages:
                                link_page = self.wiki.page(link)
                                if link_page.exists():
                                    state.documents.append((link, link_page.text[:5000]))
            
            return state
            
        except Exception as e:
            self.logger.error(f"Error searching Wikipedia: {str(e)}")
            return state

    def _tokenize(self, text: str) -> List[str]:
        """Tokenize text for BM25."""
        return self.word_tokenize(text.lower())

    def _rank_documents(self, state: State) -> State:
        """Rank documents using BM25."""
        node_name = f"rank_{state.current_iteration}"
        ranked_name = f"ranked_{state.current_iteration}"
        
        # Add dynamic nodes and edges to graph
        self.graph.add_node(node_name, type="operation", label=f"Rank {state.current_iteration}")
        self.graph.add_node(ranked_name, type="data", label=f"Ranked {state.current_iteration}")
        self.graph.add_edge(node_name, ranked_name, label=f"rank_{state.current_iteration}")
        
        if not state.documents:
            return state
            
        try:
            # Tokenize documents
            tokenized_docs = [self._tokenize(doc[1]) for doc in state.documents]
            
            # Create BM25 instance
            bm25 = BM25Okapi(tokenized_docs)
            
            # Get scores
            tokenized_query = self._tokenize(state.current_query)
            doc_scores = bm25.get_scores(tokenized_query)
            
            # Sort documents by score
            ranked_docs = [(state.documents[i][0], state.documents[i][1], score) 
                          for i, score in enumerate(doc_scores)]
            ranked_docs.sort(key=lambda x: x[2], reverse=True)
            
            # Update state with ranked documents
            for title, content, _ in ranked_docs[:2]:  # Keep top 2 documents
                if title not in state.visited_pages:
                    state.visited_pages.add(title)
                    state.context_history.append(f"From {title}:\n{content[:2000]}")
            
            return state
            
        except Exception as e:
            self.logger.error(f"Error ranking documents: {str(e)}")
            return state

    def _generate_next_query(self, state: State) -> State:
        """Generate a search query using Gemini API with rate limiting."""
        node_name = f"query_{state.current_iteration}"
        
        # Add dynamic node and edge to graph
        self.graph.add_node(node_name, type="operation", label=f"Generate Query {state.current_iteration}")
        self.graph.add_edge(node_name, f"search_{state.current_iteration+1}", label=f"generate_{state.current_iteration}")
        
        try:
            # Rate limiting
            current_time = time.time()
            time_since_last = current_time - self.last_api_call
            if time_since_last < self.min_delay:
                time.sleep(self.min_delay - time_since_last)
            
            # Use last 3 context pieces for query generation
            context = "\n\n".join(state.context_history[-3:])
            
            prompt = f"Given this context: {context}\nAnd this question: {state.question}\nGenerate a concise search query (max 5 words) to find relevant Wikipedia articles."
            response = self.model.generate_content(prompt)
            
            self.last_api_call = time.time()
            state.current_query = response.text.strip()
            state.current_iteration += 1
            
            return state
            
        except Exception as e:
            self.logger.error(f"Error generating search query: {str(e)}")
            # Fallback to basic query from question
            state.current_query = ' '.join(state.question.split()[:5])
            state.current_iteration += 1
            return state

    def _generate_answer(self, state: State) -> State:
        """Generate final answer using collected context."""
        node_name = "synthesize"
        
        # Add dynamic edge to graph
        self.graph.add_edge(node_name, "end", label="final_answer")
        
        try:
            # Rate limiting
            current_time = time.time()
            time_since_last = current_time - self.last_api_call
            if time_since_last < self.min_delay:
                time.sleep(self.min_delay - time_since_last)
            
            if len(state.context_history) > 1:
                # Combine context with newlines between documents
                context = "\n\n".join(state.context_history)
                
                prompt = f"""Based on the following context, answer the question. Include only information that is supported by the context.

Question: {state.question}

Context:
{context}

Answer:"""
                
                response = self.model.generate_content(prompt)
                self.last_api_call = time.time()
                state.answer = response.text.strip()
            else:
                state.answer = "I could not find relevant information to answer this question."
            
            return state
            
        except Exception as e:
            self.logger.error(f"Error generating answer: {str(e)}")
            state.answer = "Sorry, I was unable to generate an answer due to an error."
            return state

    def retrieve(self, question: str) -> Tuple[str, List[str], set]:
        """
        Perform multi-hop retrieval with visualization.
        
        Returns:
            Tuple containing:
            - Final answer
            - List of context documents
            - Set of visited page titles
        """
        # Initialize state
        state = State(
            question=question,
            current_query=question  # Initial query is the question itself
        )
        
        # Run the workflow
        while state.current_iteration < 3:  # Max 3 hops
            state = self._search_wikipedia(state)
            state = self._rank_documents(state)
            state = self._generate_next_query(state)
        
        # Generate final answer
        state = self._generate_answer(state)
        
        # Save visualization with improved layout
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
        
        return state.answer, state.context_history, state.visited_pages
