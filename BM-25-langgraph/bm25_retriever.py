from typing import Dict, List, Tuple, Any, Annotated, TypedDict, Optional
import os
from dotenv import load_dotenv
import google.generativeai as genai
import wikipediaapi
import time
from rank_bm25 import BM25Okapi
import logging
import nltk
from langchain_core.messages import HumanMessage
from dataclasses import dataclass, field
from langgraph.graph import StateGraph
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
import random

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('bm25_retriever_langgraph')

# Define keys that can be updated
Question = Annotated[str, "question"]
Context = Annotated[List[str], "context"]
CurrentIteration = Annotated[int, "current_iteration"]
CurrentQueryNum = Annotated[int, "current_query_num"]
VisitedPages = Annotated[set, "visited_pages"]
Answer = Annotated[Optional[str], "answer"]

class AgentState(TypedDict):
    """State for the multi-hop retrieval agent."""
    question: Question
    context: Context
    current_iteration: CurrentIteration
    current_query_num: CurrentQueryNum
    visited_pages: VisitedPages
    answer: Answer

def create_empty_state(question: str) -> AgentState:
    """Create an empty initial state."""
    return {
        "question": question,
        "context": [question],
        "current_iteration": 0,
        "current_query_num": 0,
        "visited_pages": set(),
        "answer": None
    }

class BM25MultiHopRetriever:
    def __init__(self, gemini_api_key=None, n_iterations=5, n_queries=5, n_docs=2):
        """Initialize the BM25 Multi-hop Retriever with LangGraph integration.
        
        Args:
            gemini_api_key: API key for Gemini
            n_iterations: Number of retrieval iterations (n in paper)
            n_queries: Number of queries per iteration (k in paper)
            n_docs: Number of documents to retrieve per query (n_docs in paper)
        """
        self.logger = logger
        self.n_iterations = n_iterations
        self.n_queries = n_queries
        self.n_docs = n_docs
        
        # Initialize NLTK
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt', quiet=True)
        self.word_tokenize = nltk.tokenize.word_tokenize

        # Configure Wikipedia
        self.wiki = wikipediaapi.Wikipedia(
            language='en',
            extract_format=wikipediaapi.ExtractFormat.WIKI,
            user_agent='BM25MultiHopRetriever/1.0'
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
        self.max_retries = 5
        
        # Create the workflow graph
        self.workflow = self._create_workflow()

    def _tokenize(self, text: str) -> List[str]:
        """Tokenize text for BM25."""
        return self.word_tokenize(text.lower())

    def _call_gemini_with_backoff(self, prompt: str) -> str:
        """Call Gemini API with exponential backoff for rate limits."""
        for attempt in range(self.max_retries):
            try:
                # Rate limiting
                current_time = time.time()
                time_since_last = current_time - self.last_api_call
                if time_since_last < self.min_delay:
                    time.sleep(self.min_delay - time_since_last)
                
                response = self.model.generate_content(prompt)
                self.last_api_call = time.time()
                return response.text.strip()
                
            except Exception as e:
                if "rate limit exceeded" in str(e).lower():
                    wait_time = (2 ** attempt) + random.uniform(0, 1)  # Exponential backoff with jitter
                    self.logger.warning(f"Rate limit hit, waiting {wait_time:.2f} seconds...")
                    time.sleep(wait_time)
                    continue
                else:
                    raise e
        
        raise Exception("Max retries exceeded for Gemini API call")

    def _search_wikipedia(self, state: AgentState) -> AgentState:
        """Search Wikipedia and retrieve documents using BM25."""
        try:
            # Generate search query
            context = "\n\n".join(state["context"][-3:])  # Use last 3 context pieces
            prompt = f"""Given this context: {context}
And this question: {state["question"]}
Generate a concise search query (max 5 words) to find relevant Wikipedia articles."""
            
            query = self._call_gemini_with_backoff(prompt)
            
            # Search Wikipedia using the wiki API
            page = self.wiki.page(query)
            if not page.exists():
                self.logger.warning(f"No Wikipedia page found for query: {query}")
                return state
                
            # Get links from the page
            links = list(page.links.values())[:self.n_docs * 2]  # Get extra links in case some fail
            
            new_docs = []
            for link_page in links:
                if len(new_docs) >= self.n_docs:
                    break
                    
                try:
                    if not link_page.exists():
                        continue
                        
                    # Skip if we've seen this page
                    if link_page.title in state["visited_pages"]:
                        continue
                    
                    # Add to visited pages and context
                    state["visited_pages"].add(link_page.title)
                    summary = link_page.summary[:1000]  # Limit summary length
                    if summary:
                        new_docs.append(f"Title: {link_page.title}\n{summary}")
                        
                except Exception as e:
                    self.logger.warning(f"Error fetching Wikipedia page {link_page.title}: {str(e)}")
                    continue
            
            # Update context with new documents
            state["context"].extend(new_docs)
            
            # Update iteration counters
            state["current_query_num"] += 1
            if state["current_query_num"] >= self.n_queries:
                state["current_iteration"] += 1
                state["current_query_num"] = 0
            
            return state
            
        except Exception as e:
            self.logger.error(f"Error in search: {str(e)}")
            return state

    def _generate_answer(self, state: AgentState) -> AgentState:
        """Generate final answer using collected context."""
        try:
            if len(state["context"]) > 1:
                # Combine context with newlines between documents
                context = "\n\n".join(state["context"])
                
                prompt = f"""Based on the following context, answer the question. Include only information that is supported by the context.

Question: {state["question"]}

Context:
{context}

Answer:"""
                
                state["answer"] = self._call_gemini_with_backoff(prompt)
            else:
                state["answer"] = "I could not find relevant information to answer this question."
            
            return state
            
        except Exception as e:
            self.logger.error(f"Error generating answer: {str(e)}")
            state["answer"] = "Sorry, I was unable to generate an answer due to an error."
            return state

    def _should_continue(self, state: AgentState) -> str:
        """Determine if we should continue searching or generate answer."""
        if state["current_iteration"] >= self.n_iterations:
            return "generate_answer"
        return "search"

    def _create_workflow(self):
        """Create the LangGraph workflow."""
        # Create the graph
        workflow = StateGraph(AgentState)
        
        # Add nodes
        workflow.add_node("search", self._search_wikipedia)
        workflow.add_node("generate_answer", self._generate_answer)
        
        # Add edges
        workflow.add_edge("search", "generate_answer")
        workflow.add_conditional_edges(
            "search",
            self._should_continue,
            {
                "search": "search",
                "generate_answer": "generate_answer"
            }
        )
        
        # Set entry point
        workflow.set_entry_point("search")
        
        # Compile with type hints
        return workflow.compile()

    def retrieve(self, question: str) -> Tuple[str, List[str], set]:
        """
        Perform multi-hop retrieval using LangGraph.
        
        Args:
            question: The question to answer
            
        Returns:
            Tuple containing:
            - Final answer
            - List of context documents
            - Set of visited page titles
        """
        # Initialize state
        state = create_empty_state(question)
        
        # Invoke the workflow
        final_state = self.workflow.invoke(state)
        
        return (
            final_state["answer"],
            final_state["context"],
            final_state["visited_pages"]
        )
