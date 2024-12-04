# Multi-Hop Wikipedia Question Answering with LangGraph

This implementation extends the LangGraph RAG template to create a sophisticated multi-hop Wikipedia question answering system using a hierarchical multi-agent architecture.

## Architecture Overview

The system consists of three main components:

1. **Index Graph**: Handles Wikipedia article indexing using Elasticsearch and OpenAI embeddings
2. **Main Retrieval Graph**: Orchestrates the overall QA process
3. **Researcher Subgraph**: Manages multi-hop research through iterative retrieval

## Key Features

### 1. Controlled Research Steps
To limit the number of research steps:

```python
@dataclass
class RetrievalConfiguration(BaseConfiguration):
    max_research_steps: int = 3  # Configurable maximum steps
    max_docs_per_query: int = 3  # Docs retrieved per query
```

### 2. Wikipedia Integration
The system uses the Wikipedia API for dynamic article retrieval:

```python
def retrieve_wiki_article(title: str) -> Document:
    wiki = wikipediaapi.Wikipedia('en')
    page = wiki.page(title)
    return Document(
        page_content=page.text,
        metadata={"title": title, "url": page.fullurl}
    )
```

### 3. Multi-Hop Research Process

1. **Query Analysis**: The main graph analyzes the question to create a research plan
2. **Iterative Research**: For each step (limited by `max_research_steps`):
   - Generate focused queries
   - Retrieve relevant Wikipedia articles
   - Extract and synthesize information
3. **Final Response**: Synthesize findings into a comprehensive answer

## Implementation Highlights

1. **State Management**
   - Uses typed state classes for each graph
   - Tracks research progress and accumulated context
   - Manages message history for better context awareness

2. **Conditional Routing**
   - Dynamic research path based on query complexity
   - Early termination if answer is found
   - Graceful handling of ambiguous queries

3. **Vector Search**
   - Dense embeddings for semantic search
   - Hybrid retrieval combining BM25 and vector similarity
   - Efficient caching of frequently accessed articles

## Usage Example

```python
config = RetrievalConfiguration(
    max_research_steps=3,
    max_docs_per_query=3
)

graph = create_retrieval_graph(
    elasticsearch_url="...",
    config=config
)

# Multi-hop question
result = graph.invoke({
    "query": "What impact did Einstein's work on special relativity have on quantum mechanics?"
})
```

## Best Practices

1. **Research Step Control**
   - Set reasonable limits for research steps
   - Implement early stopping when sufficient information is found
   - Use confidence scoring to validate findings

2. **Context Management**
   - Maintain relevant context between hops
   - Prune irrelevant information
   - Track source documents for citations

3. **Error Handling**
   - Graceful degradation with API limits
   - Fallback strategies for failed retrievals
   - Clear feedback on research progress
