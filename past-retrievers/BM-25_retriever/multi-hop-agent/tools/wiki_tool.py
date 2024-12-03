from langchain_community.utilities import WikipediaAPIWrapper
import wikipedia
import re

# Configure Wikipedia
wikipedia.set_lang("en")
wikipedia.set_rate_limiting(True)

# Initialize Wikipedia API wrapper with the wikipedia package as the client
wiki_wrapper = WikipediaAPIWrapper(wiki_client=wikipedia, top_k_results=1)

def clean_title(title):
    """Clean and normalize the article title."""
    # Remove special characters and normalize spaces
    cleaned = re.sub(r'[^\w\s-]', '', title)
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()
    return cleaned

def get_best_match_title(query):
    """
    Search Wikipedia and return the most relevant article title.
    Handles disambiguation and redirects.
    """
    try:
        # First try exact match
        try:
            page = wikipedia.page(query, auto_suggest=False)
            return page.title
        except wikipedia.DisambiguationError as e:
            # If it's a disambiguation page, take the first suggestion
            if e.options:
                return e.options[0]
            return None
        except wikipedia.PageError:
            pass

        # If exact match fails, try searching
        search_results = wikipedia.search(query, results=5)
        if not search_results:
            return None

        # Try to find the most relevant result
        query_words = set(clean_title(query).lower().split())
        best_match = None
        best_score = 0

        for result in search_results:
            result_words = set(clean_title(result).lower().split())
            # Calculate word overlap score
            score = len(query_words & result_words) / len(query_words | result_words)
            if score > best_score:
                best_score = score
                best_match = result

        return best_match

    except Exception as e:
        print(f"Error in title matching: {e}")
        return None

def get_wikipedia_article(article_title):
    """
    Fetch the Wikipedia article for the given title.
    Uses fuzzy matching to find the most relevant article.
    """
    try:
        # Find the best matching article title
        best_match = get_best_match_title(article_title)
        if not best_match:
            print(f"No matching article found for: {article_title}")
            return None

        # If the matched title is different from the input, log it
        if best_match.lower() != article_title.lower():
            print(f"Using closest match: '{best_match}' for query: '{article_title}'")

        # Get the article content using the initialized wrapper
        content = wiki_wrapper.run(best_match)
        
        # Format the response
        return f"Page: {best_match}\nSummary: {content}"

    except Exception as e:
        print(f"Error fetching article for {article_title}: {e}")
        return None
