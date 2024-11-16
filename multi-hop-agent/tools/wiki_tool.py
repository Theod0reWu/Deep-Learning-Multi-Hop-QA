from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper

# Initialize Wikipedia API wrapper
wikipedia = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())

def get_wikipedia_article(article_title):
    """Fetch the Wikipedia article for the given title."""
    try:
        return wikipedia.run(article_title)
    except Exception as e:
        print(f"Error fetching article for {article_title}: {e}")
        return None
