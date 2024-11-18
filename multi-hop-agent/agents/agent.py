import time
from tools.wiki_tool import get_wikipedia_article
from prompts.prompt_templates import prompt_template  # Import the prompt template
from dotenv import load_dotenv
import os
import google.generativeai as genai
import re

# Load environment variables from the .env file
load_dotenv()

# Use the loaded API key to configure Gemini
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# Function to initialize the Gemini model
def select_model(model_name="Gemini"):
    try:
        model = genai.GenerativeModel("gemini-1.5-flash")  # You can replace with any other available Gemini model
        return model
    except Exception as e:
        print(f"Error initializing Gemini model: {e}")
        raise

# Check if the response is sufficient (can be customized)
def is_sufficient_response(response, query):
    """
    Generalized function to check if the response from a Wikipedia article contains
    enough information to answer the query.
    """
    # Check if response is longer than a minimal threshold
    if len(response.split()) < 50:
        return False

    # If the query is about the person's details, such as birthplace, achievements, or life events
    if "born" in query.lower() or "birthplace" in query.lower():
        if "born" in response.lower() or "birthplace" in response.lower():
            return True

    # Check if the query involves an event (e.g., "happened", "event")
    if "event" in query.lower() or "happened" in query.lower():
        if "happened" in response.lower() or "event" in response.lower():
            return True

    return False




# Generalize the query refinement based on article context
def refine_query(previous_response, original_query):
    """Generalized refinement to generate the next query based on the previous article content."""
    # If the article mentions a name, it might be useful to look deeper into their achievements or life events
    if "born" in previous_response.lower():
        return "Find more details about this person's achievements or professional work."

    # If the article contains references to places, further exploration of that location could be useful
    if "location" in previous_response.lower():
        return "What significant events happened in this location?"

    # If there are mentions of dates, refine the query to explore specific periods in the person’s life
    if "date" in previous_response.lower():
        return "What important events happened around this time?"

    # If the article does not provide the expected answer, continue exploring
    return original_query


# Function to perform dynamic multi-hop query generation and retrieval with retries
def dynamic_query_agent(query, model, max_hops=10):
    queries = [query]  # Initialize with the original query
    responses = []
    retries = 0
    articles_used = []  # Track articles used during multi-hop process

    # Loop to perform multi-hop query generation and retrieval
    for hop in range(max_hops):
        current_query = queries[-1]  # The most recent query

        # Use the template to generate the next Wikipedia article title
        prompt = prompt_template.format(previous_response=responses[-1] if responses else "", original_query=query)
        response = model.generate_content(prompt)  # Get next article title from the model
        next_article_title = response.text.strip()

        print(f"Generated next article title: {next_article_title}")

        # Query Wikipedia for the next article
        wiki_response = get_wikipedia_article(next_article_title)
        responses.append(wiki_response)

        # Track the Wikipedia article used in the retrieval
        if wiki_response:  # Ensure a valid response
            article_link = f"https://en.wikipedia.org/wiki/{next_article_title.replace(' ', '_')}"
            articles_used.append(article_link)
            print(f"Article link: {article_link}")

        # Check if the response is sufficient to stop the process
        if is_sufficient_response(wiki_response, query):
            break

        # Retry logic - refine query if needed
        if retries < 3:
            retries += 1
            refined_query = refine_query(wiki_response, query)
            queries.append(refined_query)
            print(f"Retry {retries}: Refining query.")
            time.sleep(2)  # Avoid overwhelming the API
        else:
            print("Max retries reached, stopping.")
            break

    return responses, articles_used
