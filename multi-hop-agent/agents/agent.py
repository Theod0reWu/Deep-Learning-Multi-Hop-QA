import time
from tools.wiki_tool import get_wikipedia_article
from prompts.prompt_templates import prompt_template

# Initialize OpenAI language model
from langchain.chat_models import ChatOpenAI

# Initialize the language model for GPT, Claude, etc.
def select_model(model_name="GPT"):
    if model_name == "GPT":
        return ChatOpenAI(temperature=0)  # Adjust the temperature based on your use case
    # Add other models like Claude, Gemini, etc. here
    else:
        raise ValueError("Unsupported model")

# Check if the response is sufficient (can be customized)
def is_sufficient_response(response):
    """Checks if the response contains enough information."""
    return "answer" in response or len(response.split()) > 50  # Example condition

# Function to refine the query based on the previous response
def refine_query(previous_response):
    """Refines the query based on the previous response."""
    if "scientist" in previous_response:
        return "More details on famous scientists"
    return "Retrieve additional relevant information"

# Function to perform dynamic multi-hop retrieval with retries
def dynamic_query_agent(query, model, max_hops=10):
    """
    Performs dynamic multi-hop query generation and retrieval.
    Arguments:
    query -- The original query to answer.
    model -- The language model for generating the next article title.
    max_hops -- The maximum number of hops (retrieval steps) to perform.

    Returns:
    responses -- List of responses at each hop.
    articles_used -- List of Wikipedia articles used during the retrieval.
    """
    queries = [query]  # Initialize with the original query
    responses = []
    retries = 0
    articles_used = []  # Track articles used during multi-hop process

    # Loop to perform multi-hop query generation and retrieval
    for hop in range(max_hops):
        current_query = queries[-1]  # The most recent query

        # Use the prompt template to generate the next Wikipedia article title
        prompt = prompt_template.format(previous_response=responses[-1] if responses else "", original_query=query)
        next_article_title = model.run(prompt)  # Generate the next article title
        print(f"Generated next article title: {next_article_title}")

        # Query Wikipedia for the next article
        response = get_wikipedia_article(next_article_title)
        responses.append(response)

        # Track the Wikipedia article used in the retrieval
        if response:  # Ensure a valid response
            article_link = f"https://en.wikipedia.org/wiki/{next_article_title.replace(' ', '_')}"
            articles_used.append(article_link)
            print(f"Article link: {article_link}")

        # Check if the response is sufficient to stop the process
        if is_sufficient_response(response):
            break

        # Retry logic - refine query if needed (based on the previous response)
        if retries < 3:  # Limit retries to 3
            retries += 1
            refined_query = refine_query(response)
            queries.append(refined_query)
            print(f"Retry {retries}: Refined query - {refined_query}")
            time.sleep(2)  # Wait to avoid overwhelming the API
        else:
            print(f"Max retries reached, stopping.")
            break

    return responses, articles_used