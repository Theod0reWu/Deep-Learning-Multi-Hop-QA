### NOTE: Set api keys in the .env file 

### Next step: adapt this file -- main.py, to ingest from db
###            and store the necessary info for evaluation, which 
###            will be done in another file
from agents.agent import dynamic_query_agent, select_model

def main():
    # Example initial query
    query = "Famous scientists of the 19th century"
    model_name = "GPT"  # Dynamically set which model to use: Claude, GPT, Gemini, Llama

    # Select the model dynamically
    model = select_model(model_name)

    # Call the dynamic multi-hop query agent
    responses, articles_used = dynamic_query_agent(query, model)

    # Print the responses and articles used during the process
    print("\nFinal answer:")
    for idx, res in enumerate(responses):
        print(f"Hop {idx + 1}: {res}")

    print("\nArticles used in the process:")
    for article in articles_used:
        print(article)

if __name__ == "__main__":
    main()
