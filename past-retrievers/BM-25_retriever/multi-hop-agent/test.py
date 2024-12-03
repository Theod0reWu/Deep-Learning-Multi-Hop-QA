from dotenv import load_dotenv
import os

# Load environment variables from the .env file
load_dotenv()

# Now you can access the API key using os.getenv()
api_key = os.getenv("GEMINI_API_KEY")

# Check if the API key is correctly loaded
print(f"API Key loaded: {api_key is not None}")
