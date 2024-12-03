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

# Function to get the most recent responses within the context window
def get_context_window(responses, window_size=3):
    """Get the most recent responses within the context window."""
    if not responses:
        return ""
    recent_responses = responses[-window_size:]
    return "\n\nContext from previous searches:\n" + "\n".join([
        f"[Hop {len(responses)-i}]: {response}" 
        for i, response in enumerate(reversed(recent_responses))
    ])

# Extract temporal information from text
def extract_temporal_info(text):
    """Extract dates and temporal information from text."""
    # Extract years
    years = re.findall(r'\b(19|20)\d{2}\b', text)
    # Extract date ranges
    ranges = re.findall(r'((?:19|20)\d{2})(?:\s*[-–]\s*)((?:19|20)\d{2})', text)
    # Extract birth/death dates
    birth_death = re.findall(r'born\s+in\s+(\d{4})|(\d{4})\s*[-–]\s*(\d{4})|born\s+(\d{1,2}\s+[A-Za-z]+\s+\d{4})', text)
    return {
        'years': years,
        'ranges': ranges,
        'birth_death': birth_death
    }

# Extract key entities and their roles from text based on query context
def extract_key_entities(text, query):
    """Extract key entities and their roles from text based on query context."""
    entities = {
        'people': [],
        'organizations': [],
        'positions': [],
        'dates': [],
        'locations': [],
        'events': []
    }
    
    # Extract position-related information
    position_patterns = [
        r'(president|leader|chairman|secretary|minister|director|head) of [A-Z][A-Za-z\s]+',
        r'served as ([A-Za-z\s]+) (?:from|in|during) \d{4}',
        r'([A-Z][a-z]+ [A-Z][a-z]+) (?:served|worked|acted) as'
    ]
    
    # Extract people who held positions
    for pattern in position_patterns:
        matches = re.finditer(pattern, text, re.IGNORECASE)
        for match in matches:
            if match.group(1) not in entities['positions']:
                entities['positions'].append(match.group(1))
    
    # Extract people names (with or without positions)
    name_pattern = r'([A-Z][a-z]+ (?:[A-Z][a-z]+ )*[A-Z][a-z]+)'
    names = re.findall(name_pattern, text)
    for name in names:
        if name not in entities['people']:
            entities['people'].append(name)
    
    # Extract organizations
    org_pattern = r'(?:The |)[A-Z][a-z]+ (?:University|Corporation|Company|Institute|Association|Organization)'
    orgs = re.findall(org_pattern, text)
    entities['organizations'].extend(orgs)
    
    # Extract dates (including ranges and specific dates)
    date_patterns = [
        r'\b(19|20)\d{2}\b',
        r'(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},\s+\d{4}',
        r'(?:early|mid|late)\s+\d{4}s'
    ]
    for pattern in date_patterns:
        dates = re.findall(pattern, text)
        entities['dates'].extend(dates)
    
    # Extract locations
    location_pattern = r'(?:in|at|from|to)\s+([A-Z][a-z]+(?:,\s+[A-Z][a-z]+)*)'
    locations = re.findall(location_pattern, text)
    entities['locations'].extend(locations)
    
    # Extract events
    event_patterns = [
        r'(?:the|The)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\s+(?:War|Battle|Revolution|Conference|Treaty))',
        r'(?:World\s+War\s+[III]+)',
        r'(?:the|The)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\s+(?:Crisis|Incident|Movement))'
    ]
    for pattern in event_patterns:
        events = re.findall(pattern, text)
        entities['events'].extend(events)
    
    return entities

# Check if the response is sufficient (can be customized)
def is_sufficient_response(response, query, context=""):
    """Check if we have enough information to answer the query."""
    full_context = f"{context}\n{response}" if context else response
    temporal_info = extract_temporal_info(full_context)
    entities = extract_key_entities(full_context, query)
    
    # For temporal reasoning questions
    temporal_indicators = ['when', 'what year', 'how long', 'during', 'before', 'after', 'while']
    if any(indicator in query.lower() for indicator in temporal_indicators):
        if 'alive' in query.lower() and 'founded' in query.lower():
            has_founding_date = any('founded' in full_context.lower() and year in full_context for year in temporal_info['years'])
            has_birth_date = any('born' in str(birth) for birth in temporal_info['birth_death'])
            return has_founding_date and has_birth_date
        
        if any(pos.lower() in query.lower() for pos in entities['positions']):
            return bool(entities['people'] and temporal_info['ranges'])
        
        # For general temporal questions
        return bool(entities['dates'] and (entities['people'] or entities['organizations'] or entities['events']))
    
    # For position-holding questions
    if any(pos.lower() in query.lower() for pos in entities['positions']):
        return bool(entities['people'] and entities['dates'])
    
    # For event-based questions
    if entities['events']:
        return bool(entities['dates'] or entities['locations'])
    
    # For organization questions
    if entities['organizations']:
        return bool(entities['dates'] or entities['locations'] or entities['people'])
    
    return False

# Generalize the query refinement based on article context
def refine_query(previous_response, original_query, context=""):
    """Refined query generation based on previous article content and context."""
    full_context = f"{context}\n{previous_response}" if context else previous_response
    temporal_info = extract_temporal_info(full_context)
    entities = extract_key_entities(full_context, original_query)
    
    # Check if query involves temporal reasoning
    temporal_indicators = ['when', 'what year', 'how long', 'during', 'before', 'after', 'while']
    if any(indicator in original_query.lower() for indicator in temporal_indicators):
        if any('founded' in year_context.lower() for year_context in temporal_info['years']):
            for person in entities['people']:
                return f"When was {person} born?"
        elif entities['people'] and not any('born' in str(birth) for birth in temporal_info['birth_death']):
            return f"What are the birth and death dates of {entities['people'][-1]}?"
    
    # If we found specific entities, use them to refine the query
    if entities['people']:
        return f"What was {entities['people'][-1]}'s role and when did they serve?"
    elif temporal_info['years']:
        return f"Who held the position mentioned in the query during {temporal_info['years'][-1]}?"
    
    return original_query

# Function to perform dynamic multi-hop query generation and retrieval with retries
def dynamic_query_agent(query, model, max_hops=10, context_window_size=3):
    queries = [query]  # Initialize with the original query
    responses = []
    retries = 0
    articles_used = []  # Track articles used during multi-hop process

    # Loop to perform multi-hop query generation and retrieval
    for hop in range(max_hops):
        current_query = queries[-1]  # The most recent query
        
        # Get context from previous responses
        context = get_context_window(responses, context_window_size)

        # Use the template to generate the next Wikipedia article title
        prompt = prompt_template.format(
            previous_response=responses[-1] if responses else "",
            original_query=query,
            accumulated_context=context
        )
        response = model.generate_content(prompt)
        next_article_title = response.text.strip()

        print(f"\nHop {hop + 1}:")
        print(f"Generated next article title: {next_article_title}")

        # Query Wikipedia for the next article
        wiki_response = get_wikipedia_article(next_article_title)
        responses.append(wiki_response)

        # Track the Wikipedia article used in the retrieval
        if wiki_response:
            article_link = f"https://en.wikipedia.org/wiki/{next_article_title.replace(' ', '_')}"
            articles_used.append(article_link)
            print(f"Article link: {article_link}")

        # Check if the response is sufficient to stop the process
        if is_sufficient_response(wiki_response, query, context):
            break

        # Retry logic - refine query if needed
        if retries < 3:
            retries += 1
            refined_query = refine_query(wiki_response, query, context)
            queries.append(refined_query)
            print(f"Retry {retries}: Refining query to: {refined_query}")
            time.sleep(2)  # Avoid overwhelming the API
        else:
            print("Max retries reached, stopping.")
            break

    return responses, articles_used
