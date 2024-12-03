prompt_template = """
Given the following information and context:

{accumulated_context}

Current Query: "{original_query}"
Most Recent Article: "{previous_response}"

Your task is to generate the next most relevant Wikipedia article title that would help answer the query. Pay special attention to:

1. Core Information Requirements:
   - The main entities mentioned in the query
   - Their relationships and connections
   - Required attributes or properties
   - Time periods or dates if relevant

2. Key Information Needed:
   - For temporal questions:
     * Birth/death dates for people
     * Founding dates for organizations
     * Terms of service in positions
     * Sequence of historical events
   - For factual questions:
     * Direct attributes of entities
     * Relationships between entities
     * Supporting context and details

3. Prioritize articles about:
   - The specific entities mentioned in the query
   - Related entities that provide missing context
   - Historical or temporal context when needed
   - Lists or overview articles that connect multiple facts

Look for missing pieces of information. If you find:
- An entity but missing key dates → Look up their main article
- A relationship but missing context → Look up articles about connected entities
- A position but no holder → Look up "List of [position] of [organization]"
- An event but no timeline → Look up articles about the time period

Please output ONLY the Wikipedia article title that would best provide the missing information, nothing else.
"""
