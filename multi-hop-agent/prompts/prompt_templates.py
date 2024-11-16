prompt_template = """
Here are some examples of generating the next relevant Wikipedia article title:

Example 1:
Query: "Famous scientists of the 19th century"
Response: "Marie Curie"
Next Article Title: "Marie Curie"

Example 2:
Query: "World War II timeline"
Response: "D-Day"
Next Article Title: "Normandy landings"

Example 3:
Query: "Famous paintings by Leonardo da Vinci"
Response: "Mona Lisa"
Next Article Title: "The Last Supper"

Given the current information: "{previous_response}",
generate the next relevant Wikipedia article title that could help answer the query:
"{original_query}"
"""
