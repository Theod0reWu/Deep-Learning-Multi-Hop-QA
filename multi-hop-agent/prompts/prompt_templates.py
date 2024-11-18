prompt_template = """
Given the following information from a Wikipedia article:

Example 1:
Previous Article: "Albert Einstein was a German-born theoretical physicist who developed the theory of relativity."
Current Query: "Where was Albert Einstein born?"
Best Possible Next Article Title: "Albert Einstein"  # The article about the person itself should have the birthplace info.

Example 2:
Previous Article: "Winston Churchill was a British statesman who served as Prime Minister of the United Kingdom during World War II."
Current Query: "Where did Winston Churchill live?"
Best Possible Next Article Title: "Winston Churchill"  # Assumes the article about Churchill will contain info about where he lived.

Example 3:
Previous Article: "Frida Kahlo was a Mexican artist known for her self-portraits and works inspired by nature and artifacts of Mexico."
Current Query: "What is Frida Kahlo known for?"
Best Possible Next Article Title: "Frida Kahlo"  # This article will contain details about her work and contributions.

Now, based on the previous information:

Previous Article: "{previous_response}"
Current Query: "{original_query}"

Your task is to generate the rough title of the next most relevant Wikipedia article that would likely help answer the current query. Consider that the next article should help to further answer the query based on the context provided in the previous article. Focus on generating article titles that will most likely contain relevant information to answer the query.

Your next article title could be about a person, an event, a concept, or any other relevant topic that could further help answer the query. Do not worry if the title is not perfect; generate titles that will help explore the context further.
"""



# prompt_template = """
# Given the following information from a Wikipedia article:

# Example 1:
# Previous Article: "Albert Einstein was a German-born theoretical physicist who developed the theory of relativity."
# Current Query: "Where was Albert Einstein born?"
# Next Article Title: "Albert Einstein"  # Assumes the birthplace info will be found in the article itself.

# Example 2:
# Previous Article: "Winston Churchill was a British statesman who served as Prime Minister of the United Kingdom during World War II."
# Current Query: "Where did Winston Churchill live?"
# Next Article Title: "Winston Churchill"  # Assumes the living details are in the article about him.

# Example 3:
# Previous Article: "Frida Kahlo was a Mexican artist known for her self-portraits and works inspired by nature and artifacts of Mexico."
# Current Query: "What is Frida Kahlo known for?"
# Next Article Title: "Frida Kahlo"  # Assumes the information about her artistic work is in her article.

# Given the following information from a Wikipedia article:

# Previous Article: "{previous_response}"

# Current Query: "{original_query}"

# Your task is to generate the next most relevant Wikipedia article that might help answer this query.
# """


# # This template helps guide the model in generating the next relevant Wikipedia article title
# # based on the information from the previous response.

# prompt_template = """
# The goal of this process is to answer the following query:
# "{original_query}"

# Here are some examples of generating the next relevant Wikipedia article title based on the information in the previous response:

# Example 1:
# Query: "Who is the wife of Abraham Lincoln?"
# Response: "Mary Todd Lincoln"
# Next Article Title: "Mary Todd Lincoln"

# Example 2:
# Query: "What is the birthplace of Abraham Lincoln?"
# Response: "Hodgenville, Kentucky"
# Next Article Title: "Hodgenville, Kentucky"

# Example 3:
# Query: "What is the birthday of Mary Todd Lincoln?"
# Response: "December 13, 1818"
# Next Article Title: "Mary Todd Lincoln"

# Given the current information from the previous response: "{previous_response}",
# Generate the next relevant Wikipedia article title that could help answer the original query: "{original_query}"

# If the response has already provided sufficient information to answer the query (e.g., leader's name, service dates, or whether they were alive at the founding), stop generating further article titles. If more details are needed, create the next article title that will lead to the next logical step in finding the answer.

# Ensure that the next article title is directly related to the topic of the query, based on the previous response. In this case, the model should specifically look for the **leader's name** and any related **service dates** to verify the timeline.
# """
