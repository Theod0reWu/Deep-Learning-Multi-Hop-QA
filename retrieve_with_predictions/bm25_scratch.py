import wikipediaapi
import google.generativeai as genai
import openai
from rank_bm25 import BM25Okapi
from typing import List, Set, Tuple, Optional
import os
import logging
import requests
from dotenv import load_dotenv
import time
import nltk
from nltk.tokenize import word_tokenize
from sentence_transformers import SentenceTransformer, util

# Set up logging
logging.basicConfig(level=logging.INFO)


class BM25MultiHopRetriever:
    def __init__(
        self, gemini_api_key=None, openai_api_key=None, llm_provider: str = "gemini"
    ):
        """Initialize the BM25 Multi-hop Retriever."""
        self.logger = logging.getLogger("bm25_retriever")
        self.logger.setLevel(logging.INFO)

        # Document cache
        self.doc_cache = {}  # title -> (title, full_text) mapping
        self.cache_hits = 0  # Track cache hits
        self.cache_misses = 0  # Track cache misses

        # Initialize NLTK
        try:
            nltk.data.find("tokenizers/punkt")
        except LookupError:
            nltk.download("punkt", quiet=True)
        self.word_tokenize = word_tokenize

        # Configure Wikipedia API
        self.wiki = wikipediaapi.Wikipedia(
            user_agent="BM25MultiHopRetriever/1.0 (https://github.com/YourRepo/)",
            language="en",
        )

        # Configure Gemini API
        # self.logger.info("Configuring Gemini API...")
        # if gemini_api_key:
        #     genai.configure(api_key=gemini_api_key)
        # else:
        #     load_dotenv()
        #     genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

        # self.model = genai.GenerativeModel("gemini-pro")
        # self.logger.info("Gemini API configured successfully")

        # Configure LLM API
        self.logger.info(f"Configuring {llm_provider.upper()} API...")
        if llm_provider == "gemini":
            # Gemini configuration (existing code)
            if gemini_api_key:
                genai.configure(api_key=gemini_api_key)
            else:
                load_dotenv()
                genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
            self.model = genai.GenerativeModel("gemini-pro")
            # self.generate_text = self._gemini_generate_text

        elif llm_provider == "openai":
            # OpenAI configuration
            if openai_api_key:
                openai.api_key = openai_api_key
            else:
                load_dotenv()
                openai.api_key = os.getenv("OPENAI_API_KEY")

            self.model = "gpt-3.5-turbo"  # Default model
            # self.generate_text = self._openai_generate_text

        else:
            raise ValueError(f"Unsupported LLM provider: {llm_provider}")

        self.logger.info(f"{llm_provider.upper()} API configured successfully")

        # Rate limiting
        self.last_api_call = 0
        self.min_delay = 1.0  # Minimum delay between API calls in seconds
        self.similarity_model = SentenceTransformer("all-MiniLM-L6-v2")

    def _get_wiki_page(self, title: str, query: str) -> Optional[Tuple[str, str]]:
        """Get the most relevant content of a Wikipedia page based on the query.

        Args:
            title: The title of the Wikipedia page.
            query: The current search query.

        Returns:
            Optional[Tuple[str, str]]: A tuple of (title, relevant_content) if found, None if the page doesn't exist.
        """
        if title in self.doc_cache:
            self.cache_hits += 1
            self.logger.info(f"Cache hit for '{title}' (Total hits: {self.cache_hits})")
            return self.doc_cache[title]

        self.cache_misses += 1
        self.logger.info(
            f"Cache miss for '{title}' (Total misses: {self.cache_misses})"
        )

        page = self.wiki.page(title)
        if not page.exists():
            return None

        full_text = page.text
        if not full_text:
            return None

        # Split the text into sentences or paragraphs
        sentences = nltk.sent_tokenize(full_text)

        # Tokenize the sentences for BM25 ranking
        tokenized_sentences = [self._tokenize(sentence) for sentence in sentences]

        # Use BM25 to score sentences based on the query
        bm25 = BM25Okapi(tokenized_sentences)
        query_tokens = self._tokenize(query)
        scores = bm25.get_scores(query_tokens)

        # Pair sentences with scores and sort by relevance
        ranked_sentences = sorted(
            zip(sentences, scores), key=lambda x: x[1], reverse=True
        )

        # Select sentences until the combined length is approximately 5000 characters
        relevant_content = []
        total_length = 0

        for sentence, score in ranked_sentences:
            if total_length + len(sentence) > 10000:
                break
            relevant_content.append(sentence)
            total_length += len(sentence)

        # Join the selected sentences and cache the result
        truncated_content = " ".join(relevant_content)
        self.doc_cache[title] = (title, truncated_content)
        # self.doc_cache[title] = (title, full_text)
        return self.doc_cache[title]

    def _search_wikipedia(self, query: str):
        """Search Wikipedia and return page objects."""
        try:
            # Direct page lookup
            direct_page = self.wiki.page(query)
            if direct_page.exists():
                self.logger.info(f"Found direct match for query: {query}")
                return [direct_page]

            # Search API
            search_url = "https://en.wikipedia.org/w/api.php"
            search_params = {
                "action": "query",
                "list": "search",
                "srsearch": query,
                "format": "json",
                "srlimit": 5,
            }
            headers = {"User-Agent": "BM25MultiHopRetriever/1.0"}

            response = requests.get(search_url, params=search_params, headers=headers)
            response.raise_for_status()

            search_results = response.json()
            if "query" in search_results and "search" in search_results["query"]:
                pages = []
                for result in search_results["query"]["search"]:
                    try:
                        page = self.wiki.page(result["title"])
                        if page.exists():
                            pages.append(page)
                            self.logger.info(f"Found page: {page.title}")
                    except Exception as e:
                        self.logger.warning(
                            f"Error fetching page {result['title']}: {str(e)}"
                        )
                return pages
            return []
        except Exception as e:
            self.logger.error(f"Error searching Wikipedia: {str(e)}")
            return []

    def _tokenize(self, text: str) -> List[str]:
        """Tokenize text for BM25."""
        return self.word_tokenize(text.lower())

    def _rank_documents(
        self, query: str, documents: List[Tuple[str, str]]
    ) -> List[Tuple[str, str, float]]:
        """Rank documents using BM25."""
        if not documents:
            return []

        try:
            # Tokenize documents
            tokenized_docs = [self._tokenize(doc[1]) for doc in documents]

            # Create BM25 instance
            bm25 = BM25Okapi(tokenized_docs)

            # Get scores
            tokenized_query = self._tokenize(query)
            doc_scores = bm25.get_scores(tokenized_query)

            # Sort documents by score
            ranked_docs = [
                (documents[i][0], documents[i][1], score)
                for i, score in enumerate(doc_scores)
            ]
            ranked_docs.sort(key=lambda x: x[2], reverse=True)

            return ranked_docs
        except Exception as e:
            self.logger.error(f"Error ranking documents: {str(e)}")
            return []

    def _generate_search_query(
        self, context: str, question: str, iteration: int, previous_queries: List[str]
    ) -> str:
        """Generate a search query using Gemini API with rate limiting."""
        try:
            # Rate limiting
            current_time = time.time()
            time_since_last = current_time - self.last_api_call
            if time_since_last < self.min_delay:
                time.sleep(self.min_delay - time_since_last)

            # Configure safety settings
            safety_settings = {
                "HARM_CATEGORY_HARASSMENT": "BLOCK_NONE",
                "HARM_CATEGORY_HATE_SPEECH": "BLOCK_NONE",
                "HARM_CATEGORY_SEXUALLY_EXPLICIT": "BLOCK_NONE",
                "HARM_CATEGORY_DANGEROUS_CONTENT": "BLOCK_NONE",
            }

            # For first iteration, extract key terms from the original question
            if iteration == 0:
                # Extract key entities and terms from the question
                prompt = f"""
                Here are some examples of questions, the required links to visit in order, and their answers:
                Question: How many years earlier would Punxsutawney Phil have to be canonically alive to have made a Groundhog Day prediction in the same state as the US capitol?
Links: [‘https://en.wikipedia.org/wiki/Punxsutawney_Phil', 'https://en.wikipedia.org/wiki/United_States_Capitol']
Answer: 87

Question: Imagine there is a building called Bronte tower whose height in feet is the same number as the dewey decimal classification for the Charlotte Bronte book that was published in 1847. Where would this building rank among tallest buildings in New York City, as of August 2024?
Links: [‘https://en.wikipedia.org/wiki/Charlotte_Bront%C3%AB', 'https://en.wikipedia.org/wiki/Jane_Eyre', 'https://en.wikipedia.org/wiki/List_of_tallest_buildings_in_New_York_City']
Answer: 37th

Question: What is the name of the vocalist from the first band to make it in the top 200 under the record label that produced the third studio album for Dismal Euphony?
Links: [‘https://en.wikipedia.org/wiki/Dismal_Euphony', 'https://en.wikipedia.org/wiki/All_Little_Devils', 'https://en.wikipedia.org/wiki/Nuclear_Blast', 'https://en.wikipedia.org/wiki/Meshuggah']
Answer: Jens Kidman

Question: If my future wife has the same first name as the 15th first lady of the United States' mother and her surname is the same as the second assassinated president's mother's maiden name, what is my future wife's name?
Links: [‘https://en.wikipedia.org/wiki/President_of_the_United_States', 'https://en.wikipedia.org/wiki/James_Buchanan', 'https://en.wikipedia.org/wiki/Harriet_Lane', 'https://en.wikipedia.org/wiki/List_of_presidents_of_the_United_States_who_died_in_office', 'https://en.wikipedia.org/wiki/James_A._Garfield']
Answer: Jane Ballou

Questions: Put these historical events in chronological order, starting with the earliest: The Beatles play Ed Sullivan, the fall of the Berlin Wall, The Great Depression, Atlanta Summer Games, World War I.
Links: [‘https://en.wikipedia.org/wiki/World_War_I', 'https://en.wikipedia.org/wiki/1996_Summer_Olympics', 'https://en.wikipedia.org/wiki/Fall_of_the_Berlin_Wall#:~:text=The%20fall%20of%20the%20Berlin,restrictions%20were%20overwhelmed%20and%20discarded.', 'https://en.wikipedia.org/wiki/The_Beatles', 'https://en.wikipedia.org/wiki/Great_Depression']
Answer: World War I, The Great Depression, The Beatles play Ed Sullivan, the fall of the Berlin Wall, Atlanta Summer Games.

Question: As of July 1, 2024, what is the parent company of the current record label of the singer of Edge of Seventeen?
Links: [‘https://en.wikipedia.org/wiki/Edge_of_Seventeen', 'https://en.wikipedia.org/wiki/Stevie_Nicks', 'https://en.wikipedia.org/wiki/Reprise_Records', 'https://en.wikipedia.org/wiki/Atlantic_Records', 'https://en.wikipedia.org/wiki/Modern_Records_(1980)', 'https://en.wikipedia.org/wiki/Warner_Music_Group', 'https://en.wikipedia.org/wiki/Warner_Records']
Answer: Warner Music Group

                Extract 3-5 key terms from the following question that would make a good Wikipedia search query.
Question: {question}

Return only the search terms, no explanation:"""

                response = self.model.generate_content(
                    prompt, safety_settings=safety_settings
                )
                return response.text.strip()

            # For subsequent iterations, generate new queries based on context
            prompt = f"""Given this context: {context}
And this question: {question}
Previous queries used: {', '.join(previous_queries)}

Here are some examples of questions, the required links to visit in order, and their answers:
                Question: How many years earlier would Punxsutawney Phil have to be canonically alive to have made a Groundhog Day prediction in the same state as the US capitol?
Links: [‘https://en.wikipedia.org/wiki/Punxsutawney_Phil', 'https://en.wikipedia.org/wiki/United_States_Capitol']
Answer: 87

Question: Imagine there is a building called Bronte tower whose height in feet is the same number as the dewey decimal classification for the Charlotte Bronte book that was published in 1847. Where would this building rank among tallest buildings in New York City, as of August 2024?
Links: [‘https://en.wikipedia.org/wiki/Charlotte_Bront%C3%AB', 'https://en.wikipedia.org/wiki/Jane_Eyre', 'https://en.wikipedia.org/wiki/List_of_tallest_buildings_in_New_York_City']
Answer: 37th

Question: What is the name of the vocalist from the first band to make it in the top 200 under the record label that produced the third studio album for Dismal Euphony?
Links: [‘https://en.wikipedia.org/wiki/Dismal_Euphony', 'https://en.wikipedia.org/wiki/All_Little_Devils', 'https://en.wikipedia.org/wiki/Nuclear_Blast', 'https://en.wikipedia.org/wiki/Meshuggah']
Answer: Jens Kidman

Question: If my future wife has the same first name as the 15th first lady of the United States' mother and her surname is the same as the second assassinated president's mother's maiden name, what is my future wife's name?
Links: [‘https://en.wikipedia.org/wiki/President_of_the_United_States', 'https://en.wikipedia.org/wiki/James_Buchanan', 'https://en.wikipedia.org/wiki/Harriet_Lane', 'https://en.wikipedia.org/wiki/List_of_presidents_of_the_United_States_who_died_in_office', 'https://en.wikipedia.org/wiki/James_A._Garfield']
Answer: Jane Ballou

Questions: Put these historical events in chronological order, starting with the earliest: The Beatles play Ed Sullivan, the fall of the Berlin Wall, The Great Depression, Atlanta Summer Games, World War I.
Links: [‘https://en.wikipedia.org/wiki/World_War_I', 'https://en.wikipedia.org/wiki/1996_Summer_Olympics', 'https://en.wikipedia.org/wiki/Fall_of_the_Berlin_Wall#:~:text=The%20fall%20of%20the%20Berlin,restrictions%20were%20overwhelmed%20and%20discarded.', 'https://en.wikipedia.org/wiki/The_Beatles', 'https://en.wikipedia.org/wiki/Great_Depression']
Answer: World War I, The Great Depression, The Beatles play Ed Sullivan, the fall of the Berlin Wall, Atlanta Summer Games.

Question: As of July 1, 2024, what is the parent company of the current record label of the singer of Edge of Seventeen?
Links: [‘https://en.wikipedia.org/wiki/Edge_of_Seventeen', 'https://en.wikipedia.org/wiki/Stevie_Nicks', 'https://en.wikipedia.org/wiki/Reprise_Records', 'https://en.wikipedia.org/wiki/Atlantic_Records', 'https://en.wikipedia.org/wiki/Modern_Records_(1980)', 'https://en.wikipedia.org/wiki/Warner_Music_Group', 'https://en.wikipedia.org/wiki/Warner_Records']
Answer: Warner Music Group

Generate a NEW and DIFFERENT search query (3-5 words) to help answer the question.
The query should:
1. Be different from previous queries
2. Include key terms from the question
3. Be suitable for Wikipedia article titles
4. Help answer aspects of the question not covered yet

Query:"""

            response = self.model.generate_content(
                prompt, safety_settings=safety_settings
            )

            self.last_api_call = time.time()
            generated_query = response.text.strip()

            # Ensure query is not too long
            words = generated_query.split()
            if len(words) > 5:
                generated_query = " ".join(words[:5])

            # If the generated query is too similar to previous queries, try again
            if generated_query in previous_queries:
                prompt += (
                    "\nIMPORTANT: The query MUST be completely different from: "
                    + ", ".join(previous_queries)
                )
                response = self.model.generate_content(
                    prompt, safety_settings=safety_settings
                )
                generated_query = response.text.strip()
                if len(words) > 5:
                    generated_query = " ".join(words[:5])

            return generated_query

        except Exception as e:
            self.logger.error(f"Error generating search query: {str(e)}")
            # Extract key terms from question for fallback
            words = question.split()
            if iteration == 0:
                return " ".join(words[:5])
            else:
                start_idx = (iteration * 3) % max(5, len(words))
                return " ".join(words[start_idx : start_idx + 5])

    def _generate_answer(self, question: str, context_history: List[str]) -> str:
        """Generate final answer using collected context."""
        try:
            # Rate limiting
            current_time = time.time()
            time_since_last = current_time - self.last_api_call
            if time_since_last < self.min_delay:
                time.sleep(self.min_delay - time_since_last)

            # Combine context
            context = "\n\n".join(context_history)
            prompt = f"""Based on the following context, answer the question. Include only information that is supported by the context. If you know the answer without the given context, return it directly. If the context does not seem to provide ample information, answer to the best of your ability.

Question: {question}

Context:
{context}

Answer:"""

            try:
                response = self.model.generate_content(
                    prompt,
                    safety_settings={
                        "HARM_CATEGORY_SEXUALLY_EXPLICIT": "BLOCK_NONE",
                        "HARM_CATEGORY_HATE_SPEECH": "BLOCK_ONLY_HIGH",
                        "HARM_CATEGORY_HARASSMENT": "BLOCK_ONLY_HIGH",
                        "HARM_CATEGORY_DANGEROUS_CONTENT": "BLOCK_ONLY_HIGH",
                    },
                )
                self.last_api_call = time.time()

                if hasattr(response, "text"):
                    return response.text.strip()
                elif hasattr(response, "parts") and response.parts:
                    return response.parts[0].text.strip()
                else:
                    return "Unable to generate response due to content safety filters. Please try rephrasing the question."

            except Exception as e:
                self.logger.error(f"Error in Gemini API call: {str(e)}")
                return "Error generating response. Please try rephrasing the question."

        except Exception as e:
            self.logger.error(f"Error generating answer: {str(e)}")
            return "Sorry, I was unable to generate an answer due to an error."

    def retrieve(
        self,
        question: str,
        ground_truth_answer: str,
        num_iterations: int,
        queries_per_iteration: int = 1,
        docs_per_query: int = 1,
        relative_score_threshold: float = 0.6,
        max_tokens: int = 10000,
    ):
        """
        Perform multi-hop retrieval.

        Args:
            question: The question to answer
            num_iterations: Target number of documents to collect (will stop early if reached)
            queries_per_iteration: Number of queries per iteration (default: 1)
            docs_per_query: Maximum documents to consider per query (default: 1)
            relative_score_threshold: Take documents scoring above (top_score * threshold) (default: 0.8)
            max_tokens: Maximum number of tokens to keep from each retrieved article (default: 2000)
        """
        self.context_history = [question]
        self.visited_pages = set()  # Track all visited pages
        self.context_docs = []  # Documents actually used in context
        self.context_titles = set()  # Track titles of documents in context
        self.processed_docs = []  # All documents processed, including those not used
        generated_queries = []  # Track all queries generated
        query_texts = []  # Track actual query texts without iteration numbers

        try:
            # Attempt to directly generate an answer
            # direct_answer = self._generate_answer(question, [])
            # gt_embedding = self.similarity_model.encode(
            #     ground_truth_answer, convert_to_tensor=True
            # )
            # answer_embedding = self.similarity_model.encode(
            #     direct_answer, convert_to_tensor=True
            # )
            # similarity = util.pytorch_cos_sim(gt_embedding, answer_embedding).item()

            # if similarity >= 0.8:
            #     self.logger.info(
            #         f"Direct answer confidence ({similarity:.2f}) exceeds threshold. Returning answer."
            #     )
            #     return (
            #         direct_answer,
            #         self.context_history,
            #         self.visited_pages,
            #         self.context_docs,
            #         self.processed_docs,
            #         generated_queries,
            #     )

            self.logger.info(
                "Direct answer confidence too low. Starting retrieval process."
            )

            iteration = 0
            self.logger.info(question)
            while (
                len(self.context_docs) < num_iterations
                and iteration < num_iterations * 2
            ):  # Allow up to 2x iterations to find enough docs
                self.logger.info(
                    f"Starting iteration {iteration + 1}, documents collected: {len(self.context_docs)}/{num_iterations}"
                )
                current_context = "\n\n".join(self.context_history)

                # Generate search query
                search_query = self._generate_search_query(
                    current_context, question, iteration, query_texts
                )
                self.logger.info(f"Generated search query: {search_query}")

                # Track queries
                query_texts.append(search_query)
                generated_queries.append(f"Iteration {iteration + 1}: {search_query}")

                # Search Wikipedia
                search_results = self._search_wikipedia(search_query)
                if not search_results:
                    iteration += 1
                    continue

                # Process search results and update cache
                candidates = []
                for page in search_results:
                    wiki_content = self._get_wiki_page(page.title, search_query)
                    if wiki_content:
                        candidates.append(wiki_content)
                        # Add to processed_docs if not already there
                        if wiki_content not in self.processed_docs:
                            self.processed_docs.append(wiki_content)

                # Filter out visited pages and pages already in context
                unvisited_candidates = [
                    doc
                    for doc in candidates
                    if doc[0] not in self.visited_pages
                    and doc[0] not in self.context_titles
                ]
                if not unvisited_candidates:
                    iteration += 1
                    continue

                # Rank unvisited documents
                ranked_docs = self._rank_documents(search_query, unvisited_candidates)
                if not ranked_docs:
                    iteration += 1
                    continue

                # Get the top score
                top_score = ranked_docs[0][2]
                score_threshold = top_score * relative_score_threshold

                # Take documents above the threshold
                docs_to_add = min(
                    docs_per_query, num_iterations - len(self.context_docs)
                )
                docs_added = 0

                for title, content, score in ranked_docs:
                    # Triple check: not visited, not in context, and meets score threshold
                    if (
                        title not in self.visited_pages
                        and title not in self.context_titles
                        and score >= score_threshold
                    ):
                        context_content = content[:max_tokens]  # Truncate to max_tokens
                        self.context_history.append(f"From {title}:\n{context_content}")
                        self.context_docs.append((title, context_content))
                        self.context_titles.add(title)  # Track title in context
                        self.visited_pages.add(title)  # Mark as visited
                        docs_added += 1

                        self.logger.info(
                            f"Added document: {title} (score: {score:.3f})"
                        )

                        # Break if we've added enough docs
                        if (
                            docs_added >= docs_to_add
                            or len(self.context_docs) >= num_iterations
                        ):
                            break

                iteration += 1

            self.logger.info(
                f"Retrieval complete. Collected {len(self.context_docs)} documents in {iteration} iterations"
            )
            self.logger.info(
                f"Cache statistics - Hits: {self.cache_hits}, Misses: {self.cache_misses}, Hit rate: {self.cache_hits/(self.cache_hits + self.cache_misses):.2%}"
            )

            # Generate final answer
            answer = self._generate_answer(question, self.context_history)

            return (
                answer,
                self.context_history,
                self.visited_pages,
                self.context_docs,
                self.processed_docs,
                generated_queries,
            )

        except Exception as e:
            self.logger.error(f"Error in retrieval process: {str(e)}")
            return "Error during retrieval", [], set(), [], [], []
