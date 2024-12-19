# Accuracy or Efficiency: Analyzing the Trade-offs in Performance when Using Predicted Hop Counts in Multi-Hop RAG

Retrieval-Augmented Generation (RAG) systems are an increasingly popular application for large language models. Multi-hop reasoning is an important class of tasks within RAG that requires reasoning over multiple documents to answer complex queries. Current RAG systems struggle with multi-hop reasoning tasks without massive amounts of context. As such, agents for completing multi-hop RAG tasks can become expensive, redundant, and inefficient at scale.  We introduce a metric - Accuracy Efficiency Index (AEI) - by which we measure this trade-off. Our work shows that using the predicted hop count plus a certain amount of retrieval steps can lead to increased accuracy while minimizing retrieval costs. By analyzing performance changes as additional context is included, our work provides insight into scalable and cost-efficient implementations for RAG systems.

# Dataset: Frames

We use the Frames Dataset introduced in the "Fact, Fetch, and Reason" paper, which consists of multi-hop reason-ing questions requiring factual retrieval from structured knowledge sources like Wikipedia. These questions and their associated links and answers are organized in a CSV file. We imported the dataset by way of hugging face and pandas. The questions are organized into categories based on the type of reasoning required to answer them. 

![image](https://github.com/user-attachments/assets/8aac3a18-7840-44e9-8d09-403c1b59ebdf)


# Method
Our methodology consists of two primary parts. In the first part, we build a version of the BM-25 pipeline to measure how the different LLMs performed across topics and different pre-determined hop counts. In the second part, we start by first building a model to pre- dict the necessary hop count. We then incorporate this prediction into our pipeline to measure performance and AEI scores.

## Modified BM-25 pipeline
![image](https://github.com/user-attachments/assets/646c2b51-767f-43fa-8544-02c59c08c991)

## Predictor Model
![image](https://github.com/user-attachments/assets/bc49cb0c-72cf-4179-964f-d6e79c8faafc)

# Results

![image](https://github.com/user-attachments/assets/b8064772-44ef-49bb-b1e8-c5d121b13f82)

![image](https://github.com/user-attachments/assets/5b59ae82-39bf-41a4-a3b6-25047299ae73)

# Environment setup
First install the requirements needed with:
```
pip install -r requirements.txt
```
To run the pipeline with Gemini set the environment variable GEMINI_API_KEY.
On Windows this is:
```
set GEMINI_API_KEY=<your api key>
```
On Linux use:
```
export GEMINI_API_KEY=<your api key>
```
Similarly you can run LLAMA and GPT by setting the environment variables HUGGING_FACE_API_KEY and OPENAI_API_KEY respectively.

# Running the Naive Method
Run the following command from the root directory:
```
python naive/run_evaluation model=<gemini, gpt or llama>
```

# Base retriever
```
python base_retriever_test/bm25_retriever_test.py --models=<gemini-pro> --samples=<number of samples to test> --similarity-threshold=<Similarity threshold for answer accuracy>
```

# Retriever with predictions
This is the general command for running the retriever using our predictive model for the number of iterations, or retrieval depth. Simply modify increment-amount to whatever value is desired (0 is p, 1 is p + 1, etc.).
```
python test_updated.py --batch-size 10 --batch-num 2 --increment-amount 0
```
Note that the current flow for the predictive retriever tests based on one reasoning type at a time. Our dataset.py, located in the src directory, offers multiple options of what data we want to test. For the current configuration, the reasoning type is controlled by altering the string at line 131:
```
        self.dataset = (
            filter_by_reasoning_type(
                get_condensed_frames_dataset(), "Tabular reasoning"
            )
```
Simply change "Tabular reasoning" to whatever the desired reasoning type is.

# References
[1] Google. “Frames Benchmark: Multi-hop QA Dataset,” arXiv preprint, 2024. Available: https://arxiv.org/pdf/2409.12941 <br>
[2] Krishna et al. “Fact, Fetch, and Reason: A Unified Evaluation of Retrieval-Augmented Generation,” arXiv preprint, 2024. Available: https://arxiv.org/html/2409.12941 <br>
[3] Yu, Z., Liu, T., Wei, Z., et al. “RankRAG: Unifying Context Ranking with Retrieval-Augmented Generation in LLMs,” arXiv preprint, 2023. Available: https://arxiv.org/abs/2407.02485 <br>
[4] Stephen E. Robertson, Steve Walker, Susan Jones, Micheline Hancock-Beaulieu, and Mike Gatford. “Okapi at TREC-3,” Proceedings of the Third Text REtrieval Conference (TREC 1994), November 1994. Gaithersburg, USA. <br>
[5] MediaWiki. “API:Opensearch,” 2024. Available: https://www.mediawiki.org/wiki/API:Opensearch <br>
[6] LangChain. “Planning Agents,” 2024. Available: https://blog.langchain.dev/planning-agents/ <br>
[7] LangChain. “Agent Supervisor: Create Tools,” 2024. Available: https://langchain-ai.github.io/langgraph/tutorials/multi_agent/agent_supervisor/#create-tools <br>
[8] J. Briggs. “LangGraph Research Agent,” 2024. Available: https://www.pinecone.io/learn/langgraph-research-agent/ <br>
[9] Tom B. Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, Sandhini Agarwal, Ariel Herbert-Voss, Gretchen Krueger, Tom Henighan, Rewon Child, Aditya Ramesh, Daniel M. Ziegler, Jeffrey Wu, Clemens Winter, Christopher Hesse, Mark Chen, Eric Sigler, Mateusz Litwin, Scott Gray, Benjamin Chess, Jack Clark, Christopher Berner, Sam McCandlish, Alec Radford, Ilya Sutskever, and Dario Amodei. “Language Models are Few-Shot Learners,” arXiv preprint, 2020. Available: https://doi.org/10.48550/arXiv.2005.14165 <br>
[10] L. Lacy. “GPT-4.0 and Gemini 1.5 Pro: How the New AI Models Compare,” CNET, May 25, 2024. Available: https://www.cnet.com/tech/services-and-software/gpt-4o-and-gemini-1-5-pro-how-the-new-ai-models-compare/ <br>
