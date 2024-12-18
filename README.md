# Accuracy or Efficiency: Analyzing the Trade-offs in Performance when Using Predicted Hop Counts in Multi-Hop RAG

Retrieval-Augmented Generation (RAG) systems are an increasingly popular application for large language models. Multi-hop reasoning is an important class of tasks within RAG that requires reasoning over multiple documents to answer complex queries. Current RAG systems struggle with multi-hop reasoning tasks without massive amounts of context. As such, agents for completing multi-hop RAG tasks can become expensive, redundant, and inefficient at scale.  We introduce a metric - Normalized Accuracy Efficiency Index (NAEI) - by which we  an measure this trade-off. Our work shows that using the predicted hop count plus a certain amount of retrieval steps can lead to increased accuracy while minimizing retrieval costs. By analyzing performance changes as additional context is included, our work provides insight into scalable and cost-efficient implementations for RAG systems.

# Dataset: Frames

We use the Frames Dataset introduced in the "Fact, Fetch, and Reason" paper, which consists of multi-hop reason-ing questions requiring factual retrieval from structured knowledge sources like Wikipedia. These questions and their associated links and answers are organized in a CSV file. We imported the dataset by way of hugging face and pandas. The questions are organized into categories based on the type of reasoning required to answer them. 

![image](https://github.com/user-attachments/assets/8aac3a18-7840-44e9-8d09-403c1b59ebdf)


# Method
Our methodology consists of two primary parts. In the first part, we build a version of the BM-25 pipeline to measure how the different LLMs performed across topics and different pre-determined hop counts. In the second part, we start by first building a model to pre- dict the necessary hop count. We then incorporate this prediction into our pipeline to measure performance and NAEI scores.

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

# References
[1] Google. “Frames Benchmark: Multi-hop QA Dataset,” arXiv preprint, 2024. Available: https://arxiv.org/pdf/2409.12941 <br>
[2] Krishna et al. “Fact, Fetch, and Reason: A Unified Evaluation of Retrieval-Augmented Generation,” arXiv preprint, 2024. Available: https://arxiv.org/html/2409.12941 <br>
[3] Yu, Z., Liu, T., Wei, Z., et al. “RankRAG: Unifying Context Rank-ing with Retrieval-Augmented Generation in LLMs,” arXiv preprint,2023. Available: https://arxiv.org/abs/2407.02485 <br>
