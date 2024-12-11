# Base Retriever Multi-LLM Testing

## Overview
This script provides a comprehensive testing framework for the Base Retriever, supporting multiple Language Models (LLMs).

## Prerequisites
- Python 3.8+
- Dependencies listed in project's `requirements.txt`
- API keys for desired LLMs (optional)

## Supported LLMs
- Gemini Pro
- GPT (OpenAI)

## Environment Setup
1. Install dependencies:
```bash
pip install -r ../requirements.txt
```

2. Set up API keys (optional):
- For Gemini: Set `GEMINI_API_KEY` environment variable
- For OpenAI: Set `OPENAI_API_KEY` environment variable

## Usage

### Basic Usage
```bash
python bm25_retriever_test.py
```

### Advanced Usage
```bash
# Test with specific models
python bm25_retriever_test.py --models gemini-pro gpt-3.5-turbo

# Limit number of test samples
python bm25_retriever_test.py --samples 50

# Adjust retrieval iterations
python bm25_retriever_test.py --iterations 5
```

### Parameters
- `--models`: List of LLM models to test (default: gemini-pro)
- `--samples`: Number of samples to test (default: all samples)
- `--iterations`: Number of retrieval iterations (default: 3)

## Output
- Prints a summary report to console
- Generates `base_retriever_test_results.csv` with detailed results

## Troubleshooting
- Ensure all dependencies are installed
- Check API key configuration
- Verify network connectivity

## Contributing
Contributions to expand LLM support are welcome! Please submit a pull request.
