# dspy-wikipedia-qa

> Minimal DSPy-ready Wikipedia Q&A with Milvus Lite retrieval.

### Screenshots

<img width="800" height="400" alt="image" src="https://github.com/user-attachments/assets/10fdda02-e0a1-46c6-a1c3-e73023a1dedc" />
<img width="800" height="600" alt="image" src="https://github.com/user-attachments/assets/dd9c7145-1c0f-4e19-851f-2b9afce631ad" />

It's easy to use: First, get the article from Wikipedia you'd like to search. Second, index the page and embed it into the vector database (Milvus Lite).
Finally, you ask a question, and the related articles will be fetched from the database, and the LLM will generate the response for you based on the fetched articles.
<img width="800" height="600" alt="image" src="https://github.com/user-attachments/assets/a3da5f37-b516-45c2-a67b-89c759d59711" />

### Installation

Create a Python virtual environment and install dependencies:
Those are listed in `requirements.txt`. And, you need an OpenAI API key to
utilize embedding models and LLMs. Get your own API key and set it as an
environment variable: `OPENAI_API_KEY`.

### Example usages

```sh
# Search candidates on Wikipedia (REST search)
python main.py search "alan turing" --k 5

# Fetch a page by exact title (bypass search)
python main.py get "Python_(programming_language)" --lang en

# Search → pick → hydrate page content
python main.py fetch "alan turing" --k 5 --pick 1

# Index a page into Milvus Lite (./milvus.db) with section-aware chunking
python main.py index_title "Alan Turing" --lang en

# Ask a question answered from your indexed chunks (requires OPENAI_API_KEY)
python main.py ask "Which award did Alan Turing receive during his lifetime?" --k 6

# List collections
python main.py db collections

# Peek sample rows (hide embedding; truncate long text)
python main.py db peek --limit 5

# Collection stats (best-effort schema/row count)
python main.py db stats

# Page titles with chunk counts
python main.py db titles

# Evaluate LLM; YAML input, compact table
python main.py eval run eval.yaml --view table

# Evaluate LLM; YAML input, wide table view with a compact able
python3 main.py eval run eval.yaml --threads 2 --view table

# Evaluate LLM; JSON output (for scripting) with 1% numeric tolerance
python main.py eval run eval.yaml --view json --tolerance-pct 1.0 --save-json eval_results.json
```

You can get a detailed command help at `python main.py <command> --help`.
