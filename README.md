# dspy-wikipedia-qa

> Minimal DSPy-ready Wikipedia Q&A with Milvus Lite retrieval.

### Installation

Create a Python virtual environment and install dependencies:
that are listed in `requirements.txt`.

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
```

You can get a detailed command help at `python main.py <command> --help`.
