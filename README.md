# AI Semantic Search Tool

![alt text](image.png)

A prototype search tool that finds text based on meaning, not just keywords.
You enter a keyword, it finds the most relevant passages using an AI model, and a second AI model gives you a short summary of those passages.

---

## How the AI Works

Here's how it works under the hood:

### 1. Embeddings (Sentence-BERT)

- Both the text passages and user queries are encoded into vectors using Sentence-BERT.
- These embeddings capture semantic meaning, so that even differently worded texts can be recognized as similar.

### 2. Vector Indexing (FAISS)

- All embeddings are stored in a FAISS index that enables fast and scalable similarity search.
- When a query is submitted, its embedding is compared to the stored ones to find the nearest matches.

### 3. Similarity Scores (cosine similarity)

- Each result includes a **similarity score**, calculated using **cosine similarity** between the query and each passage.
- A score close to **1.0** means high semantic overlap; a score closer to **0.0** means weak or no similarity.
- For example, `(score: 0.89)` means the model finds that result highly relevant to the query's meaning.

## Tech Stack

- **Python 3.11+**
- **Sentence-BERT** (all-mpnet-base-v2 model) for text embeddings
- **FAISS** (IndexFlatIP) for fast vector similarity search
- **BART** (facebook/bart-large-cnn) for AI summary generation
