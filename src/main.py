import logging
import warnings
from pathlib import Path

import nltk
from nltk.tokenize import sent_tokenize
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer, util
from transformers import pipeline

warnings.filterwarnings("ignore")
logging.getLogger("transformers").setLevel(logging.ERROR)

DATA_DIR = "./data"
TOP_K = 3


def load_documents(data_dir):
    texts = []
    for path in sorted(Path(data_dir).iterdir()):
        if path.suffix == ".pdf":
            texts.append(" ".join(page.extract_text() for page in PdfReader(path).pages))
        elif path.suffix in {".txt", ".md"}:
            texts.append(path.read_text(encoding="utf-8"))
    return " ".join(" ".join(texts).split())


def generate_summary(summary_model, query, passages):
    text = f"Question: {query}. {' '.join(passages)}"
    result = summary_model(text, max_length=100, min_length=20, do_sample=False, truncation=True)
    return result[0]["summary_text"]


def main():
    nltk.download("punkt", quiet=True)
    nltk.download("punkt_tab", quiet=True)

    documents = sent_tokenize(load_documents(DATA_DIR), language="english")

    embedding_model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")
    doc_embeddings = embedding_model.encode(documents)
    summary_model = pipeline("summarization", model="facebook/bart-large-cnn")

    print("\nSemantic Search & Summary Tool")
    print("=" * 50)

    while True:
        try:
            query = input("\nEnter query (or 'quit' to exit): ")
            if query.lower() == "quit":
                break
            if not query.strip():
                continue

            hits = util.semantic_search(embedding_model.encode(query), doc_embeddings, top_k=TOP_K)[0]
            passages = [documents[hit["corpus_id"]] for hit in hits]

            print("\nTop results:")
            for passage, hit in zip(passages, hits):
                print(f"- {passage[:100]}{'...' if len(passage) > 100 else ''} (score: {hit['score']:.2f})")

            print(f"\nSummary: {generate_summary(summary_model, query, passages)}")

        except (KeyboardInterrupt, EOFError):
            break
        except Exception as e:
            print(f"Error during search: {e}")


if __name__ == "__main__":
    main()
