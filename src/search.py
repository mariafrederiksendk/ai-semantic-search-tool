import logging
import warnings

import faiss
import nltk
from nltk.tokenize import sent_tokenize
from pypdf import PdfReader
from pypdf.errors import PdfReadError
from sentence_transformers import SentenceTransformer

warnings.filterwarnings("ignore")
logging.getLogger("transformers").setLevel(logging.ERROR)

TOP_K = 3

nltk.download("punkt_tab", quiet=True)
embedding_model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")


def read_file(path):
    if path.suffix.lower() == ".pdf":
        return " ".join(page.extract_text() for page in PdfReader(path).pages)
    return path.read_text(encoding="utf-8")


def build_index(path):
    try:
        text = read_file(path)
    except (PdfReadError, UnicodeDecodeError) as e:
        raise ValueError(f"Could not read {path.name}") from e

    sentences = sent_tokenize(" ".join(text.split()), language="english")
    if not sentences:
        raise ValueError(f"No text found in {path.name}")

    embeddings = embedding_model.encode(sentences)
    faiss.normalize_L2(embeddings)
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    return sentences, index


def search(query, documents, index):
    query_embedding = embedding_model.encode([query])
    faiss.normalize_L2(query_embedding)
    scores, ids = index.search(query_embedding, min(TOP_K, len(documents)))
    return [(documents[i], score) for i, score in zip(ids[0], scores[0])]
