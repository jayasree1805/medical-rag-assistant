import logging
import numpy as np
import chromadb
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi

logger = logging.getLogger(__name__)

# ── Models & DB ────────────────────────────────────────────────

# Embedding model: MedEmbed by Abhinand Balachandran
# Citation: https://github.com/abhinand5/MedEmbed
# @software{balachandran2024medembed, author={Balachandran, Abhinand},
#   title={MedEmbed}, year={2024}, url={https://github.com/abhinand5/MedEmbed}}

embedder = SentenceTransformer("abhinand/MedEmbed-small-v0.1")
embedder = SentenceTransformer("abhinand/MedEmbed-small-v0.1")

CHROMA_PATH = "data/chroma_db"
DISTANCE_THRESHOLD = 1.2

_chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
collection = _chroma_client.get_or_create_collection("medical_docs")

# ── State ──────────────────────────────────────────────────────
_bm25_index = None
_loaded_chunks = []
_chunk_embeddings = None    # precomputed once at startup


# ── Data loading ───────────────────────────────────────────────
def load_data(file_path: str) -> list[str]:
    global _loaded_chunks, _chunk_embeddings
    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()
    _loaded_chunks = [c.strip() for c in text.split("---") if len(c.strip()) > 50]
    logger.info(f"Loaded {len(_loaded_chunks)} chunks. Precomputing embeddings...")
    _chunk_embeddings = embedder.encode(_loaded_chunks, show_progress_bar=True)
    logger.info("Chunk embeddings precomputed and cached.")
    return _loaded_chunks


def get_loaded_chunks() -> list[str]:
    return _loaded_chunks


# ── Indexing ───────────────────────────────────────────────────
def build_bm25_index(chunks: list[str]):
    global _bm25_index
    tokenized = [chunk.lower().split() for chunk in chunks]
    _bm25_index = BM25Okapi(tokenized)
    logger.info(f"BM25 index built with {len(chunks)} chunks.")


def store_embeddings(chunks: list[str]):
    for i, chunk in enumerate(chunks):
        embedding = embedder.encode(chunk).tolist()
        collection.upsert(
            ids=[str(i)],
            documents=[chunk],
            embeddings=[embedding]
        )
    logger.info(f"Upserted {len(chunks)} embeddings.")
    print("Stored all embeddings!")


# ── Retrieval ──────────────────────────────────────────────────
def hybrid_retrieve(query: str, n_results: int = 5) -> tuple[list[str], str]:
    """
    BM25 + semantic hybrid search using precomputed embeddings.
    Uses dynamic confidence threshold based on score distribution.
    Returns (chunks, mode).
    """
    if not _loaded_chunks or _chunk_embeddings is None:
        return _chroma_retrieve(query, n_results)

    query_embedding = embedder.encode(query)
    semantic_scores = _cosine_similarity(query_embedding, _chunk_embeddings)

    if _bm25_index is not None:
        bm25_raw = _bm25_index.get_scores(query.lower().split())
        bm25_max = max(bm25_raw) if max(bm25_raw) > 0 else 1
        bm25_scores = [s / bm25_max for s in bm25_raw]
    else:
        bm25_scores = [0.0] * len(_loaded_chunks)

    combined = [
        (0.6 * sem + 0.4 * bm25, i, chunk)
        for i, (sem, bm25, chunk) in enumerate(
            zip(semantic_scores, bm25_scores, _loaded_chunks)
        )
    ]
    combined.sort(key=lambda x: x[0], reverse=True)

    top_n = combined[:n_results]
    top_scores = [score for score, _, _ in top_n]

    # Dynamic threshold: mean of top scores * 0.6
    dynamic_threshold = np.mean(top_scores) * 0.6
    logger.info(f"Dynamic threshold: {dynamic_threshold:.3f} | Top score: {top_scores[0]:.3f}")

    filtered = [(score, i, chunk) for score, i, chunk in top_n if score >= dynamic_threshold]

    if not filtered:
        logger.info("No chunks passed dynamic threshold — falling back to generic.")
        return [], "generic"

    chunks_out = [chunk for _, _, chunk in filtered]
    indices_out = [i for _, i, _ in filtered]

    logger.info(f"Hybrid retrieval: {len(chunks_out)} chunks passed threshold")
    return chunks_out, "chunks"


def rerank_chunks(
    query: str,
    chunks: list[str],
    top_n: int = 3,
    chunk_indices: list[int] = None
) -> tuple[list[str], list[str]]:
    """
    Reranks chunks by cosine similarity.
    Uses precomputed embeddings via chunk_indices if provided — avoids recomputing.
    Returns (reranked_chunks, source_snippets).
    """
    if not chunks or len(chunks) <= top_n:
        sources = [_make_source(c) for c in chunks]
        return chunks, sources

    if chunk_indices is not None and _chunk_embeddings is not None:
        # Reuse precomputed embeddings — no recompute
        chunk_vecs = np.array([_chunk_embeddings[i] for i in chunk_indices])
    else:
        chunk_vecs = embedder.encode(chunks)

    query_embedding = embedder.encode(query)
    scores = _cosine_similarity(query_embedding, chunk_vecs)

    ranked = sorted(zip(scores, chunks), key=lambda x: x[0], reverse=True)
    top_chunks = [chunk for _, chunk in ranked[:top_n]]
    sources = [_make_source(chunk) for chunk in top_chunks]

    logger.info(f"Reranked {len(chunks)} → kept top {len(top_chunks)}")
    return top_chunks, sources


def _make_source(chunk: str) -> str:
    """Extracts the question line from a Q:A chunk as a source label."""
    lines = chunk.strip().split("\n")
    for line in lines:
        if line.startswith("Q:"):
            return line[2:].strip()
    return chunk[:80].strip() + "..."


def _chroma_retrieve(query: str, n_chunks: int) -> tuple[list[str], str]:
    query_embedding = embedder.encode(query).tolist()
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_chunks,
        include=["documents", "distances"]
    )
    docs = results["documents"][0]
    distances = results["distances"][0]
    filtered = [d for d, dist in zip(docs, distances) if dist < DISTANCE_THRESHOLD]
    if not filtered:
        return [], "generic"
    return filtered, "chunks"


def _cosine_similarity(query_vec: np.ndarray, chunk_vecs: np.ndarray) -> list[float]:
    query_norm = query_vec / (np.linalg.norm(query_vec) + 1e-10)
    chunk_norms = chunk_vecs / (np.linalg.norm(chunk_vecs, axis=1, keepdims=True) + 1e-10)
    return (chunk_norms @ query_norm).tolist()


if __name__ == "__main__":
    chunks = load_data("data/medical_kb.txt")
    store_embeddings(chunks)