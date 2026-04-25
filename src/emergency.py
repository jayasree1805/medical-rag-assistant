import logging
import numpy as np
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

# Use same embedder as retrieval for consistency

# Embedding model: MedEmbed by Abhinand Balachandran
# Citation: https://github.com/abhinand5/MedEmbed
# @software{balachandran2024medembed, author={Balachandran, Abhinand},
#   title={MedEmbed}, year={2024}, url={https://github.com/abhinand5/MedEmbed}}

embedder = SentenceTransformer("abhinand/MedEmbed-small-v0.1")
_embedder = SentenceTransformer("abhinand/MedEmbed-small-v0.1")

# Representative emergency phrases — one per category
# The embedder generalises to paraphrases automatically
EMERGENCY_PHRASES = [
    "I have severe chest pain or tightness",
    "I cannot breathe or am struggling to breathe",
    "I think I am having a stroke or seizure",
    "I want to kill myself or end my life",
    "I took an overdose of medication or poison",
    "I am having a severe allergic reaction",
    "I have severe uncontrolled bleeding",
    "Someone is unconscious and not responding",
]

EMERGENCY_THRESHOLD = 0.72   # cosine similarity threshold

# Precompute emergency embeddings once at module load
_emergency_embeddings = _embedder.encode(EMERGENCY_PHRASES)

EMERGENCY_RESPONSE = """⚠️ This sounds like a medical emergency.

Please take immediate action:
- Call emergency services: 112 (India) / 911 (US)
- Go to the nearest emergency room immediately
- Do not wait or self-medicate

I'm a healthcare assistant and cannot handle emergencies.
Please contact medical professionals immediately."""


def check_emergency(query: str) -> bool:
    """
    Embedding-based emergency detection.
    Catches paraphrases, Hinglish inputs, and spelling variations
    that keyword matching would miss.
    """
    query_embedding = _embedder.encode(query)
    scores = _cosine_similarity(query_embedding, _emergency_embeddings)
    max_score = max(scores)
    best_match = EMERGENCY_PHRASES[scores.index(max_score)]

    if max_score >= EMERGENCY_THRESHOLD:
        logger.warning(
            f"Emergency detected (score={max_score:.3f}) "
            f"matched: '{best_match}' | query: '{query}'"
        )
        return True
    return False


def _cosine_similarity(query_vec: np.ndarray, phrase_vecs: np.ndarray) -> list[float]:
    query_norm = query_vec / (np.linalg.norm(query_vec) + 1e-10)
    phrase_norms = phrase_vecs / (np.linalg.norm(phrase_vecs, axis=1, keepdims=True) + 1e-10)
    return (phrase_norms @ query_norm).tolist()