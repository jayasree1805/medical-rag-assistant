import json
import logging
import numpy as np
from dotenv import load_dotenv
load_dotenv()

from sentence_transformers import SentenceTransformer
from src.retrieval import load_data, build_bm25_index, hybrid_retrieve, rerank_chunks
from src.analyzer import analyze_query
from src.query import query_gem
from src.prompt import build_prompt

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger(__name__)

EVAL_FILE = "data/eval.json"
_embedder = SentenceTransformer("abhinand/MedEmbed-small-v0.1")


def semantic_score(answer: str, expected_keywords: list[str]) -> float:
    """
    Embedding-based semantic similarity between answer
    and expected keywords joined as a reference sentence.
    More robust than exact keyword matching.
    """
    reference = " ".join(expected_keywords)
    vecs = _embedder.encode([answer, reference])
    a = vecs[0] / (np.linalg.norm(vecs[0]) + 1e-10)
    b = vecs[1] / (np.linalg.norm(vecs[1]) + 1e-10)
    return float(np.dot(a, b))


def run_eval():
    try:
        with open(EVAL_FILE, "r") as f:
            eval_set = json.load(f)
    except FileNotFoundError:
        print(f"Eval file not found at {EVAL_FILE}.")
        return

    print("Loading knowledge base...")
    chunks = load_data("data/medical_kb.txt")
    build_bm25_index(chunks)
    print(f"Loaded {len(chunks)} chunks.\n")

    total = len(eval_set)
    keyword_passed = 0
    semantic_passed = 0
    total_keyword_score = 0.0
    total_semantic_score = 0.0
    results = []

    for i, item in enumerate(eval_set):
        query = item["query"]
        expected = [kw.lower() for kw in item["expected_keywords"]]
        print(f"[{i+1}/{total}] Query: {query}")

        try:
            analysis = analyze_query(query, {})
            intent = analysis["intent"]
            symptoms = analysis["symptoms"]
            rewritten = analysis["rewritten_query"]

            retrieved, mode = hybrid_retrieve(rewritten, n_results=5)
            sources = []
            if mode == "chunks" and retrieved:
                retrieved, sources = rerank_chunks(rewritten, retrieved, top_n=3)

            prompt = build_prompt(
                query=query,
                retrieved_chunks=retrieved,
                mode=mode,
                user_profile={},
                history_context="",
                symptoms=symptoms,
                sources=sources
            )

            answer = query_gem(prompt)
            answer_lower = answer.lower()

            # Keyword score
            matched = [kw for kw in expected if kw in answer_lower]
            missing = [kw for kw in expected if kw not in answer_lower]
            kw_score = len(matched) / len(expected)
            total_keyword_score += kw_score
            kw_pass = kw_score >= 0.5
            keyword_passed += 1 if kw_pass else 0

            # Semantic score
            sem_score = semantic_score(answer_lower, expected)
            total_semantic_score += sem_score
            sem_pass = sem_score >= 0.5
            semantic_passed += 1 if sem_pass else 0

            status = "✅" if kw_pass and sem_pass else "⚠️" if kw_pass or sem_pass else "❌"
            print(f"  {status} KW: {len(matched)}/{len(expected)} ({kw_score*100:.0f}%) | "
                  f"Semantic: {sem_score*100:.0f}%")
            print(f"  Matched : {matched}")
            print(f"  Missing : {missing}")
            print(f"  Intent  : {intent} | Mode: {mode} | Rewritten: '{rewritten}'\n")

            results.append({
                "query": query,
                "keyword_score": round(kw_score, 2),
                "semantic_score": round(sem_score, 2),
                "status": status,
                "matched": matched,
                "missing": missing,
                "intent": intent,
                "mode": mode,
                "rewritten_query": rewritten,
            })

        except Exception as e:
            logger.error(f"Error on '{query}': {e}")
            print(f"  ⚠️  ERROR: {e}\n")
            results.append({"query": query, "status": "error", "error": str(e)})

    avg_kw = total_keyword_score / total if total > 0 else 0
    avg_sem = total_semantic_score / total if total > 0 else 0

    print("=" * 60)
    print("EVAL COMPLETE")
    print(f"  Keyword  — passed: {keyword_passed}/{total} | avg: {avg_kw*100:.1f}%")
    print(f"  Semantic — passed: {semantic_passed}/{total} | avg: {avg_sem*100:.1f}%")
    print("=" * 60)

    with open("data/eval_results.json", "w") as f:
        json.dump({
            "summary": {
                "total": total,
                "keyword_passed": keyword_passed,
                "semantic_passed": semantic_passed,
                "avg_keyword_score": round(avg_kw * 100, 1),
                "avg_semantic_score": round(avg_sem * 100, 1),
            },
            "results": results
        }, f, indent=2)

    print("Saved to data/eval_results.json")


if __name__ == "__main__":
    run_eval()