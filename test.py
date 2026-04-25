import logging
from dotenv import load_dotenv
load_dotenv()

from src.retrieval import load_data, build_bm25_index, hybrid_retrieve, rerank_chunks
from src.prompt import build_prompt
from src.memory import build_history_context
from src.analyzer import analyze_query, detect_and_translate, translate_response, translate_bot_message
from src.query import query_gem
from src.user_profile import collect_profile_interactively
from src.emergency import check_emergency, EMERGENCY_RESPONSE

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger(__name__)

chat_history = []
user_profile = {}


def initialize():
    logger.info("Loading knowledge base and building BM25 index...")
    chunks = load_data("data/medical_kb.txt")
    build_bm25_index(chunks)
    logger.info("Ready.")


def rag_answer(query: str) -> str:
    preferred_lang = user_profile.get("preferred_language", "english")

    # Step 0 — Emergency check (embedding-based, no API call)
    if check_emergency(query):
        logger.warning(f"Emergency detected: '{query}'")
        chat_history.append({"user": query, "bot": EMERGENCY_RESPONSE})
        return EMERGENCY_RESPONSE

    # Step 1 — Language detection + translation (skipped if English)
    lang_info = detect_and_translate(query, preferred_language=preferred_lang)
    english_query = lang_info["english_query"]

    # Step 2 — Analyze query (1 API call)
    analysis = analyze_query(english_query, user_profile)
    intent = analysis["intent"]
    symptoms = analysis["symptoms"]
    rewritten = analysis["rewritten_query"]

    # Step 3 — Build history context (local)
    history_context = build_history_context(chat_history)

    try:
        if intent == "casual":
            name = user_profile.get("name", "")
            greeting = f"Hi {name}! " if name else "Hi! "
            answer = (
                f"{greeting}I'm Sakhii, your personal health assistant "
                f"created by Sakhii Care Foundation. "
                f"I can help with symptoms, illnesses, medications, "
                f"and general health advice. What's on your mind?"
            )
            answer = translate_bot_message(answer, preferred_lang)

        elif intent == "irrelevant":
            answer = "I can only answer health related questions."
            answer = translate_bot_message(answer, preferred_lang)

        else:  # health
            # Step 4 — Hybrid retrieval (local, uses precomputed embeddings)
            retrieved, mode = hybrid_retrieve(rewritten, n_results=5)
            logger.info(f"Mode: {mode} | Retrieved: {len(retrieved)} chunks")

            sources = []

            # Step 5 — Rerank using cached embeddings, get sources
            if mode == "chunks" and retrieved:
                retrieved, sources = rerank_chunks(rewritten, retrieved, top_n=3)
                logger.info(f"After reranking: {len(retrieved)} chunks | Sources: {sources}")

            # Step 6 — Build prompt with sources
            prompt = build_prompt(
                query=english_query,
                retrieved_chunks=retrieved,
                mode=mode,
                user_profile=user_profile,
                history_context=history_context,
                symptoms=symptoms,
                sources=sources
            )

            # Step 7 — Generate answer (1 API call)
            answer = query_gem(prompt)

            # Step 8 — Translate if needed (1 API call, non-English only)
            answer = translate_response(answer, preferred_lang)

    except Exception as e:
        logger.error(f"Pipeline error: {e}")
        answer = "Sorry, something went wrong. Please try again."
        answer = translate_bot_message(answer, preferred_lang)

    chat_history.append({"user": query, "bot": answer})
    return answer


if __name__ == "__main__":
    print("\n╔══════════════════════════════════════════════╗")
    print("║         Sakhii Care Foundation               ║")
    print("║   Your Personal Health Assistant — Sakhii   ║")
    print("╚══════════════════════════════════════════════╝")

    initialize()
    user_profile = collect_profile_interactively()

    name = user_profile.get("name", "")
    lang = user_profile.get("preferred_language", "english").capitalize()
    print(f"Sakhii: All set{', ' + name if name else ''}! Chatting in {lang}.")
    print("Type 'exit' to quit.\n")

    while True:
        q = input("You: ").strip()
        if not q:
            continue
        if q.lower() in ("exit", "quit"):
            print("Sakhii: Take care! Goodbye! 👋")
            break
        response = rag_answer(q)
        print(f"\nSakhii: {response}\n")