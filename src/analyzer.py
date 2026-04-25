import json
import logging
from src.query import query_gem

logger = logging.getLogger(__name__)

_mix_instructions = {
    "hinglish": (
        "Write in Hinglish — mix Hindi and English like a young Indian would type on WhatsApp. "
        "Use Roman script ONLY (no Devanagari). "
        "Use Hindi for conversational/emotional parts, English for medical terms. "
        "Use 'aap' for a respectful tone. Sound warm and friendly."
    ),
}

_VALID_INTENTS = {"health", "casual", "irrelevant"}


# ── Translation ────────────────────────────────────────────────
def detect_and_translate(query: str, preferred_language: str = "english") -> dict:
    if preferred_language == "english" and _looks_like_english(query):
        return {"original_language": "english", "english_query": query, "is_english": True}

    prompt = f"""
Analyze this text and return a JSON object with exactly these fields:
{{
  "detected_language": "english" | "hindi" | "other",
  "english_translation": "translated text in English, or original if already English"
}}

Rules:
- detected_language: identify the primary language
- english_translation: translate naturally to English. If already English, return as-is.
- Return ONLY the JSON. No explanation, no markdown.

Text: "{query}"
"""
    try:
        result = _parse_json(query_gem(prompt))
        detected = str(result.get("detected_language", "english")).lower()
        translated = str(result.get("english_translation", query))
        if detected not in ("english", "hindi", "other"):
            detected = "english"
        if detected != "english":
            logger.info(f"Detected: {detected} | Translated: '{translated}'")
        return {
            "original_language": detected,
            "english_query": translated,
            "is_english": detected == "english"
        }
    except Exception as e:
        logger.warning(f"Language detection failed: {e} — assuming English")
        return {"original_language": "english", "english_query": query, "is_english": True}


def _looks_like_english(text: str) -> bool:
    ascii_count = sum(1 for c in text if ord(c) < 128)
    return (ascii_count / max(len(text), 1)) > 0.85


def translate_response(response: str, preferred_language: str) -> str:
    if preferred_language in ("english", "other", None):
        return response

    style_note = _mix_instructions.get(
        preferred_language,
        f"Translate fully to {preferred_language}. Keep medical terms in English."
    )
    prompt = f"""
You are translating a healthcare assistant's response.

Style: {style_note}

Rules:
- Use Roman script ONLY — no native script characters
- Keep medical terms, drug names, dosages in English
- Keep the same friendly tone
- Do not add or remove medical information
- Return ONLY the translated text, nothing else

Text:
{response}
"""
    try:
        translated = query_gem(prompt).strip()
        logger.info(f"Response translated to {preferred_language}")
        return translated
    except Exception as e:
        logger.warning(f"Translation failed: {e} — returning English")
        return response


def translate_bot_message(message: str, preferred_language: str) -> str:
    if preferred_language == "english":
        return message
    return translate_response(message, preferred_language)


# ── Query Analysis ─────────────────────────────────────────────
def analyze_query(query: str, user_profile: dict) -> dict:
    """Single API call: classify intent + extract symptoms + rewrite query."""
    profile_str = _format_profile(user_profile)
    prompt = f"""
Analyze this user query and return a JSON object with exactly these fields:
{{
  "intent": "health" | "casual" | "irrelevant",
  "symptoms": ["symptom1", "symptom2"] or [],
  "rewritten_query": "precise medical search query under 15 words"
}}

Rules:
- intent: "health" if medical/health topic, "casual" if small talk or meta questions, "irrelevant" otherwise
- symptoms: list of medical symptoms explicitly mentioned, empty list if none
- rewritten_query: rewrite into precise clinical terms for retrieval
- If the user asks for a diagnosis, still classify as "health" but rewritten_query should seek general information only

User Profile:
{profile_str}

User query: "{query}"

Return ONLY the JSON object. No explanation, no markdown.
"""
    try:
        result = _parse_json(query_gem(prompt))

        # Strict schema validation
        intent = str(result.get("intent", "irrelevant")).lower()
        if intent not in _VALID_INTENTS:
            logger.warning(f"Invalid intent '{intent}' — defaulting to irrelevant")
            intent = "irrelevant"

        symptoms = result.get("symptoms", [])
        if not isinstance(symptoms, list):
            symptoms = []
        symptoms = [str(s) for s in symptoms if s]

        rewritten = str(result.get("rewritten_query", query)).strip()
        if not rewritten:
            rewritten = query

        logger.info(f"Analysis → intent: {intent} | symptoms: {symptoms} | rewritten: '{rewritten}'")
        return {"intent": intent, "symptoms": symptoms, "rewritten_query": rewritten}

    except Exception as e:
        logger.error(f"Query analysis failed: {e} — using safe defaults")
        return {"intent": "health", "symptoms": [], "rewritten_query": query}


def _parse_json(raw: str) -> dict:
    """Strips markdown fences and parses JSON. Raises on failure."""
    cleaned = raw.strip().replace("```json", "").replace("```", "").strip()
    return json.loads(cleaned)


def _format_profile(profile: dict) -> str:
    if not profile:
        return "Not provided"
    parts = []
    if profile.get("age"):         parts.append(f"Age: {profile['age']}")
    if profile.get("gender"):      parts.append(f"Gender: {profile['gender']}")
    if profile.get("conditions"):  parts.append(f"Conditions: {', '.join(profile['conditions'])}")
    if profile.get("medications"): parts.append(f"Medications: {', '.join(profile['medications'])}")
    return ", ".join(parts) if parts else "Not provided"