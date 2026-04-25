import logging
from src.query import query_gem

logger = logging.getLogger(__name__)

MAX_RECENT_TURNS = 3
SUMMARIZE_AFTER = 6


def build_history_context(chat_history: list[dict]) -> str:
    """
    Returns history as a formatted string for prompt injection.
    Summarizes older turns automatically to save context window space.
    """
    if not chat_history:
        return ""
    if len(chat_history) <= MAX_RECENT_TURNS:
        return _format_turns(chat_history)
    if len(chat_history) > SUMMARIZE_AFTER:
        older = chat_history[:-MAX_RECENT_TURNS]
        recent = chat_history[-MAX_RECENT_TURNS:]
        summary = _summarize_turns(older)
        return f"[Earlier summary]:\n{summary}\n\n[Recent turns]:\n{_format_turns(recent)}"
    return _format_turns(chat_history[-MAX_RECENT_TURNS:])


def _format_turns(turns: list[dict]) -> str:
    return "\n".join(f"User: {t['user']}\nBot: {t['bot']}" for t in turns)


def _summarize_turns(turns: list[dict]) -> str:
    prompt = f"""
Summarize this healthcare conversation in 3-4 sentences.
Focus on: symptoms mentioned, advice given, and follow-up topics.

Conversation:
{_format_turns(turns)}

Summary:
"""
    try:
        return query_gem(prompt).strip()
    except Exception as e:
        logger.warning(f"Summarization failed: {e}")
        return _format_turns(turns[-1:])