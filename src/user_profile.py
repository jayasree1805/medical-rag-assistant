import logging

logger = logging.getLogger(__name__)

SUPPORTED_LANGUAGES = {
    "1": "english",
    "2": "hinglish",
}

GREETINGS = {
    "english": (
        "Hi! I'm Sakhii, your personal health assistant\n"
        "   created by Sakhii Care Foundation.\n"
        "   I can help with symptoms, illnesses,\n"
        "   medications, and general health advice."
    ),
    "hinglish": (
        "Namaste! Main hoon Sakhii, aapki personal health assistant,\n"
        "   jo banaya hai Sakhii Care Foundation ne.\n"
        "   Main symptoms, illnesses, medications,\n"
        "   aur general health advice mein help kar sakti hoon."
    ),
}

PROFILE_PROMPTS = {
    "english": {
        "intro": "Before we start, I'd like to know a bit about you\nso I can give personalized advice.\n(All fields optional — type 'skip' or 'none')\n",
        "age": "Sakhii: How old are you? ",
        "gender": "Sakhii: What's your gender? ",
        "conditions": "Sakhii: Any existing medical conditions? (e.g. diabetes / none) ",
        "medications": "Sakhii: Any current medications? (or none) ",
        "name": "Sakhii: What should I call you? ",
    },
    "hinglish": {
        "intro": "Shuru karne se pehle, main aapke baare mein thoda jaanna chahti hoon\ntaaki main personalized advice de sakoon.\n(Sab optional hai — 'skip' ya 'none' type karo)\n",
        "age": "Sakhii: Aapki age kya hai? ",
        "gender": "Sakhii: Aapka gender kya hai? ",
        "conditions": "Sakhii: Koi existing medical conditions hain? (e.g. diabetes / none) ",
        "medications": "Sakhii: Aap koi current medications le rahe hain? (or none) ",
        "name": "Sakhii: Main aapko kya bulaaon? ",
    },
}


def collect_profile_interactively() -> dict:
    print("\n🌐 Choose your preferred language:\n")
    print("  1. English")
    print("  2. Hinglish (Hindi + English mix)\n")

    while True:
        choice = input("Enter number (1-2): ").strip()
        if choice in SUPPORTED_LANGUAGES:
            preferred_language = SUPPORTED_LANGUAGES[choice]
            break
        print("Please enter 1 or 2.")

    print(f"\n✅ Language set to: {preferred_language.capitalize()}\n")

    print("━" * 45)
    print(GREETINGS[preferred_language])
    print("━" * 45)

    prompts = PROFILE_PROMPTS[preferred_language]
    print(f"\n{prompts['intro']}")

    profile = {"preferred_language": preferred_language}

    age_input = input(prompts["age"]).strip()
    if age_input.lower() not in ("skip", ""):
        try:
            profile["age"] = int(age_input)
        except ValueError:
            profile["age"] = age_input

    gender_input = input(prompts["gender"]).strip()
    if gender_input.lower() not in ("skip", ""):
        profile["gender"] = gender_input

    conditions_input = input(prompts["conditions"]).strip()
    if conditions_input.lower() not in ("none", "skip", ""):
        profile["conditions"] = [c.strip() for c in conditions_input.split(",")]

    medications_input = input(prompts["medications"]).strip()
    if medications_input.lower() not in ("none", "skip", ""):
        profile["medications"] = [m.strip() for m in medications_input.split(",")]

    name_input = input(prompts["name"]).strip()
    if name_input.lower() not in ("skip", ""):
        profile["name"] = name_input

    print()
    return profile


def format_profile_for_prompt(profile: dict) -> str:
    if not profile:
        return "No user profile available."

    lines = ["User Profile:"]
    if profile.get("name"):
        lines.append(f"  - Name: {profile['name']}")
    if profile.get("age"):
        lines.append(f"  - Age: {profile['age']}")
    if profile.get("gender"):
        lines.append(f"  - Gender: {profile['gender']}")
    if profile.get("conditions"):
        lines.append(f"  - Existing conditions: {', '.join(profile['conditions'])}")
    if profile.get("medications"):
        lines.append(f"  - Current medications: {', '.join(profile['medications'])}")
    if profile.get("preferred_language"):
        lines.append(f"  - Preferred language: {profile['preferred_language']}")

    return "\n".join(lines)