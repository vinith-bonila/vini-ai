"""
Canonical intent catalog.

Each intent maps to a set of example utterances. The intent classifier learns
the decision boundary from these examples at runtime (TF-IDF or sentence
embeddings), so adding a new skill is as simple as adding an entry here plus a
handler in `assistant/router.py` - no if/elif editing anywhere.
"""
from __future__ import annotations

# Intent name -> representative example phrasings.
INTENT_EXAMPLES: dict[str, list[str]] = {
    "PLAY_MUSIC": [
        "play believer by imagine dragons",
        "play some music",
        "put on shape of you",
        "play kesariya on youtube",
        "can you play a song for me",
        "stream despacito",
    ],
    "WIKIPEDIA": [
        "who is albert einstein",
        "tell me about the eiffel tower",
        "search wikipedia for photosynthesis",
        "what is quantum computing",
        "give me a summary of the french revolution",
        "wikipedia machine learning",
    ],
    "WEB_SEARCH": [
        "google best laptops under 50000",
        "search the web for python tutorials",
        "look up how to make pasta",
        "find information about electric cars",
        "search for latest smartphone reviews",
    ],
    "YOUTUBE_SEARCH": [
        "find a video about neural networks",
        "search youtube for guitar lessons",
        "show me cooking videos",
        "youtube python full course",
    ],
    "TRANSLATE": [
        "translate hello to french",
        "how do you say thank you in spanish",
        "translate good morning into german",
        "what is water in japanese",
        "convert this sentence to hindi",
    ],
    "WEATHER": [
        "what is the weather today",
        "will it rain tomorrow",
        "how hot is it in mumbai",
        "weather forecast for delhi",
        "is it going to be sunny",
    ],
    "NEWS": [
        "what is in the news today",
        "give me the latest headlines",
        "any news about technology",
        "show me top stories",
        "latest news",
        "news update please",
        "tell me the news",
        "what's happening in the news",
    ],
    "TIME": [
        "what time is it",
        "tell me the current time",
        "what is the time right now",
        "give me the time",
    ],
    "DATE": [
        "what is today's date",
        "what day is it today",
        "tell me the date",
        "what is the date today",
    ],
    "CALCULATOR": [
        "what is 15 times 12",
        "calculate 250 plus 380",
        "how much is 45 divided by 9",
        "compute 2 to the power of 10",
        "what is 18 percent of 4500",
    ],
    "UNIT_CONVERSION": [
        "convert 10 kilometers to miles",
        "how many grams in 5 pounds",
        "convert 100 fahrenheit to celsius",
        "change 3 hours to minutes",
    ],
    "DICTIONARY": [
        "define serendipity",
        "what does ephemeral mean",
        "meaning of ubiquitous",
        "give me the definition of resilience",
    ],
    "JOKE": [
        "tell me a joke",
        "make me laugh",
        "say something funny",
        "do you know any jokes",
    ],
    "OPEN_APP": [
        "open notepad",
        "launch the calculator app",
        "open my file explorer",
        "start the browser",
    ],
    "GREETING": [
        "hello there",
        "hi vini",
        "hey good morning",
        "how are you doing",
    ],
    "GOODBYE": [
        "goodbye",
        "exit",
        "bye see you later",
        "stop the assistant",
        "shut down",
    ],
    "GENERAL_CHAT": [
        "what do you think about space travel",
        "tell me a story about a dragon",
        "help me write an email to my manager",
        "explain how a car engine works",
        "give me some ideas for a birthday gift",
        "why is the sky blue",
    ],
}

# Intents that must never leave the browser / require desktop control.
LOCAL_ONLY_INTENTS = {"OPEN_APP"}

# Intents handled by talking to the LLM rather than a deterministic skill.
LLM_INTENTS = {"GENERAL_CHAT"}

ALL_INTENTS = tuple(INTENT_EXAMPLES.keys())
