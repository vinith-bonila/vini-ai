"""
Quick command-line demo of the VINI AI NLU + routing stack (no UI, no audio).

    python demo.py

Runs a set of example utterances through the full pipeline and prints the
intent, entities and response for each. Useful as a smoke test.
"""
from __future__ import annotations

from assistant.conversation import Conversation
from assistant.dispatcher import handle

EXAMPLES = [
    "hello",
    "play Believer by Imagine Dragons",
    "translate good morning to french",
    "what is 18 percent of 4500",
    "convert 10 km to miles",
    "who is Alan Turing",
    "what's the weather in Mumbai",
    "define serendipity",
    "tell me a joke",
    "what time is it",
    "help me write a leave email to my manager",
    "goodbye",
]


def main() -> None:
    conv = Conversation()
    for text in EXAMPLES:
        result = handle(text, conv)
        ents = ", ".join(f"{e.text}:{e.label}" for e in result.nlu.entities) or "-"
        print(f"\nYou: {text}")
        print(f"  intent     : {result.nlu.intent} ({result.nlu.confidence*100:.0f}%)")
        print(f"  entities   : {ents}")
        print(f"  VINI AI    : {result.response.speech}")


if __name__ == "__main__":
    main()
