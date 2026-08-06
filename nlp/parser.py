"""POS tagging and dependency parsing via spaCy."""
from __future__ import annotations

from nlp._spacy_loader import get_nlp
from nlp.schemas import Token


def parse(text: str) -> list[Token]:
    """Return per-token linguistic features (lemma, POS, tag, dep, head)."""
    doc = get_nlp()(text or "")
    return [
        Token(
            text=t.text,
            lemma=t.lemma_,
            pos=t.pos_,
            tag=t.tag_,
            dep=t.dep_,
            head=t.head.text,
        )
        for t in doc
        if not t.is_space
    ]
