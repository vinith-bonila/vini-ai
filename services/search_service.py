"""
Live web answering (retrieval-augmented).

For questions that need current, real-world facts ("who is the current PM of
India", "latest ... ", "price of ...") the assistant:
  1. runs a live web search (DuckDuckGo, no API key required),
  2. feeds the fresh result snippets to the LLM as grounding context,
  3. returns a concise answer based on those snippets, with source links.

This gives up-to-date answers even though the underlying LLM has a training
cut-off and no built-in web access. Degrades gracefully: if search fails it
says so; if no LLM key is set it returns the top snippets directly.
"""
from __future__ import annotations

from assistant.schemas import Link, SkillResponse
from config import settings
from utils.logger import get_logger

logger = get_logger(__name__)

_SYNTH_SYSTEM = (
    "You answer the user's question using ONLY the live web search results "
    "provided. Be accurate, direct and concise. Lead with the answer. If the "
    "results do not clearly contain the answer, say what you found and note the "
    "uncertainty. Do not invent facts beyond the results."
)


def _search(query: str, max_results: int = 5) -> list[dict]:
    """Return a list of {title, body, url} dicts from DuckDuckGo."""
    from ddgs import DDGS

    rows: list[dict] = []
    with DDGS() as ddgs:
        for r in ddgs.text(query, max_results=max_results):
            rows.append(
                {
                    "title": r.get("title", ""),
                    "body": r.get("body", "") or r.get("description", ""),
                    "url": r.get("href") or r.get("url") or "",
                }
            )
    return rows


def _synthesize(question: str, results: list[dict]) -> str:
    context_block = "\n".join(
        f"- {r['title']}: {r['body']}" for r in results if r["body"]
    )[:4000]

    if not settings.openai_enabled:
        # No LLM configured: hand back the most relevant snippet directly.
        top = next((r for r in results if r["body"]), None)
        if top:
            return f"Here's what I found: {top['title']} - {top['body']}"
        return "I found some results but couldn't summarise them without a language model."

    from openai import OpenAI

    client = OpenAI(api_key=settings.openai_api_key)  # base_url read from env (Groq/OpenAI)
    prompt = (
        f"Question: {question}\n\n"
        f"Live web search results:\n{context_block}\n\n"
        f"Answer the question directly using only these results."
    )
    resp = client.chat.completions.create(
        model=settings.openai_chat_model,
        messages=[
            {"role": "system", "content": _SYNTH_SYSTEM},
            {"role": "user", "content": prompt},
        ],
        temperature=0.3,
        max_tokens=600,
    )
    return resp.choices[0].message.content.strip()


def run(text: str, entities=None, context=None) -> SkillResponse:
    try:
        results = _search(text, max_results=5)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Web search failed: %s", exc)
        return SkillResponse(
            speech="I couldn't reach live web search just now. Please try again in a moment.",
            success=False,
        )

    if not results:
        return SkillResponse(
            speech="I searched the web but didn't find anything useful for that. "
                   "Try rephrasing the question.",
            success=False,
        )

    answer = _synthesize(text, results)
    links = [Link(r["title"][:60] or "Source", r["url"]) for r in results[:3] if r["url"]]
    return SkillResponse(speech=answer, links=links, data={"query": text, "num_results": len(results)})
