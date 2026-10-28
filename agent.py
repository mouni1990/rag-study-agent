"""
RAG Study Agent - Answer Generation Module
Sends the retrieved note chunks plus the question to Claude and streams the answer.
"""

import anthropic

MODEL = "claude-opus-5-5"

SYSTEM_PROMPT = """You are a friendly study assistant. You help a student understand their own notes.

Answer using the numbered note excerpts provided with each question. Cite the excerpts you rely on with their numbers in square brackets, like [1] or [2][3].

If the excerpts don't contain the answer, say so plainly, then offer a short general explanation clearly labelled as not coming from their notes.

Keep answers clear and well organised. Use short paragraphs or bullet points, and Markdown formatting where it helps."""

# Only the most recent turns are sent, so long chats stay fast and cheap
MAX_HISTORY_MESSAGES = 10

_client = None


def get_client():
    """Create the Anthropic client once. It reads ANTHROPIC_API_KEY from the environment."""
    global _client
    if _client is None:
        _client = anthropic.Anthropic()
    return _client


def build_prompt(question, excerpts):
    """Combine the retrieved excerpts and the question into one user message."""
    if not excerpts:
        notes = "(No matching excerpts were found in the notes.)"
    else:
        notes = "\n\n".join(
            f'<excerpt number="{i}" source="{c["source"]}">\n{c["text"].strip()}\n</excerpt>'
            for i, c in enumerate(excerpts, start=1)
        )
    return f"<notes>\n{notes}\n</notes>\n\nQuestion: {question}"


def stream_answer(question, excerpts, history=None):
    """
    Stream Claude's answer piece by piece.

    Args:
        question (str): The student's question.
        excerpts (list[dict]): Chunks returned by rag.search().
        history (list[dict]): Earlier {"role", "content"} messages in this chat.

    Yields:
        str: Pieces of the answer text as they arrive.
    """
    messages = list(history or [])[-MAX_HISTORY_MESSAGES:]
    messages.append({"role": "user", "content": build_prompt(question, excerpts)})

    try:
        stream = get_client().messages.create(
            model=MODEL,
            max_tokens=64000,
            system=SYSTEM_PROMPT,
            messages=messages,
            output_config={"effort": "medium"},
            stream=True,
            # If a safety filter declines the request, retry it on a fallback model
            extra_headers={"anthropic-beta": "server-side-fallback-2026-07-01"},
            extra_body={"fallbacks": "default"},
        )
        for event in stream:
            if event.type == "content_block_delta" and event.delta.type == "text_delta":
                yield event.delta.text
            elif event.type == "message_delta" and event.delta.stop_reason == "refusal":
                yield "\n\n_Sorry, I can't help with that request._"
            elif event.type == "message_delta" and event.delta.stop_reason == "max_tokens":
                yield "\n\n_(The answer was cut off because it got too long.)_"
    except anthropic.AuthenticationError:
        yield "**Error:** your Anthropic API key is missing or invalid. Check the `.env` file."
    except anthropic.RateLimitError:
        yield "**Error:** too many requests right now. Please wait a moment and try again."
    except anthropic.APIConnectionError:
        yield "**Error:** couldn't reach the Anthropic API. Check your internet connection."
    except anthropic.APIStatusError as e:
        yield f"**Error from the API ({e.status_code}):** {e.message}"
