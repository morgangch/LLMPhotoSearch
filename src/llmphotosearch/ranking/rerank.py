"""VLM re-ranking: the match percentage is the model's probability of answering "yes"."""

import base64
import io
import math
from pathlib import Path
from typing import Protocol

from llmphotosearch.core.images import load_image

# Phrasing matters: "does this image match" is lenient with short queries (a plate
# of food "matches" a kitchen), "is this a photo of" separates matches best.
PROMPT = 'Is this a photo of: "{description}"? Answer with only yes or no.'
TRANSLATE_PROMPT = (
    "Translate this image search query into English. If it is already in English, "
    "repeat it unchanged. Output only the query.\n\n{description}"
)
YES = frozenset({"yes", "oui"})
NO = frozenset({"no", "non"})


class ChatServer(Protocol):
    def chat(self, messages: list[dict], **params) -> dict: ...


def encode_image(path: Path, max_side: int) -> str:
    """Image as a JPEG data URI, downscaled so its longest side is at most `max_side`."""
    img = load_image(path, max_side)
    img.thumbnail((max_side, max_side))
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=90)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def match_probability(
    server: ChatServer, path: Path, description: str, max_side: int = 768
) -> float:
    """Probability in [0, 1] that the image at `path` matches `description`."""
    response = server.chat(
        [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": encode_image(path, max_side)}},
                    {"type": "text", "text": PROMPT.format(description=description)},
                ],
            }
        ],
        max_tokens=1,
        temperature=0,
        logprobs=True,
        top_logprobs=20,
    )
    return yes_probability(response)


def yes_probability(response: dict) -> float:
    """P(yes) / (P(yes) + P(no)) from the first generated token's top logprobs."""
    first = response["choices"][0]["logprobs"]["content"][0]
    p_yes = p_no = 0.0
    for candidate in first["top_logprobs"]:
        token = candidate["token"].strip().lower()
        if token in YES:
            p_yes += math.exp(candidate["logprob"])
        elif token in NO:
            p_no += math.exp(candidate["logprob"])
    return p_yes / (p_yes + p_no) if p_yes + p_no > 0 else 0.0


def to_english(server: ChatServer, description: str) -> str:
    """The description translated to English, for the multilingual-weak pre-filter."""
    response = server.chat(
        [{"role": "user", "content": TRANSLATE_PROMPT.format(description=description)}],
        max_tokens=64,
        temperature=0,
    )
    return response["choices"][0]["message"]["content"].strip().strip('"') or description
