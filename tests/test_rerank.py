import math

import pytest
from conftest import make_image

from llmphotosearch.ranking.rerank import encode_image, match_probability, to_english


def logprobs_response(**probs: float) -> dict:
    top = [{"token": tok, "logprob": math.log(p)} for tok, p in probs.items()]
    return {"choices": [{"logprobs": {"content": [{"top_logprobs": top}]}}]}


class FakeServer:
    def __init__(self, response: dict):
        self.response = response
        self.requests = []

    def chat(self, messages, **params):
        self.requests.append((messages, params))
        return self.response


def test_match_probability_normalizes_yes_over_no(tmp_path):
    server = FakeServer(logprobs_response(Yes=0.6, yes=0.1, No=0.2, Maybe=0.1))

    p = match_probability(server, make_image(tmp_path / "a.jpg"), "un chat")

    assert p == pytest.approx(0.7 / 0.9)
    messages, params = server.requests[0]
    assert params["max_tokens"] == 1 and params["logprobs"] is True
    assert '"un chat"' in messages[0]["content"][1]["text"]


def test_match_probability_without_yes_or_no_is_zero(tmp_path):
    server = FakeServer(logprobs_response(Maybe=0.9))

    assert match_probability(server, make_image(tmp_path / "a.jpg"), "x") == 0.0


def test_encode_image_downscales(tmp_path):
    from PIL import Image

    path = tmp_path / "big.png"
    Image.new("RGB", (2000, 1000)).save(path)

    uri = encode_image(path, max_side=500)

    import base64
    import io

    img = Image.open(io.BytesIO(base64.b64decode(uri.split(",", 1)[1])))
    assert uri.startswith("data:image/jpeg;base64,")
    assert img.size == (500, 250)


def test_to_english_strips_quotes():
    server = FakeServer({"choices": [{"message": {"content": ' "two cats on a sofa"\n'}}]})

    assert to_english(server, "deux chats sur un canapé") == "two cats on a sofa"
