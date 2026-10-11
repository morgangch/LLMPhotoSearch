"""SigLIP 2 image/text encoder running on ONNX Runtime."""

from dataclasses import dataclass

import numpy as np
import onnxruntime as ort
from huggingface_hub import hf_hub_download
from PIL import Image
from tokenizers import Tokenizer


@dataclass(frozen=True)
class SigLIPSpec:
    repo: str
    image_size: int
    # Learned similarity calibration, read from the original checkpoint
    # (`logit_scale` is stored as log).
    logit_scale: float
    logit_bias: float
    vision_file: str = "onnx/vision_model.onnx"
    # The int8 text encoder matches fp32 closely; the int8 vision encoder does not.
    text_file: str = "onnx/text_model_quantized.onnx"
    max_text_tokens: int = 64


MODELS = {
    "siglip2-base-224": SigLIPSpec(
        repo="onnx-community/siglip2-base-patch16-224-ONNX",
        image_size=224,
        logit_scale=4.724453449249268,
        logit_bias=-16.771724700927734,
    ),
}
DEFAULT_MODEL = "siglip2-base-224"


def _providers() -> list[str]:
    preferred = ["CUDAExecutionProvider", "DmlExecutionProvider", "CoreMLExecutionProvider"]
    available = ort.get_available_providers()
    return [p for p in preferred if p in available] + ["CPUExecutionProvider"]


class SigLIP:
    """Embeds images and texts into a shared space; scores are match probabilities."""

    def __init__(self, name: str = DEFAULT_MODEL):
        self.name = name
        self.spec = MODELS[name]
        self._vision: ort.InferenceSession | None = None
        self._text: ort.InferenceSession | None = None
        self._tokenizer: Tokenizer | None = None

    @property
    def image_size(self) -> int:
        return self.spec.image_size

    def _session(self, file: str) -> ort.InferenceSession:
        return ort.InferenceSession(hf_hub_download(self.spec.repo, file), providers=_providers())

    def preprocess(self, image: Image.Image) -> np.ndarray:
        """RGB image -> normalized CHW float array, as the SigLIP processor does."""
        size = self.spec.image_size
        img = image.resize((size, size), Image.Resampling.BILINEAR)
        x = np.asarray(img, dtype=np.float32) / 127.5 - 1.0
        return x.transpose(2, 0, 1)

    def embed_images(self, pixels: np.ndarray) -> np.ndarray:
        """Batch of preprocessed images (N, 3, H, W) -> unit embeddings (N, D)."""
        if self._vision is None:
            self._vision = self._session(self.spec.vision_file)
        (out,) = self._vision.run(["pooler_output"], {"pixel_values": pixels})
        return _normalize(out)

    def embed_text(self, text: str) -> np.ndarray:
        """Text -> unit embedding (D,)."""
        if self._tokenizer is None:
            self._tokenizer = Tokenizer.from_file(hf_hub_download(self.spec.repo, "tokenizer.json"))
            self._tokenizer.enable_truncation(self.spec.max_text_tokens)
        if self._text is None:
            self._text = self._session(self.spec.text_file)
        # SigLIP 2 was trained on lowercased text.
        ids = np.array([self._tokenizer.encode(text.lower()).ids], dtype=np.int64)
        (out,) = self._text.run(["pooler_output"], {"input_ids": ids})
        return _normalize(out)[0]

    def probability(self, similarity: np.ndarray) -> np.ndarray:
        """Cosine similarity -> probability that image and text match."""
        logits = np.exp(self.spec.logit_scale) * similarity + self.spec.logit_bias
        return 1.0 / (1.0 + np.exp(-logits))


def _normalize(x: np.ndarray) -> np.ndarray:
    return x / np.linalg.norm(x, axis=-1, keepdims=True)
