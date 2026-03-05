"""로컬 추론 — CUDA → MPS → CPU"""

from __future__ import annotations

import logging
from typing import Any

from modal_gpu_ez._types import ModelInfo

logger = logging.getLogger("modal_gpu_ez")


def detect_device() -> str:
    """CUDA → MPS → CPU 순서로 탐색"""
    try:
        import torch
    except ImportError:
        logger.warning("torch 미설치 — CPU로 폴백")
        return "cpu"

    if torch.cuda.is_available():
        logger.info(f"로컬 디바이스: CUDA ({torch.cuda.get_device_name(0)})")
        return "cuda"
    if torch.backends.mps.is_available():
        logger.info("로컬 디바이스: MPS (Apple Silicon)")
        return "mps"
    logger.info("로컬 디바이스: CPU")
    return "cpu"


def run(
    model_id: str,
    input_data: Any,
    info: ModelInfo,
    hf_token: str | None,
    **kwargs: Any,
) -> Any:
    """로컬 디바이스에서 추론. info.library_name으로 로더 결정."""
    device = detect_device()
    lib = info.library_name or ""
    tag = info.pipeline_tag or ""

    if lib == "sentence-transformers" or tag == "sentence-similarity":
        return _run_sentence_transformers(model_id, input_data, device, hf_token, **kwargs)
    if lib == "diffusers" or tag in ("text-to-image", "image-to-image"):
        return _run_diffusers(model_id, input_data, device, hf_token, **kwargs)
    if tag == "translation":
        return _run_translation(model_id, input_data, device, hf_token, **kwargs)
    return _run_transformers(model_id, input_data, device, hf_token, **kwargs)


def _get_dtype(device: str) -> Any:
    import torch
    return torch.float16 if device == "cuda" else torch.float32


def _run_transformers(
    model_id: str, input_data: Any, device: str, token: str | None, **kwargs: Any,
) -> Any:
    """transformers pipeline"""
    from transformers import pipeline as hf_pipeline
    dtype = _get_dtype(device)
    if device == "cuda":
        pipe = hf_pipeline(model=model_id, torch_dtype=dtype, device_map="auto", token=token)
    else:
        pipe = hf_pipeline(model=model_id, torch_dtype=dtype, device=device, token=token)
    return pipe(input_data, **kwargs)


def _run_sentence_transformers(
    model_id: str, input_data: Any, device: str, token: str | None, **kwargs: Any,
) -> Any:
    """sentence-transformers"""
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(model_id, token=token, device=device)
    embeddings = model.encode(input_data, **kwargs)
    if hasattr(embeddings, "tolist"):
        return embeddings.tolist()
    return embeddings


def _run_translation(
    model_id: str, input_data: Any, device: str, token: str | None, **kwargs: Any,
) -> Any:
    """MarianMT 등 번역 모델"""
    import torch
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
    dtype = _get_dtype(device)
    tokenizer = AutoTokenizer.from_pretrained(model_id, token=token)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_id, torch_dtype=dtype, token=token).to(device)
    if isinstance(input_data, str):
        input_data = [input_data]
    inputs = tokenizer(input_data, return_tensors="pt", padding=True, truncation=True).to(device)
    with torch.no_grad():
        outputs = model.generate(**inputs, **kwargs)
    return [tokenizer.decode(t, skip_special_tokens=True) for t in outputs]


def _run_diffusers(
    model_id: str, input_data: Any, device: str, token: str | None, **kwargs: Any,
) -> Any:
    """diffusers — 이미지 결과는 PNG bytes"""
    import io
    from diffusers import AutoPipelineForText2Image
    dtype = _get_dtype(device)
    pipe = AutoPipelineForText2Image.from_pretrained(model_id, torch_dtype=dtype, token=token)
    pipe.to(device)
    result = pipe(input_data, **kwargs)
    if hasattr(result, "images") and result.images:
        buf = io.BytesIO()
        result.images[0].save(buf, format="PNG")
        return buf.getvalue()
    return result
