"""원격 추론 — Modal 컨테이너"""

from __future__ import annotations

from typing import Any

import modal

from modal_gpu_ez._types import CacheStatus, ModelInfo

CACHE_MOUNT_PATH = "/cache"

# 동일 패키지셋이면 Image 재빌드 방지
_image_cache: dict[tuple[str, ...], modal.Image] = {}


def _get_image(deps: list[str]) -> modal.Image:
    """동일 패키지셋이면 캐시된 Image 반환"""
    key = tuple(sorted(deps))
    if key not in _image_cache:
        _image_cache[key] = (
            modal.Image.debian_slim(python_version="3.12")
            .pip_install(*deps)
            .env({"HF_HUB_CACHE": CACHE_MOUNT_PATH, "TRANSFORMERS_CACHE": CACHE_MOUNT_PATH})
        )
    return _image_cache[key]


def run(
    model_id: str,
    gpu: str,
    input_data: Any,
    info: ModelInfo,
    ctx: Any,  # _Context
    cache_status: CacheStatus,
    **kwargs: Any,
) -> Any:
    """Modal 컨테이너에서 추론 실행"""
    from modal_gpu_ez._model import get_deps

    deps = get_deps(info)
    image = _get_image(deps)

    # Volume 마운트
    volumes: dict[str, Any] = {}
    if ctx.volume:
        volumes[CACHE_MOUNT_PATH] = ctx.volume

    # Secret 설정
    secrets: list[Any] = []
    if ctx.env.hf_token:
        secrets.append(modal.Secret.from_dict({"HF_TOKEN": ctx.env.hf_token}))

    app = modal.App("modal-gpu-ez")

    @app.function(
        image=image,
        gpu=gpu,
        volumes=volumes,
        secrets=secrets,
        timeout=300,
        scaledown_window=60,
        serialized=True,
    )
    def _run_remote(
        repo_id: str,
        library_name: str,
        pipeline_tag: str,
        user_input: Any,
        cache_path: str,
        extra_kwargs: dict[str, Any],
    ) -> Any:
        # 컨테이너 내부 — modal_gpu_ez 없이 독립 실행
        # HF_HUB_CACHE=/cache 이미 설정됨 → repo_id로 로딩하면 캐시 자동 사용
        import os
        token = os.environ.get("HF_TOKEN")

        if library_name == "sentence-transformers" or pipeline_tag == "sentence-similarity":
            from sentence_transformers import SentenceTransformer
            model = SentenceTransformer(repo_id, token=token)
            embeddings = model.encode(user_input, **extra_kwargs)
            return embeddings.tolist() if hasattr(embeddings, "tolist") else embeddings

        if pipeline_tag == "translation":
            import torch
            from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
            tokenizer = AutoTokenizer.from_pretrained(repo_id, token=token)
            model = AutoModelForSeq2SeqLM.from_pretrained(repo_id, torch_dtype=torch.float16, token=token).to("cuda")
            texts = [user_input] if isinstance(user_input, str) else user_input
            inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True).to("cuda")
            with torch.no_grad():
                outputs = model.generate(**inputs, **extra_kwargs)
            return [tokenizer.decode(t, skip_special_tokens=True) for t in outputs]

        if library_name == "diffusers" or pipeline_tag in ("text-to-image", "image-to-image"):
            import io
            import torch
            from diffusers import AutoPipelineForText2Image
            pipe = AutoPipelineForText2Image.from_pretrained(repo_id, torch_dtype=torch.float16, token=token)
            pipe.to("cuda")
            result = pipe(user_input, **extra_kwargs)
            if hasattr(result, "images") and result.images:
                buf = io.BytesIO()
                result.images[0].save(buf, format="PNG")
                return buf.getvalue()
            return result

        # 기본: transformers pipeline
        import torch
        from transformers import pipeline as hf_pipeline
        pipe = hf_pipeline(model=repo_id, torch_dtype=torch.float16, device_map="auto", token=token)
        return pipe(user_input, **extra_kwargs)

    with app.run():
        raw = _run_remote.remote(
            info.repo_id,
            info.library_name or "",
            info.pipeline_tag or "",
            input_data,
            cache_status.cache_path,
            kwargs,
        )

    return raw
