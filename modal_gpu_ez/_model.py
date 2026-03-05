"""HF 모델 메타데이터 조회 (인메모리 캐싱)"""

from __future__ import annotations

from modal_gpu_ez._types import ModelInfo

# 인메모리 캐시: model_id → ModelInfo
_cache: dict[str, ModelInfo] = {}

# library_name/pipeline_tag → pip 패키지 매핑
_DEPS_MAP: dict[str, list[str]] = {
    "sentence-transformers": ["torch", "transformers", "sentence-transformers", "huggingface-hub"],
    "diffusers": ["torch", "diffusers", "transformers", "accelerate", "safetensors", "huggingface-hub"],
}

_DEFAULT_DEPS = ["torch", "transformers", "accelerate", "huggingface-hub"]


class GatedModelError(Exception):
    """gated 모델에 토큰 없이 접근 시 발생"""


def resolve(model_id: str, hf_token: str | None = None) -> ModelInfo:
    """HF Hub 조회 → ModelInfo 반환. 동일 model_id는 캐시 히트."""
    if model_id in _cache:
        return _cache[model_id]

    from huggingface_hub import model_info as hf_model_info

    info = hf_model_info(model_id, token=hf_token)

    # gated 모델 검증
    is_gated = bool(getattr(info, "gated", False))
    if is_gated and not hf_token:
        raise GatedModelError(f"'{model_id}'는 gated 모델입니다. HF_TOKEN을 .env에 설정하세요.")

    # 모델 파일 크기 합산
    size_bytes = 0
    weight_ext = (".safetensors", ".bin", ".pt", ".ckpt")
    if info.siblings:
        for s in info.siblings:
            if s.rfilename and any(s.rfilename.endswith(e) for e in weight_ext):
                size_bytes += s.size or 0

    result = ModelInfo(
        repo_id=model_id,
        pipeline_tag=getattr(info, "pipeline_tag", None),
        library_name=getattr(info, "library_name", None),
        model_size_bytes=size_bytes,
    )
    _cache[model_id] = result
    return result


def get_deps(info: ModelInfo) -> list[str]:
    """ModelInfo → 필요한 pip 패키지 리스트"""
    if info.library_name and info.library_name in _DEPS_MAP:
        deps = list(_DEPS_MAP[info.library_name])
    else:
        deps = list(_DEFAULT_DEPS)
    # 번역 모델은 sentencepiece 필요 (MarianMT 등)
    if info.pipeline_tag == "translation" and "sentencepiece" not in deps:
        deps.append("sentencepiece")
    return deps
