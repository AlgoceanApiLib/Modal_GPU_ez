"""GPU 레지스트리 + 선택 + 비용"""

from __future__ import annotations

from modal_gpu_ez._types import CostEstimate, GPUSpec

# Modal 지원 GPU 목록 (2026-03 기준)
GPU_REGISTRY: dict[str, GPUSpec] = {
    "B200": GPUSpec("B200", 192, "B200", 7.53),
    "H200": GPUSpec("H200", 141, "H200", 5.49),
    "H100": GPUSpec("H100", 80, "H100", 3.89),
    "A100-80GB": GPUSpec("A100-80GB", 80, "A100-80GB", 2.78),
    "A100": GPUSpec("A100-40GB", 40, "A100", 1.64),
    "L40S": GPUSpec("L40S", 48, "L40S", 1.58),
    "L4": GPUSpec("L4", 24, "L4", 0.59),
    "A10G": GPUSpec("A10G", 24, "A10G", 0.54),
    "T4": GPUSpec("T4", 16, "T4", 0.27),
}

# 별칭 (소문자 → 정식 키)
_ALIASES: dict[str, str] = {
    "a100-40gb": "A100",
    "a100-80": "A100-80GB",
    "a100_80gb": "A100-80GB",
}


class UnsupportedGPUError(Exception):
    """지원하지 않는 GPU"""


def _find(name: str) -> GPUSpec | None:
    """이름으로 GPU 스펙을 찾는다 (대소문자 무관)."""
    upper = name.upper()
    if upper in GPU_REGISTRY:
        return GPU_REGISTRY[upper]
    lower = name.lower()
    if lower in _ALIASES:
        return GPU_REGISTRY[_ALIASES[lower]]
    return None


def list_gpus() -> list[GPUSpec]:
    """사용 가능한 GPU 목록을 출력하고 반환한다."""
    gpus = sorted(GPU_REGISTRY.values(), key=lambda g: g.price_per_hour, reverse=True)
    print(f"\n{'GPU':<12} {'VRAM':<8} {'$/hr':<8}")
    print("-" * 30)
    for g in gpus:
        print(f"{g.name:<12} {g.vram_gb}GB{'':<4} ${g.price_per_hour:.2f}")
    print()
    return gpus


def select_gpu(gpu: str, model_size_gb: float = 0.0) -> str:
    """GPU 이름 검증 후 modal_string 반환. 'auto'면 최적 선택."""
    if gpu.lower() == "local":
        return "Local"

    if gpu.lower() == "auto":
        required = model_size_gb * 1.2
        candidates = sorted(
            [g for g in GPU_REGISTRY.values() if g.vram_gb >= required],
            key=lambda g: g.price_per_hour,
        )
        best = candidates[0] if candidates else max(GPU_REGISTRY.values(), key=lambda g: g.vram_gb)
        return best.modal_string

    spec = _find(gpu)
    if spec is None:
        available = ", ".join(GPU_REGISTRY.keys())
        raise UnsupportedGPUError(f"'{gpu}'은 지원하지 않는 GPU입니다. 사용 가능: {available}")
    return spec.modal_string


def estimate_cost(model_id: str, gpu: str, hours: float = 1.0) -> CostEstimate:
    """GPU 사용 예상 비용을 계산한다."""
    spec = _find(gpu)
    price = spec.price_per_hour if spec else 0.0
    result = CostEstimate(
        gpu_name=gpu,
        price_per_hour=price,
        estimated_total=round(price * hours, 2),
    )
    print(f"\n  GPU: {result.gpu_name}\n  시간당: ${result.price_per_hour:.2f}\n  {hours}시간 예상: ${result.estimated_total:.2f} {result.currency}\n")
    return result
