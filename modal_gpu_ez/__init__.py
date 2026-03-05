"""
modal_gpu_ez — Modal 서버리스 GPU + HuggingFace 모델을 원라이너로 사용

사용법:
    import modal_gpu_ez as mg

    result = mg.use("distilgpt2", "T4", input="hi", max_new_tokens=5)
    result = mg.use("BAAI/bge-small-en-v1.5", "Local", input=["hello"])

    # 비동기
    result = await mg.async_use("distilgpt2", "T4", input="hi")

    # 예산 설정
    mg.budget.monthly_limit_usd = 30.0

    mg.check()
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any

from modal_gpu_ez import _core, _model, _local, _remote, _gpu, _check, _db
from modal_gpu_ez._cache import CacheAPI, check_cache
from modal_gpu_ez._types import BudgetConfig

__version__ = "1.0.0"

logger = logging.getLogger("modal_gpu_ez")

cache = CacheAPI()
budget = BudgetConfig()


def _check_budget(gpu: str) -> str | None:
    """예산 초과 여부 확인. 초과 시 경고 메시지 반환, 정상이면 None."""
    if gpu.lower() == "local":
        return None

    current = _db.get_monthly_cost()
    limit = budget.monthly_limit_usd
    warn_threshold = limit * budget.warn_at_percent / 100

    if current >= limit:
        return f"월 예산 초과! (${current:.2f} / ${limit:.2f})"
    if current >= warn_threshold:
        logger.warning(f"월 예산 경고: ${current:.2f} / ${limit:.2f} ({current/limit*100:.0f}%)")
    return None


def use(model: str, gpu: str, *, input: Any, **kwargs: Any) -> Any:
    """
    원라이너 추론 — input을 모델에 그대로 전달, 결과를 그대로 반환.

    Args:
        model: HF 모델 ID (예: "distilgpt2", "BAAI/bge-small-en-v1.5")
        gpu: GPU 이름 ("T4", "H100", "Local" 등)
        input: 모델 입력 (그대로 전달됨)
    """
    # 예산 체크
    over_msg = _check_budget(gpu)
    if over_msg and budget.auto_fallback_to_local:
        logger.warning(f"{over_msg} → Local 자동 폴백")
        gpu = "Local"
    elif over_msg:
        raise RuntimeError(over_msg)

    ctx = _core.get_ctx()
    info = _model.resolve(model, ctx.env.hf_token)
    start = time.time()

    try:
        if gpu.lower() == "local":
            result = _local.run(model, input, info, ctx.env.hf_token, **kwargs)
        else:
            if not ctx.env.has_modal:
                raise RuntimeError(
                    "원격 GPU 실행에는 MODAL_TOKEN_ID/MODAL_TOKEN_SECRET이 필요합니다. "
                    "로컬 실행은 gpu='Local'을 사용하세요."
                )
            gpu_string = _gpu.select_gpu(gpu, info.model_size_gb)
            cache_status = check_cache(model, ctx.volume)
            result = _remote.run(model, gpu_string, input, info, ctx, cache_status, **kwargs)

        elapsed = time.time() - start
        _db.log_use(model, gpu, elapsed, "success")
        _db.log_cache(model, "local" if gpu.lower() == "local" else "modal")

        # 원격 실행이면 비용 기록
        if gpu.lower() != "local":
            spec = _gpu._find(gpu)
            price = spec.price_per_hour if spec else 0.0
            _db.log_cost(model, gpu, elapsed, price)

        return result

    except Exception as e:
        elapsed = time.time() - start
        _db.log_use(model, gpu, elapsed, f"error: {type(e).__name__}")
        raise


async def async_use(model: str, gpu: str, *, input: Any, **kwargs: Any) -> Any:
    """
    use()의 비동기 버전 — FastAPI 등에서 await로 호출.

    사용법:
        result = await mg.async_use("distilgpt2", "T4", input="hi")
    """
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, lambda: use(model, gpu, input=input, **kwargs))


def listGPUs() -> list:
    """사용 가능한 GPU 목록"""
    return _gpu.list_gpus()


def estimateCost(model_id: str, gpu: str, hours: float = 1.0) -> Any:
    """GPU 사용 예상 비용"""
    return _gpu.estimate_cost(model_id, gpu, hours)


def monthlyCost() -> float:
    """이번 달 누적 비용 (USD)"""
    cost = _db.get_monthly_cost()
    print(f"\n  이번 달 사용 비용: ${cost:.4f} USD")
    print(f"  월 예산: ${budget.monthly_limit_usd:.2f} USD")
    remaining = budget.monthly_limit_usd - cost
    print(f"  남은 예산: ${remaining:.4f} USD\n")
    return cost


def check() -> None:
    """상태 대시보드"""
    _check.check()


def help() -> None:
    """사용법 출력"""
    _check.show_help()


__all__ = [
    "use", "async_use", "cache", "budget",
    "listGPUs", "estimateCost", "monthlyCost",
    "check", "help",
]
