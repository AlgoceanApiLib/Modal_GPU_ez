"""데이터 타입 정의"""

from __future__ import annotations

from dataclasses import dataclass


def _mask(value: str | None) -> str:
    """토큰 마스킹: 앞 4자만 보여주고 나머지는 *"""
    if not value:
        return "(없음)"
    if len(value) <= 4:
        return "****"
    return value[:4] + "*" * (len(value) - 4)


@dataclass(frozen=True)
class EnvConfig:
    """환경변수 설정 — 토큰이 절대 노출되지 않는다."""
    modal_token_id: str
    modal_token_secret: str
    hf_token: str | None = None

    @property
    def has_modal(self) -> bool:
        return bool(self.modal_token_id and self.modal_token_secret)

    def __repr__(self) -> str:
        return (
            f"EnvConfig(modal_token_id='{_mask(self.modal_token_id)}', "
            f"modal_token_secret='{_mask(self.modal_token_secret)}', "
            f"hf_token='{_mask(self.hf_token)}')"
        )

    def __str__(self) -> str:
        return self.__repr__()


@dataclass(frozen=True)
class ModelInfo:
    """HF Hub 모델 메타정보"""
    repo_id: str
    pipeline_tag: str | None
    library_name: str | None
    model_size_bytes: int

    @property
    def model_size_gb(self) -> float:
        return self.model_size_bytes / (1024 ** 3)


@dataclass(frozen=True)
class GPUSpec:
    """GPU 하드웨어 스펙"""
    name: str
    vram_gb: int
    modal_string: str
    price_per_hour: float


@dataclass(frozen=True)
class CostEstimate:
    """비용 예측 결과"""
    gpu_name: str
    price_per_hour: float
    estimated_total: float
    currency: str = "USD"


@dataclass(frozen=True)
class CacheResult:
    """캐시 업로드 결과"""
    model_id: str
    cache_path: str
    elapsed_sec: float


@dataclass(frozen=True)
class CacheStatus:
    """캐시 존재 여부"""
    is_cached: bool
    cache_path: str = ""


@dataclass(frozen=True)
class CleanResult:
    """캐시 삭제 결과"""
    deleted_count: int


@dataclass
class BudgetConfig:
    """월 예산 설정"""
    monthly_limit_usd: float = 50.0
    warn_at_percent: float = 80.0
    auto_fallback_to_local: bool = True
