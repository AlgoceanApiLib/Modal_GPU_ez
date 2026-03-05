"""초기화 + 환경 설정 (싱글톤 컨텍스트)"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

from modal_gpu_ez._types import EnvConfig


class _Context:
    """초기화된 앱 컨텍스트"""
    __slots__ = ("env", "volume", "logger")

    def __init__(self, env: EnvConfig, volume: object, logger: logging.Logger) -> None:
        self.env = env
        self.volume = volume  # modal.Volume | None
        self.logger = logger


_ctx: _Context | None = None

_LOGGER_NAME = "modal_gpu_ez"
_VOLUME_NAME = "modal-gpu-ez-cache"


def _setup_logger() -> logging.Logger:
    """[modal_gpu_ez] 접두사 로거를 반환한다."""
    logger = logging.getLogger(_LOGGER_NAME)
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter("\033[36m[modal_gpu_ez]\033[0m %(message)s"))
        logger.addHandler(handler)
        logger.propagate = False
    return logger


def _setup_model_dir() -> Path:
    """로컬 모델 저장 경로를 프로젝트 루트/Models로 설정한다."""
    project_root = Path(__file__).resolve().parent.parent
    models_dir = project_root / "Models"
    models_dir.mkdir(exist_ok=True)
    os.environ["HF_HOME"] = str(models_dir)
    os.environ["HF_HUB_CACHE"] = str(models_dir / "hub")
    os.environ["SENTENCE_TRANSFORMERS_HOME"] = str(models_dir / "sentence_transformers")
    return models_dir


def _init() -> _Context:
    """최초 1회: .env 로드 → Modal 환경변수 설정 → Volume 생성"""
    global _ctx

    logger = _setup_logger()
    models_dir = _setup_model_dir()
    logger.info(f"모델 저장 경로: {models_dir}")

    # .env 로드
    from dotenv import load_dotenv
    env_path = Path(".env")
    if env_path.exists():
        load_dotenv(env_path)

    env = EnvConfig(
        modal_token_id=os.environ.get("MODAL_TOKEN_ID", ""),
        modal_token_secret=os.environ.get("MODAL_TOKEN_SECRET", ""),
        hf_token=os.environ.get("HF_TOKEN") or None,
    )

    # Modal 환경변수 설정
    volume = None
    if env.has_modal:
        os.environ["MODAL_TOKEN_ID"] = env.modal_token_id
        os.environ["MODAL_TOKEN_SECRET"] = env.modal_token_secret
        import modal
        volume = modal.Volume.from_name(_VOLUME_NAME, create_if_missing=True)
    else:
        logger.warning("MODAL_TOKEN_ID/MODAL_TOKEN_SECRET 미설정 — Local 모드만 사용 가능")

    logger.info("modal_gpu_ez 초기화 완료")
    _ctx = _Context(env=env, volume=volume, logger=logger)
    return _ctx


def get_ctx() -> _Context:
    """싱글톤 반환. 미초기화 시 자동 _init()"""
    if _ctx is None:
        return _init()
    return _ctx
