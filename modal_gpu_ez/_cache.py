"""Modal Volume 캐시 관리"""

from __future__ import annotations

import logging
import time
from typing import Any

import modal

from modal_gpu_ez._types import CacheResult, CacheStatus, CleanResult

logger = logging.getLogger("modal_gpu_ez")

CACHE_MOUNT_PATH = "/cache"
VOLUME_NAME = "modal-gpu-ez-cache"


def check_cache(model_id: str, volume: Any) -> CacheStatus:
    """Volume에서 모델 캐시 존재 여부 확인"""
    if volume is None:
        return CacheStatus(is_cached=False)

    cache_dir = f"models--{model_id.replace('/', '--')}"
    expected = f"{CACHE_MOUNT_PATH}/{cache_dir}"
    try:
        entries = list(volume.listdir(cache_dir))
        if entries:
            return CacheStatus(is_cached=True, cache_path=expected)
    except Exception as e:
        logger.debug(f"캐시 확인 실패 ({model_id}): {e}")
    return CacheStatus(is_cached=False)


class CacheAPI:
    """mg.cache.upload() / list() / clear()"""

    def _require_modal(self) -> Any:
        """Modal 연결 검증 후 context 반환"""
        from modal_gpu_ez._core import get_ctx
        ctx = get_ctx()
        if not ctx.env.has_modal:
            raise RuntimeError("캐시 기능은 MODAL_TOKEN_ID/MODAL_TOKEN_SECRET이 필요합니다.")
        return ctx

    def upload(self, model_id: str) -> CacheResult:
        """모델을 Volume에 사전 업로드"""
        ctx = self._require_modal()
        logger.info(f"모델 캐시 업로드 시작: {model_id}")

        volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)
        image = (
            modal.Image.debian_slim(python_version="3.12")
            .pip_install("huggingface-hub", "hf-transfer")
            .env({"HF_HUB_CACHE": CACHE_MOUNT_PATH, "HF_HUB_ENABLE_HF_TRANSFER": "1"})
        )

        app = modal.App("modal-gpu-ez-cache")
        secrets: list[Any] = []
        if ctx.env.hf_token:
            secrets.append(modal.Secret.from_dict({"HF_TOKEN": ctx.env.hf_token}))

        start = time.time()

        @app.function(
            image=image,
            volumes={CACHE_MOUNT_PATH: volume},
            secrets=secrets,
            timeout=1800,
            serialized=True,
        )
        def _download(repo_id: str) -> str:
            import os
            from huggingface_hub import snapshot_download
            return snapshot_download(repo_id, cache_dir=CACHE_MOUNT_PATH, token=os.environ.get("HF_TOKEN"))

        with app.run():
            cache_path = _download.remote(model_id)

        elapsed = time.time() - start
        logger.info(f"캐시 업로드 완료: {model_id} ({elapsed:.1f}s)")

        from modal_gpu_ez._db import log_cache
        log_cache(model_id, "modal")

        return CacheResult(model_id=model_id, cache_path=cache_path, elapsed_sec=elapsed)

    def list(self) -> list[CacheStatus]:
        """캐시된 모델 목록"""
        ctx = self._require_modal()
        results: list[CacheStatus] = []
        try:
            entries = list(ctx.volume.listdir("/"))
            for entry in entries:
                name = getattr(entry, "path", str(entry))
                if name.startswith("models--"):
                    results.append(CacheStatus(is_cached=True, cache_path=f"{CACHE_MOUNT_PATH}/{name}"))
        except Exception as e:
            logger.warning(f"캐시 목록 조회 실패: {e}")

        if results:
            print(f"\n캐시된 모델: {len(results)}개")
            for r in results:
                print(f"  - {r.cache_path}")
        else:
            print("\n캐시된 모델이 없습니다.")
        print()
        return results

    def clear(self, model_id: str | None = None) -> CleanResult:
        """캐시 삭제"""
        ctx = self._require_modal()
        deleted = 0
        if ctx.volume is None:
            return CleanResult(deleted_count=0)

        if model_id:
            cache_dir = f"models--{model_id.replace('/', '--')}"
            try:
                ctx.volume.remove_file(cache_dir)
                deleted = 1
                logger.info(f"캐시 삭제: {model_id}")
            except Exception:
                logger.warning(f"캐시를 찾을 수 없습니다: {model_id}")
        else:
            try:
                for entry in list(ctx.volume.listdir("/")):
                    name = getattr(entry, "path", str(entry))
                    if name.startswith("models--"):
                        ctx.volume.remove_file(name)
                        deleted += 1
                logger.info(f"전체 캐시 삭제: {deleted}개")
            except Exception as e:
                logger.warning(f"캐시 삭제 중 오류: {e}")

        return CleanResult(deleted_count=deleted)
