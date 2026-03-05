"""사용 이력 DB — .modal_gpu_ez/ 폴더에 txt 파일"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger("modal_gpu_ez")


def _db_dir() -> Path:
    """DB 디렉토리 반환. 없으면 생성."""
    d = Path(__file__).resolve().parent.parent / ".modal_gpu_ez"
    d.mkdir(exist_ok=True)
    return d


def _read(filename: str) -> list[dict[str, Any]]:
    """txt에서 JSON 레코드 읽기"""
    path = _db_dir() / filename
    if not path.exists():
        return []
    records = []
    for line in path.read_text(encoding="utf-8").strip().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            logger.warning(f"손상된 레코드 무시 ({filename}): {line[:50]}")
    return records


def _append(filename: str, record: dict[str, Any]) -> None:
    """JSON 레코드 한 줄 추가"""
    path = _db_dir() / filename
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def _overwrite(filename: str, records: list[dict[str, Any]]) -> None:
    """레코드 전체 덮어쓰기"""
    path = _db_dir() / filename
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


# --- 모델 사용 기록 ---

def log_use(model_id: str, gpu: str, elapsed: float, status: str = "success") -> None:
    _append("history.txt", {
        "model_id": model_id, "gpu": gpu,
        "elapsed_sec": round(elapsed, 1), "status": status,
        "time": datetime.now().isoformat(timespec="seconds"),
    })


def get_history() -> list[dict[str, Any]]:
    return _read("history.txt")


# --- 캐시 상태 ---

def log_cache(model_id: str, location: str) -> None:
    records = _read("caches.txt")
    records = [r for r in records if not (r["model_id"] == model_id and r["location"] == location)]
    records.append({
        "model_id": model_id, "location": location,
        "time": datetime.now().isoformat(timespec="seconds"),
    })
    _overwrite("caches.txt", records)


def get_caches() -> list[dict[str, Any]]:
    return _read("caches.txt")


def remove_cache_record(model_id: str, location: str | None = None) -> None:
    records = _read("caches.txt")
    if location:
        records = [r for r in records if not (r["model_id"] == model_id and r["location"] == location)]
    else:
        records = [r for r in records if r["model_id"] != model_id]
    _overwrite("caches.txt", records)


# --- 비용 추적 ---

def log_cost(model_id: str, gpu: str, elapsed_sec: float, price_per_hour: float) -> None:
    """원격 실행 비용을 기록한다."""
    cost = round(price_per_hour * elapsed_sec / 3600, 6)
    _append("costs.txt", {
        "model_id": model_id, "gpu": gpu,
        "elapsed_sec": round(elapsed_sec, 1),
        "price_per_hour": price_per_hour,
        "cost_usd": cost,
        "month": datetime.now().strftime("%Y-%m"),
        "time": datetime.now().isoformat(timespec="seconds"),
    })


def get_monthly_cost(month: str | None = None) -> float:
    """해당 월의 누적 비용(USD)을 반환한다. month 미지정 시 이번 달."""
    if month is None:
        month = datetime.now().strftime("%Y-%m")
    records = _read("costs.txt")
    return sum(r.get("cost_usd", 0.0) for r in records if r.get("month") == month)


def get_cost_history() -> list[dict[str, Any]]:
    """전체 비용 이력을 반환한다."""
    return _read("costs.txt")
