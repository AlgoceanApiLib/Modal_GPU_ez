"""대시보드 + 도움말"""

from __future__ import annotations

from modal_gpu_ez._db import get_caches, get_history, get_monthly_cost


def check() -> None:
    """현재 상태 대시보드 출력"""
    print("\n╔══════════════════════════════════════════════════════════════╗")
    print("║                  modal_gpu_ez 상태 대시보드                  ║")
    print("╠══════════════════════════════════════════════════════════════╣")

    # 캐시 상태
    caches = get_caches()
    print("║                                                              ║")
    print("║  [캐시된 모델]                                               ║")
    if caches:
        for c in caches:
            mid = c["model_id"]
            loc = c["location"]
            print(f"║    - {mid:<30} [{loc:>5}]              ║")
    else:
        print("║    (없음)                                                    ║")

    # 최근 사용 이력
    history = get_history()
    recent = history[-10:]
    print("║                                                              ║")
    print("║  [최근 사용 이력]                                            ║")
    if recent:
        for h in reversed(recent):
            mid = h["model_id"]
            gpu = h["gpu"]
            elapsed = h.get("elapsed_sec", 0)
            status = h.get("status", "")
            short = mid if len(mid) <= 25 else f"...{mid[-22:]}"
            print(f"║    {short:<25} {gpu:<8} {elapsed:>5.1f}s {status:<10} ║")
    else:
        print("║    (없음)                                                    ║")

    # 통계
    total = len(history)
    local_runs = sum(1 for h in history if h["gpu"].lower() == "local")
    remote = total - local_runs
    success = sum(1 for h in history if h.get("status") == "success")

    print("║                                                              ║")
    print("║  [통계]                                                      ║")
    print(f"║    총 실행: {total}회 (로컬: {local_runs}, 원격: {remote})              ║")
    if total > 0:
        print(f"║    성공률: {success}/{total} ({success/total*100:.0f}%)                                      ║")
    else:
        print("║    성공률: -                                                  ║")

    # 비용
    monthly = get_monthly_cost()
    print("║                                                              ║")
    print("║  [이번 달 비용]                                              ║")
    print(f"║    누적: ${monthly:.4f} USD                                       ║")

    print("║                                                              ║")
    print("╚══════════════════════════════════════════════════════════════╝\n")


def show_help() -> None:
    """사용법 출력"""
    print("""
╔══════════════════════════════════════════════════════════════╗
║                    modal_gpu_ez 사용법                       ║
╠══════════════════════════════════════════════════════════════╣
║                                                              ║
║  import modal_gpu_ez as mg                                   ║
║                                                              ║
║  # 1. 원라이너 추론 (input 그대로 전달, 결과 그대로 반환)    ║
║  result = mg.use("model_id", "H200", input="안녕!")          ║
║                                                              ║
║  # 2. 로컬 실행 (CUDA → MPS → CPU 자동 감지, 과금 없음)     ║
║  result = mg.use("model_id", "Local", input="hello")         ║
║                                                              ║
║  # 3. 비동기 추론 (FastAPI 등)                               ║
║  result = await mg.async_use("model_id", "T4", input="hi")  ║
║                                                              ║
║  # 4. 상태 확인                                              ║
║  mg.check()                                                  ║
║                                                              ║
║  # 5. 캐시 관리 (콜드스타트 최소화)                          ║
║  mg.cache.upload("model_id")                                 ║
║  mg.cache.list()                                             ║
║  mg.cache.clear("model_id")                                  ║
║                                                              ║
║  # 6. GPU 정보                                               ║
║  mg.listGPUs()                                               ║
║  mg.estimateCost("model_id", "H200", hours=2)                ║
║                                                              ║
║  # 7. 예산 관리                                              ║
║  mg.budget.monthly_limit_usd = 30.0  # 월 예산 설정         ║
║  mg.budget.auto_fallback_to_local = True  # 초과 시 Local    ║
║  mg.monthlyCost()  # 이번 달 비용 확인                       ║
╚══════════════════════════════════════════════════════════════╝
""")
