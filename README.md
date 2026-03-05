# modal_gpu_ez

Modal 서버리스 GPU + HuggingFace 모델을 **원라이너**로 사용하는 Python 라이브러리.

```python
import modal_gpu_ez as mg

result = mg.use("distilgpt2", "T4", input="Hello world", max_new_tokens=50)
```

---

## 권장 사용법

### 1단계: 환경 설정

`.env` 파일에 토큰을 설정한다. 이후 모든 호출에서 자동으로 로드된다.

```env
MODAL_TOKEN_ID=ak-xxxxxxxxxxxxxxxxxxxx
MODAL_TOKEN_SECRET=as-xxxxxxxxxxxxxxxxxxxx
HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxx
```

### 2단계: 모델 사전 업로드 (권장)

원격 GPU를 사용할 모델은 **반드시 먼저 Volume에 업로드**한다. 업로드하지 않으면 매 호출마다 모델을 다운로드하여 속도가 크게 느려진다.

```python
import modal_gpu_ez as mg

# 사용할 모델을 1회 업로드 (이후 재업로드 불필요)
mg.cache.upload("distilgpt2")
mg.cache.upload("Helsinki-NLP/opus-mt-tc-big-en-ko")
mg.cache.upload("BAAI/bge-small-en-v1.5")

# 업로드 상태 확인
mg.cache.list()
```

> 로컬 실행(`gpu="Local"`)만 사용할 경우 업로드는 필요 없다.

### 3단계: 추론

```python
# 원격 GPU — 사전 업로드된 모델은 즉시 실행
result = mg.use("distilgpt2", "T4", input="Hello", max_new_tokens=20)

# 로컬 — 업로드 없이 바로 사용 (CUDA → MPS → CPU 자동 감지)
result = mg.use("distilgpt2", "Local", input="Hello", max_new_tokens=20)
```

### 4단계: 예산 설정 (선택)

원격 GPU 비용이 걱정되면 월 예산을 설정한다. 초과 시 자동으로 Local에서 실행된다.

```python
mg.budget.monthly_limit_usd = 10.0   # 월 $10 제한
mg.monthlyCost()                      # 현재 사용량 확인
```

### 5단계: 상태 확인

```python
mg.check()   # 캐시, 이력, 통계, 비용 한눈에 확인
```

### 요약

| 순서 | 할 일 | 빈도 |
|------|-------|------|
| 1 | `.env`에 토큰 설정 | 최초 1회 |
| 2 | `mg.cache.upload("모델ID")` | 모델당 1회 |
| 3 | `mg.use("모델ID", "GPU", input=...)` | 매 호출 |
| 4 | `mg.budget.monthly_limit_usd = N` | 선택 |
| 5 | `mg.check()` | 수시 |

---

## 특징

- **원라이너 추론** — 초기화 없이 `mg.use()` 한 줄로 추론 실행
- **모델 타입 자동 감지** — text-generation, translation, embedding, diffusion 등
- **로컬/원격 자동 전환** — `"Local"` 지정 시 CUDA → MPS → CPU 자동 감지
- **비용 추적 + 예산 관리** — 월 사용량 추적, 예산 초과 시 Local 자동 폴백
- **비동기 지원** — `async_use()`로 FastAPI 등에서 바로 사용
- **콜드스타트 최소화** — `mg.cache.upload()`으로 모델 사전 업로드
- **토큰 자동 마스킹** — 로그/출력에 토큰이 절대 노출되지 않음

---

## 설치

```bash
pip install modal-gpu-ez
```

모델 타입별 추가 의존성:

```bash
pip install modal-gpu-ez[embedding]     # sentence-transformers
pip install modal-gpu-ez[translation]   # sentencepiece (MarianMT 등)
pip install modal-gpu-ez[image]         # diffusers (이미지 생성)
pip install modal-gpu-ez[all]           # 전부 설치
```

---

## 환경 설정

프로젝트 루트에 `.env` 파일을 생성한다:

```env
MODAL_TOKEN_ID=ak-xxxxxxxxxxxxxxxxxxxx
MODAL_TOKEN_SECRET=as-xxxxxxxxxxxxxxxxxxxx
HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxx
```

| 변수 | 필수 | 설명 |
|------|------|------|
| `MODAL_TOKEN_ID` | 원격 GPU 시 필수 | Modal 토큰 ID (`ak-` 접두사) |
| `MODAL_TOKEN_SECRET` | 원격 GPU 시 필수 | Modal 토큰 Secret (`as-` 접두사) |
| `HF_TOKEN` | 선택 | HuggingFace 토큰. gated 모델(Llama, Gemma 등) 접근 시 필요 |

> 로컬 실행(`gpu="Local"`)만 사용할 경우 Modal 토큰 없이도 동작한다.

### Modal 토큰 발급

1. [modal.com](https://modal.com) 가입/로그인
2. Settings → API Tokens → Create Token
3. Token ID(`ak-`)와 Token Secret(`as-`) 복사

또는 CLI:
```bash
pip install modal && modal token new
```

---

## API

### `mg.use(model, gpu, *, input, **kwargs) -> Any`

모델에 input을 전달하고 결과를 반환한다. 모델 타입을 자동 감지한다.

```python
import modal_gpu_ez as mg

result = mg.use("distilgpt2", "T4", input="Hello", max_new_tokens=20)
```

| 파라미터 | 타입 | 설명 |
|---------|------|------|
| `model` | `str` | HuggingFace 모델 ID |
| `gpu` | `str` | GPU 이름 (`"T4"`, `"H100"`, `"auto"`, `"Local"`) |
| `input` | `Any` | 모델 입력 — 모델이 기대하는 타입 그대로 전달 |
| `**kwargs` | | 모델별 추가 파라미터 (아래 예시 참고) |

---

### 모델 타입별 사용법

#### 텍스트 생성 (text-generation)

```python
result = mg.use("distilgpt2", "T4",
    input="Once upon a time",
    max_new_tokens=100,
    temperature=0.7,
    do_sample=True
)
# result: [{'generated_text': 'Once upon a time there was...'}]
```

`input`은 `str`. `**kwargs`는 HuggingFace `generate()` 파라미터를 그대로 전달한다.

#### 번역 (translation)

```python
result = mg.use("Helsinki-NLP/opus-mt-tc-big-en-ko", "T4",
    input="The weather is beautiful today."
)
# result: ['오늘 날씨가 아름답습니다.']
```

`input`은 `str` 또는 `list[str]`. MarianMT 계열은 `sentencepiece` 설치가 필요하다.

#### 임베딩 (sentence-similarity / feature-extraction)

```python
vectors = mg.use("BAAI/bge-small-en-v1.5", "Local",
    input=["hello world", "goodbye world"]
)
# vectors: [[0.012, -0.034, ...], [0.045, 0.023, ...]]
```

`input`은 `list[str]`. 결과는 `list[list[float]]`.

#### 이미지 생성 (text-to-image)

```python
image_bytes = mg.use("stabilityai/sdxl-turbo", "A100",
    input="a cat in space",
    num_inference_steps=4
)
# image_bytes: PNG 바이너리 (bytes)

with open("output.png", "wb") as f:
    f.write(image_bytes)
```

`input`은 `str` (프롬프트). 결과는 PNG `bytes`. `diffusers` 설치가 필요하다.

#### 로컬 실행

```python
result = mg.use("distilgpt2", "Local", input="Hello", max_new_tokens=10)
```

디바이스를 **CUDA → MPS → CPU** 순서로 자동 감지한다. 과금이 없다.

---

### `await mg.async_use(model, gpu, *, input, **kwargs) -> Any`

`use()`의 비동기 버전. FastAPI 등에서 사용한다.

```python
from fastapi import FastAPI
import modal_gpu_ez as mg

app = FastAPI()

@app.post("/generate")
async def generate(prompt: str):
    result = await mg.async_use("distilgpt2", "T4", input=prompt, max_new_tokens=50)
    return {"result": result}
```

---

### `mg.cache`

Modal Volume에 모델을 사전 업로드하여 콜드스타트를 최소화한다.

```python
mg.cache.upload("distilgpt2")   # Volume에 사전 업로드 (1회)
mg.cache.list()                  # 캐시된 모델 목록
mg.cache.clear("distilgpt2")   # 특정 모델 삭제
mg.cache.clear()                 # 전체 삭제
```

사전 업로드 후 `mg.use()` 호출 시 모델 다운로드를 건너뛰어 속도가 대폭 향상된다.

---

### `mg.budget`

월 예산을 설정하고 비용을 관리한다.

```python
mg.budget.monthly_limit_usd = 30.0       # 월 예산 (기본 $50)
mg.budget.warn_at_percent = 80.0          # 경고 시점 (기본 80%)
mg.budget.auto_fallback_to_local = True   # 초과 시 Local 자동 폴백 (기본 True)
```

예산 초과 시:
- `auto_fallback_to_local = True` → 자동으로 Local에서 실행
- `auto_fallback_to_local = False` → `RuntimeError` 발생

```python
mg.monthlyCost()
#   이번 달 사용 비용: $0.0014 USD
#   월 예산: $30.00 USD
#   남은 예산: $29.9986 USD
```

---

### `mg.listGPUs() -> list[GPUSpec]`

사용 가능한 GPU 목록을 출력한다.

```python
mg.listGPUs()
```

```
GPU          VRAM     $/hr
------------------------------
B200         192GB     $7.53
H200         141GB     $5.49
H100         80GB     $3.89
A100-80GB    80GB     $2.78
A100-40GB    40GB     $1.64
L40S         48GB     $1.58
L4           24GB     $0.59
A10G         24GB     $0.54
T4           16GB     $0.27
```

---

### `mg.estimateCost(model_id, gpu, hours) -> CostEstimate`

GPU 사용 예상 비용을 계산한다.

```python
mg.estimateCost("distilgpt2", "H200", hours=2)
#   GPU: H200
#   시간당: $5.49
#   2.0시간 예상: $10.98 USD
```

---

### `mg.check()`

상태 대시보드를 출력한다. 캐시 상태, 사용 이력, 통계, 월 비용을 확인할 수 있다.

```python
mg.check()
```

---

### `mg.help()`

전체 사용법을 출력한다.

---

## GPU 선택 가이드

| GPU | VRAM | 시간당 가격 | 추천 용도 |
|-----|------|-----------|----------|
| T4 | 16GB | $0.27 | 소형 모델 테스트, 임베딩 |
| A10G | 24GB | $0.54 | 중형 모델, 번역 |
| L4 | 24GB | $0.59 | 임베딩, 소형 LLM |
| L40S | 48GB | $1.58 | 7B LLM, 이미지 생성 |
| A100 | 40GB | $1.64 | 대형 모델 추론 |
| A100-80GB | 80GB | $2.78 | 13B+ LLM |
| H100 | 80GB | $3.89 | 대규모 추론 |
| H200 | 141GB | $5.49 | 70B LLM |
| B200 | 192GB | $7.53 | 초대형 모델 |

- `"auto"` — 모델 크기 기반 최적 GPU 자동 선택
- `"Local"` — 로컬 디바이스 (CUDA → MPS → CPU 자동 감지, 과금 없음)

---

## 토큰 보호

모든 토큰은 내부적으로 마스킹되어 `repr()`, `str()`, 로그 어디에서도 원문이 노출되지 않는다.

```python
from modal_gpu_ez._core import get_ctx
print(repr(get_ctx().env))
# EnvConfig(modal_token_id='ak-Q***', modal_token_secret='as-f***', hf_token='hf_q***')
```

---

## 프로젝트 구조

```
modal_gpu_ez/
├── __init__.py   # 공개 API
├── _types.py     # 데이터 타입
├── _core.py      # 초기화 + 환경
├── _gpu.py       # GPU 레지스트리
├── _model.py     # HF 모델 조회 (캐싱)
├── _local.py     # 로컬 추론
├── _remote.py    # 원격 추론
├── _cache.py     # Volume 캐시
├── _check.py     # 대시보드
└── _db.py        # 사용 이력
```

---

## 라이선스

Apache License 2.0
