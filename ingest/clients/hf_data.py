"""HuggingFace 추론 API 연동 (외부 연동 계층 · U5 결정)

**모델을 내려받지 않는다.** `transformers` + `torch` 는 2GB 가 넘어 Vercel 함수 한도(500MB)에
들어가지 않는다 (실측 — 지금 site-packages 가 341MB 다). 그래서 원격 추론 API 를 HTTP 로 부른다.

    https://router.huggingface.co/hf-inference/models/{모델}

인증키
------
`.key` 에 `HUGGINGFACE_ACCESS_TOKEN = hf_...` 한 줄. 다른 키와 **같은 규칙**으로
`common/secrets.py` 가 읽는다 (환경변수 → `.env` → `.key`).

실제 호출로 확인한 것 (2026-08-02)
--------------------------------
1. **옛 주소 `api-inference.huggingface.co` 는 DNS 조차 안 잡힌다.** `router.huggingface.co`
   경로를 써야 한다.

2. **`top_k` 를 주지 않으면 응답 모양이 애매하다.** 텍스트를 3개 보내면
   `[[{negative},{neutral},{positive}]]` 가 오는데, 이게 "3개 입력의 top-1" 인지
   "1개 입력의 3라벨 분포" 인지 모양만 봐서는 구분되지 않는다 (라벨이 3개인 모델이라 겹친다).
   → **`parameters.top_k = 1` 을 반드시 붙인다.** 그러면 입력 하나당 리스트 하나로 갈라져
   `[[{...}],[{...}],[{...}]]` 가 되어 순서 대응이 확실해진다.

3. **지연 실측** — 감성 배치 30건 콜드 3.65초 · 웜 0.66초. 제로샷 분류 1.14초.
   서버리스 60초 안에 들어가지만, 그래도 **없으면 못 도는 자리에는 쓰지 않는다** (아래 4번).

4. **한국어 제로샷은 모델을 가려야 한다.** 같은 문장("반도체 메모리 D램 낸드플래시 파운드리")으로
   재 봤을 때

       facebook/bart-large-mnli        bank 0.30 · automobile 0.26 · semiconductor 0.25  ❌ 오답
       joeddav/xlm-roberta-large-xnli  반도체 0.986                                       ✅ 정답

   영어 전용 NLI 모델은 한국어를 사실상 못 읽는다. 다국어 모델을 쓴다.

책임 경계
--------
**이 모듈이 죽어도 리서치는 돌아야 한다.** 감성·분류는 '있으면 좋은 것' 이지
없으면 판단이 안 되는 것이 아니다. 그래서 실패를 예외로 올리지 않고
`{"available": False, "reason": ...}` 로 돌려준다. 부르는 쪽(knowledge·workstreams)이
규칙 기반으로 떨어지고 그 사실을 `G-SOURCE` 로 밝힌다 (GIC 불변원칙 §2-2).
"""

from __future__ import annotations

import json
import threading
import time
from typing import Dict, List, Optional, Sequence, Tuple
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request, urlopen

from common import secrets

HF_BASE_URL = "https://router.huggingface.co/hf-inference/models"
REQUEST_TIMEOUT = 20                 # 콜드스타트 실측 3.65초 — 여유를 두되 서버리스 60초는 넘기지 않는다
CACHE_TTL = 3600                     # 같은 공시 제목은 하루 종일 같은 감성이다. 1시간이면 충분하다
MAX_BATCH = 64                       # 한 번에 보낼 텍스트 수 (그보다 많으면 나눠 부른다)

KEY_NAMES = ("HUGGINGFACE_ACCESS_TOKEN", "HF_TOKEN", "HUGGINGFACE_API_KEY")

# 쓰는 모델 세 개 — 감성·분류는 위 4번, 임베딩은 `scripts/build_report_index.py` 머리말 참고
SENTIMENT_MODEL = "snunlp/KR-FinBert-SC"
ZEROSHOT_MODEL = "joeddav/xlm-roberta-large-xnli"
# 한국어 검색 전용 모델. 실측 top1 91% · top3 95% (사업보고서 42곳 · 자연어 질의 22건).
# ⚠️ **빌드 스크립트와 반드시 같은 모델이어야 한다.** 다르면 질의와 색인이 다른 공간에
#    놓여 결과가 무작위가 된다 — `report_index.nearest` 가 차원이 다르면 빈 결과를 준다.
EMBED_MODEL = "nlpai-lab/KURE-v1"
EMBED_PATH = "/pipeline/feature-extraction"     # 기본 경로는 400 이 난다 (아래 embed 참고)

# 감성 라벨 → 우리 표기 (모델마다 라벨 이름이 다르므로 여기서 한 번 맞춘다)
LABEL_MAP = {"positive": "긍정", "negative": "부정", "neutral": "중립",
             "LABEL_0": "부정", "LABEL_1": "중립", "LABEL_2": "긍정"}

_cache: Dict[Tuple, Tuple[float, object]] = {}
_cache_lock = threading.Lock()

# 마지막 호출 결과 (`/api/dashboard/summary` 가 재호출 없이 상태를 보여주기 위함)
_last_attempt: Dict[str, Optional[str]] = {"result": None, "detail": None}


def load_hf_key() -> Tuple[str, str]:
    """토큰과 그 출처를 돌려준다. 없으면 `("", "none")`."""
    return secrets.load_key(KEY_NAMES)


def available() -> bool:
    return bool(load_hf_key()[0])


def _cached(key: Tuple, producer):
    """`CACHE_TTL` 초 안의 같은 요청은 저장해 둔 값을 돌려준다."""
    now = time.monotonic()
    with _cache_lock:
        hit = _cache.get(key)
        if hit and now - hit[0] < CACHE_TTL:
            return hit[1]

    value = producer()                      # 느린 네트워크 호출은 잠금 밖에서 한다

    with _cache_lock:
        _cache[key] = (time.monotonic(), value)
    return value


def _post(model: str, payload: Dict, timeout: int = REQUEST_TIMEOUT) -> object:
    """추론 API 를 부르고 파싱된 JSON 을 돌려준다.

    토큰은 Authorization 헤더에만 싣는다 — ECOS 처럼 URL 에 박히면 접근 로그에 남는다.
    """
    token, _ = load_hf_key()
    if not token:
        raise RuntimeError("HuggingFace 토큰이 없다 — .key 에 HUGGINGFACE_ACCESS_TOKEN 을 넣는다")

    request = Request(
        f"{HF_BASE_URL}/{quote(model)}",
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Authorization": f"Bearer {token}",
                 "Content-Type": "application/json",
                 "User-Agent": "api-test-GIC-harness/M5"},
    )
    with urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def _failure(reason: str) -> Dict:
    """실패를 **예외가 아니라 값으로** 돌려준다 (이 모듈의 책임 경계 참고)."""
    _last_attempt.update(result="error", detail=reason[:200])
    return {"available": False, "reason": reason, "model": ""}


# ─────────────────────────────────────────────────────────────
# 1. 금융 감성 — 공시 제목·문장 묶음
# ─────────────────────────────────────────────────────────────
def sentiment(texts: Sequence[str], model: str = SENTIMENT_MODEL) -> Dict:
    """텍스트 묶음의 금융 감성을 판정한다.

    돌려주는 `rows` 는 **입력과 같은 순서·같은 길이**다. 순서가 어긋나면 어느 공시가
    부정인지 알 수 없게 되므로, 위 2번의 `top_k=1` 규칙으로 대응을 못박아 두었다.
    """
    clean = [str(t or "").strip() for t in texts]
    wanted = [t for t in clean if t]
    if not wanted:
        return _failure("판정할 텍스트가 없다")

    key = ("sentiment", model, tuple(wanted))

    def produce() -> Dict:
        started = time.monotonic()
        rows: List[Dict] = []
        try:
            for offset in range(0, len(wanted), MAX_BATCH):
                chunk = wanted[offset:offset + MAX_BATCH]
                raw = _post(model, {"inputs": chunk, "parameters": {"top_k": 1}})
                if not isinstance(raw, list) or len(raw) != len(chunk):
                    return _failure(f"응답 모양이 예상과 다르다 (입력 {len(chunk)} · 응답 "
                                    f"{len(raw) if isinstance(raw, list) else type(raw).__name__})")
                for text, entry in zip(chunk, raw, strict=False):
                    top = entry[0] if isinstance(entry, list) and entry else entry
                    label = str((top or {}).get("label", ""))
                    rows.append({
                        "text": text,
                        "label": LABEL_MAP.get(label, label),
                        "raw_label": label,
                        "score": round(float((top or {}).get("score") or 0.0), 4),
                    })
        except HTTPError as error:
            return _failure(f"HTTP {error.code} — {error.reason}")
        except (URLError, TimeoutError) as error:
            return _failure(f"연결 실패 — {error}")
        except Exception as error:                     # 모양이 바뀌어도 리서치를 죽이지 않는다
            return _failure(f"{type(error).__name__}: {str(error)[:120]}")

        counts: Dict[str, int] = {}
        for row in rows:
            counts[row["label"]] = counts.get(row["label"], 0) + 1
        _last_attempt.update(result="ok", detail=f"{len(rows)}건 판정")
        return {
            "available": True,
            "model": model,
            "rows": rows,
            "counts": counts,
            "elapsed_sec": round(time.monotonic() - started, 2),
            "note": "금융 도메인 감성 모델이다. 공시 제목은 짧아 문맥이 없다는 한계가 있다.",
        }

    return _cached(key, produce)


# ─────────────────────────────────────────────────────────────
# 2. 제로샷 분류 — 산업 후보 중 어디에 가까운가
# ─────────────────────────────────────────────────────────────
def classify(text: str, labels: Sequence[str], model: str = ZEROSHOT_MODEL,
             multi_label: bool = False) -> Dict:
    """문장 하나를 후보 라벨들에 견준다 (제로샷).

    ⚠️ 라벨은 **한국어로** 준다. 영어 전용 모델을 쓰면 한국어를 못 읽는다는 것을
    위 4번에서 실측으로 확인했다. 이 함수는 다국어 모델을 기본으로 쓴다.
    """
    text = str(text or "").strip()
    candidates = [str(label).strip() for label in labels if str(label).strip()]
    if not text:
        return _failure("분류할 문장이 없다")
    if len(candidates) < 2:
        return _failure("후보 라벨이 둘 미만이라 분류할 것이 없다")

    key = ("classify", model, text, tuple(candidates), multi_label)

    def produce() -> Dict:
        started = time.monotonic()
        try:
            raw = _post(model, {"inputs": text,
                                "parameters": {"candidate_labels": candidates,
                                               "multi_label": multi_label}})
        except HTTPError as error:
            return _failure(f"HTTP {error.code} — {error.reason}")
        except (URLError, TimeoutError) as error:
            return _failure(f"연결 실패 — {error}")
        except Exception as error:
            return _failure(f"{type(error).__name__}: {str(error)[:120]}")

        if not isinstance(raw, dict) or "labels" not in raw:
            return _failure(f"응답 모양이 예상과 다르다 — {str(raw)[:120]}")

        pairs = list(zip(raw.get("labels", []), raw.get("scores", []), strict=False))
        _last_attempt.update(result="ok", detail=f"{len(pairs)}라벨 분류")
        return {
            "available": True,
            "model": model,
            "text": text,
            "top": pairs[0][0] if pairs else "",
            "top_score": round(float(pairs[0][1]), 4) if pairs else 0.0,
            "rows": [{"label": label, "score": round(float(score), 4)} for label, score in pairs],
            "elapsed_sec": round(time.monotonic() - started, 2),
            "note": "제로샷이라 학습된 분류기가 아니다. 후보 라벨 문구를 바꾸면 결과가 달라진다.",
        }

    return _cached(key, produce)


# ─────────────────────────────────────────────────────────────
# 3. 임베딩 — 자연어 질의를 벡터로 (M8 · 변경노트 N85)
# ─────────────────────────────────────────────────────────────
def embed(texts: Sequence[str], model: str = EMBED_MODEL) -> Dict:
    """문장 묶음을 벡터로 바꾼다.

    ⚠️ **경로가 다르다.** 감성·분류와 달리 이 모델들은 sentence-similarity 로 등록돼
    있어서 기본 경로로 부르면 400 이 난다 (실측 2026-08-04):

        POST …/models/nlpai-lab/KURE-v1
        → 400 SentenceSimilarityPipeline.__call__() missing 'sentences'

        POST …/models/nlpai-lab/KURE-v1/pipeline/feature-extraction
        → 200 [[…1024개…], …]

    **문서 쪽은 여기서 임베딩하지 않는다.** 종목 하나의 사업보고서에 표가 2천~4천 개라
    요청 안에서 돌리면 서버리스 60초를 몇 배로 넘긴다. 문서는 빌드타임에 미리 계산해
    `data/report_index.db` 에 넣어 두고(`scripts/build_report_index.py`),
    런타임에는 **질의 한 건만** 이 함수로 임베딩한다.

    실패하면 예외가 아니라 값으로 돌려준다 (이 모듈의 책임 경계) — 부르는 쪽이
    낱말 검색으로 떨어지고 그 사실을 화면에 밝힌다.
    """
    clean = [str(t or "").strip() for t in texts]
    wanted = [t for t in clean if t]
    if not wanted:
        return _failure("임베딩할 문장이 없다")

    key = ("embed", model, tuple(wanted))

    def produce() -> Dict:
        started = time.monotonic()
        vectors: List[List[float]] = []
        try:
            for offset in range(0, len(wanted), MAX_BATCH):
                chunk = wanted[offset:offset + MAX_BATCH]
                raw = _post(f"{model}{EMBED_PATH}", {"inputs": chunk})
                if not isinstance(raw, list) or len(raw) != len(chunk):
                    return _failure(f"응답 모양이 예상과 다르다 (입력 {len(chunk)} · 응답 "
                                    f"{len(raw) if isinstance(raw, list) else type(raw).__name__})")
                for item in raw:
                    if not isinstance(item, list) or not item:
                        return _failure("응답 안의 벡터 모양이 예상과 다르다")
                    vectors.append([float(x) for x in item])
        except HTTPError as error:
            return _failure(f"HTTP {error.code} — {error.reason}")
        except (URLError, TimeoutError) as error:
            return _failure(f"연결 실패 — {error}")
        except Exception as error:
            return _failure(f"{type(error).__name__}: {str(error)[:120]}")

        _last_attempt.update(result="ok", detail=f"{len(vectors)}건 임베딩")
        return {
            "available": True, "model": model, "texts": wanted, "vectors": vectors,
            "dim": len(vectors[0]) if vectors else 0,
            "elapsed_sec": round(time.monotonic() - started, 2),
            "note": "문서 쪽 벡터는 빌드타임에 미리 계산한다 — 여기서는 질의만 임베딩한다.",
        }

    return _cached(key, produce)


def get_status() -> Dict:
    """인증키 상태 진단 (토큰 값은 노출하지 않는다)."""
    token, source = load_hf_key()
    return {
        "configured": bool(token),
        "source": source,
        "masked": secrets.mask(token) if token else "",
        "endpoint": HF_BASE_URL,
        "models": {"sentiment": SENTIMENT_MODEL, "zeroshot": ZEROSHOT_MODEL,
                   "embed": EMBED_MODEL},
        "last_result": _last_attempt.get("result"),
        "last_detail": _last_attempt.get("detail"),
        "note": ("원격 추론이라 transformers·torch 를 설치하지 않는다 (배포 번들 한도). "
                 "실패해도 리서치는 규칙 기반으로 계속 돈다."),
    }
