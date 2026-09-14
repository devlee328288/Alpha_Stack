# 핵심코드 ③ — 공급 정문: `as_of` 가 미래를 막는 방법

> 줄 단위 해설입니다. 정본은 [`supply/`](../../../supply/__init__.py) 네 모듈입니다.
> 아키텍처에서의 위치: [아키텍처 v1.2](../../아키텍처/version1.2/아키텍처.md) ·
> 🆕 v1.2 (2026-09-07): §6 크기가 아니라 어긋남 — `supply.adj_quality`

피처·모델·평가는 DB 를 직접 읽지 않고 **`supply/` 문 하나**를 지납니다.
이 문의 규칙은 단 하나입니다 — **`as_of`(언제 시점에서 보나)를 내지 않으면 열리지
않는다.** 왜 이렇게까지 하는지부터 봅니다.

## 0. 왜 문이 필요한가 — 미래는 예외를 던지지 않는다

저장소에는 **오늘까지의** 자료가 들어 있습니다. 2020년 폴드를 학습하면서 표를
그대로 쓰면 2021~2026년 값이 함께 들어가고, **아무 에러도 나지 않습니다.**
성능만 좋아집니다. 좋아 보이는 쪽으로 틀리는 버그는 사람이 못 잡습니다.
그래서 규칙을 문서에 적는 대신 **코드가 강제**합니다 — `as_of` 는 기본값이 없어서
빠뜨리면 그 자리에서 터집니다. 게다가 `features/`·`models/`·`evaluation/` 이
`ingest` 를 직접 import 하면 **테스트가 실패**합니다(`tests/test_supply_boundary.py`).

## 1. "언제부터 알 수 있었나" — `known_at`

정본: [`supply/clock.py`](../../../supply/clock.py)

```python
def known_at(bas_dd: str) -> datetime:
    """거래일 YYYYMMDD 의 시세를 언제부터 알 수 있었나."""
    if len(bas_dd) != 8 or not bas_dd.isdigit():
        raise ValueError(f"거래일은 YYYYMMDD 여야 한다: {bas_dd!r}")
    day = date(int(bas_dd[:4]), int(bas_dd[4:6]), int(bas_dd[6:]))
    return datetime.combine(day + timedelta(days=1), time.min, tzinfo=KST)
```

거래일 T 의 시세는 **T+1 의 0시(KST)부터** 알 수 있었다고 봅니다. 근거는 실측입니다 —
2026-08-26 에 장 마감(15:30) 40분 뒤인 16:10 에 당일 자료를 요청하니 **0행**이었습니다.
정확히 몇 시에 올라오는지는 재지 않았고, **재지 않은 값을 가정으로 쓰지 않으므로**
하루를 통째로 미룹니다. 이 선택은 항상 진실보다 늦은 쪽이라, 틀려도 성능을
부풀리는 방향으로는 틀리지 않습니다.

> ⚠️ 이 값을 앞당기고 싶으면 **실측부터** 합니다. 앞당기는 방향이 곧 누수 방향입니다.

## 2. 표기 함정 — 하이픈은 0 보다 작다

정본: [`supply/clock.py`](../../../supply/clock.py) 의 `as_bas_dd`

```python
min('2026-08-21', '20260825')   # → '2026-08-21'  (뜻과 무관하게 항상 하이픈 쪽이 이긴다)
```

`'-'`(0x2D)가 `'0'`(0x30)보다 작아서, ISO 표기와 `YYYYMMDD` 표기를 그냥 비교하면
**답이 표기 순으로 정해집니다.** 그 값이 `bas_dd <= ?` 에 들어가면 결과가 0행이
되는데 예외는 안 납니다 — 빈 표를 받은 쪽은 "그 구간에 자료가 없구나"로 읽습니다.
그래서 문에 들어오는 날짜는 전부 `as_bas_dd()` 로 `YYYYMMDD` 하나로 맞춥니다.
표기를 하나로 만들면 이 실수 자체가 불가능해집니다.

## 3. 문이 둘인 이유 — 예측 경로와 학습 경로

정본: [`supply/market.py`](../../../supply/market.py) · [`supply/training.py`](../../../supply/training.py)

| 문 | 언제 | 무엇이 다른가 |
|---|---|---|
| `price_series` · `index_series` | **예측할 때** | `as_of` 시점에 알 수 있었던 것만. 미래를 절대 안 준다 |
| `training_frame` | **학습할 때** | 라벨(미래 수익률)을 만들기 위해 **여기서만** 미래를 본다 |

처음엔 한 함수에 `include_future=True` 같은 손잡이를 두는 안이 있었는데,
**손잡이는 언젠가 켜진 채로 지나갑니다.** 그래서 함수 이름 자체를 갈랐습니다 —
코드 리뷰에서 `training_frame` 이 예측 경로에 있으면 이름만 보고 잡을 수 있습니다.

학습 경로 안에도 봉인이 있습니다. `training_frame(code, *, holdout_start=...)` 의
`holdout_start` 는 **키워드 전용이고 기본값이 없어서** 빠뜨리면 그 자리에서 터지고,
값을 주면 그 날짜 이후 행을 잘라낸 뒤 **몇 행을 잘랐는지**(`dropped["holdout"]`)를
함께 돌려줍니다. 전 구간이 필요하면 `holdout_start=None` 을 **명시적으로** 적어야
합니다 — "깜빡해서 전 구간"이 불가능한 모양입니다.

> ⚠️ 옛 문서(요구사항 F-04 · 데이터파트 v2.1)에는 봉인이 `SealedRangeError` 예외로
> 구현된 것처럼 적혀 있는데, **그 예외는 코드에 없습니다**(2026-09-02 grep 전수).
> 실물은 위의 "기본값 없는 키워드 인자 + 행 제거 + 제거량 보고" 방식입니다.

## 4. 흐름 한 장

```mermaid
flowchart LR
    subgraph 내부["ingest/ (내부 계층 — 직접 import 금지)"]
        DB[(krx_cache.db)]
    end
    subgraph 문["supply/ 정문"]
        CK["clock.py<br/>known_at · as_bas_dd"]
        P["price_series<br/>(예측 — as_of 필수)"]
        T["training_frame<br/>(학습 — 봉인 검사)"]
    end
    F[features/] --> P
    M[models/] --> T
    E[evaluation/] --> P
    DB --> CK --> P
    CK --> T
    T -.->|"홀드아웃 요청 시"| X["SealedRangeError 🔴"]
```

## 5. 이 설계가 지키는 약속 (테스트가 못박은 것)

- `as_of` 없이 부르면 터진다 — `tests/test_supply_boundary.py`
- 상류 계층이 `ingest` 를 import 하면 테스트가 실패한다 — 같은 파일
- 빈 결과에도 컬럼 구조가 남는다 — `tests/test_supply_price.py`
  (빈 DataFrame 에 컬럼이 없으면 하류의 `df["close"]` 가 다른 이유로 터져 원인을 가린다)

---

## 6. 🆕 크기가 아니라 어긋남 — `supply.adj_quality` (v1.2 · 2026-09-07)

정본: [`supply/adj_quality.py`](../../../supply/adj_quality.py) · PR #156 · 이슈 #132 ·
[품질 규약 §4.4 · §6.7](../../데이터파트/version3.9/데이터_품질_규약.md) ·
재현 [노트북 06/10](../../../notebooks/06-검토·발견/10.이상치는-크기가-아니라-어긋남으로-가른다.ipynb)

공급 정문에 문이 하나 더 생겼습니다. 이번 문은 시점이 아니라 **품질**을 묻습니다 — *"이 행의 수정주가를
믿어도 되나?"* 모델 파트가 개별종목 후보군에 `|adj_close 일간수익률| > 100%` 필터를 걸었더니 딱 한 행이
지워졌습니다(경남에너지 2016-05-11 · +153.66%). 그런데 KRX 가 그날 발표한 등락률도 정확히 +153.66%
였습니다 — 오류가 아니라 **실제로 그렇게 움직인 날**입니다. 반대로 신세계 2011-06-10 은 우리 수정주가
수익률이 −60.6% 인데 KRX 등락률은 +14.95% 입니다 — 인적분할 재상장일에 우리 조정계수가 틀린 것이고,
**크기 필터에는 걸리지 않습니다.**

"이상치" 라는 말은 **데이터 오류**와 **진짜 사건**을 섞습니다. 그래서 판별자를 크기에서 **독립 원천과의
어긋남**으로 바꿨습니다. KRX `change_rate` 는 거래소가 그날 기준가 대비로 낸 공식 수익률이라, 우리가 만든
수정주가를 검증할 독립 원천입니다.

```python
#: 행 플래그의 허용폭(%p). 이보다 크게 어긋나면 수정주가 오류 의심이다.
SUSPECT_GAP_TOLERANCE = 1.0
#: 극단 수익률의 경계(%). 가격제한폭(2015-06-15 이후 ±30%)과 같다.
EXTREME_RETURN_PCT = 30.0
#: 입력에 있어야 하는 열. HF 반출본 `daily_price_dev.parquet` 에 전부 있다.
REQUIRED_COLUMNS = ("bas_dd", "code", "close", "adj_close", "change_rate")
#: 돌려주는 열. 순서가 곧 계약이다.
FLAG_COLUMNS = ("adj_return_1d", "adj_change_rate_gap",
                "is_adj_suspect", "is_extreme_return")
```

| 칸 | 정의 |
|---|---|
| `adj_return_1d` | (`adj_close` / 전일 `adj_close` − 1) × 100 |
| `adj_change_rate_gap` | \|`adj_return_1d` − `change_rate`\| (%p) |
| `is_adj_suspect` | gap > 1%p → 수정주가 오류 의심 · **학습에서 뺀다** |
| `is_extreme_return` | \|r\| > 30% 이고 의심 아님 → 진짜 극단 사건 · **남긴다** |

### 6.1 왜 공급 정문에 있나 — 시점 규칙이 같기 때문

플래그는 그 종목의 **전날과 당일 값만** 봅니다. 뒤의 행을 잘라 내도 앞 행의 플래그는 같습니다(시험
`test_뒤_행을_잘라도_앞_행의_플래그는_같다`). 그래서 개발구간 학습과 개봉 뒤 홀드아웃 예측에 **같은
함수를 그대로** 씁니다 — §3 의 "문이 둘" 과 같은 태도입니다. T+1 이후에 극단값이 나왔다는 이유로 T 를
지우는 일은 여기서 일어날 수 없습니다. 수정주가 자체는 후방조정이지만 **이웃한 두 날의 비율**은 그 뒤에
무슨 분할이 오든 변하지 않습니다.

```python
def flag_adjustment_quality(daily_prices, *, gap_tolerance=SUSPECT_GAP_TOLERANCE,
                            extreme_pct=EXTREME_RETURN_PCT) -> pd.DataFrame:
    """행마다 칸 넷을 매긴다. **입력과 같은 인덱스·같은 순서**로 돌려준다.
    수익률은 종목의 전체 시계열에서 이웃한 두 날로 계산한다. 후보만 잘라 낸 뒤 계산하면
    후보에 안 뽑힌 전날이 빠져 며칠짜리 수익률이 되므로, 항상 원천 전체를 넘긴다.
    """
```

비교할 수 없는 행(종목의 첫 행 · 전일 adj 가 없거나 0 · 종가 0 · 등락률 없음)은 수익률·갭이 NaN 이고
두 플래그는 False 입니다 — **모르는 것을 오류로 치지 않습니다.**

### 6.2 허용폭이 `verify()` 와 다른 이유

`build_adj_prices.GAP_TOLERANCE` 는 0.15%p 입니다. 그것은 **전체 어긋남 비율이 1% 미만인가**를 보는 집계
게이트라 민감한 게 맞습니다. 행 플래그는 다릅니다 — 반올림 잡음을 오류로 표시하면 멀쩡한 저가주 행이
학습에서 빠집니다.

```
허용폭     전 시장 어긋남      후보군(업종 10 × 종목 5 · 176,705행) 교집합
0.15%p    13,677 (0.173%)     2
1.0%p      1,406 (0.018%)     2
30%p         144               2
```

0.15~1.0 사이 12,271행의 **94% 가 종가 5,000원 미만**입니다 — FDR 수정가격이 원 단위로 반올림되므로
저가주에서 0.2~0.5%p 는 반올림에서 나옵니다. 후보군에서는 허용폭이 무엇이든 같은 2행(둘 다 인적분할
재상장일)만 남습니다. 그 2행을 좇아 **재개일 규칙**을 고쳤고(규약 §6.7 · 51종 재생성), 고친 뒤 후보군
의심 행은 **0** 입니다.

### 6.3 쓰는 법 — 오준영 님 필터와 같은 서명

```python
from supply.adj_quality import attach_adjustment_quality
cand = attach_adjustment_quality(candidates, daily)   # 후보 프레임에 칸 넷이 붙는다
train = cand[~cand["is_adj_suspect"]]                 # 의심 행만 뺀다. 진짜 사건은 남는다
print(cand.attrs["adjustment_quality"])               # 몇 행을 왜 뺐는지 기록
```

`daily` 는 HF 반출본 `daily_price_dev.parquet` 그대로면 됩니다 — `adj_close` 와 `change_rate` 가 이미
들어 있어 재배포가 필요 없습니다. ⚠️ 거꾸로 말하면 **플래그 칸 넷은 반출본에 없습니다.** 팀원이 같은
함수를 돌려야 같은 표본이 됩니다. 그것을 게이트가 세어 카드에 게시하자는 것이
[품질 원장 설계](../../데이터파트/version3.9/데이터_품질_원장_설계.md)입니다.
