"""수정주가 적재 — FDR 이 준 구간과 우리가 이어 붙인 구간 (저장소 계층)

`daily_price.adj_open`·`adj_high`·`adj_low`·`adj_close` 를 채운다.

## 왜 두 가지 출처를 섞나

FDR(네이버 fchart)은 **최근 3,000거래일만** 준다(20140613~). 우리 달력은 4,102일이라
**20100104~20140612 의 1,103일(2,146,042행·23.3%)** 이 비고, 홀드아웃이 20240901 이므로
그 구멍은 **전부 학습구간 안**이다. 그렇다고 그 구간을 미조정 원가격으로 두면 #51 이
지적한 -98% 가 그대로 남는다.

그래서 **FDR 이 닿는 가장 이른 날을 앵커로 삼아, 그 앞을 우리 조정계수로 이어 붙인다.**
계수 계산은 이미 `common/corporate_actions.py` 에 있다(평상일은 기준가, 재개일은
상장주식수 배율 — 그 파일의 주석에 왜 둘로 나뉘는지가 있다).

    20100104 ────────────── 20140612 │ 20140613 ────────── 20260901
      chain (계수로 뒤로 이어 붙임)   │      fdr (외부 실측)
                                     ↑
                              여기가 앵커. 이 날의 배율을
                              FDR 이 알려 주므로 그 앞이 정해진다

칸 하나(`adj_source`)로 어느 쪽인지 행마다 남긴다. **날짜로는 유추할 수 없다** —
2012년에 상장폐지된 종목은 FDR 이 아예 없어 전 구간이 `chain` 이고, 2020년 상장 종목은
전 구간이 `fdr` 다.

## 배율을 어떻게 옮기나

`scale[i] = adj_close[i] / close[i]` — 원가격을 수정가격으로 바꾸는 배율이다.
앵커에서는 FDR 이 알려 준다. 거기서 양쪽으로 퍼뜨린다:

    뒤로 (과거 방향)   scale[i]   = scale[i+1] * factor[i+1]
    앞으로 (미래 방향) scale[i]   = scale[i-1] / factor[i]

`factor[i]` 는 **그 날 일어난 조정**이고 **그 앞의 행들에** 적용된다 — 그래서 과거로 갈 때
곱하고 미래로 갈 때 나눈다. (`corporate_actions.back_adjusted_closes` 가 같은 방향으로 돈다.)

⚠️ **배율은 `Fraction` 으로 옮기고 마지막에 한 번만 `float` 한다.** 1/50 · 1/3 같은 계수가
   수천 행에 걸쳐 곱해지므로 부동소수로 누적하면 반올림 잡음이 되돌아온다.

## 왜 O/H/L 에 같은 배율을 쓰나

하루 안의 시·고·저·종가는 **같은 스케일**이다. 배율 하나를 넷에 똑같이 곱하면 그 날의
고저 폭과 시종 관계가 정확히 보존된다. 넷을 따로 조정하면 `low ≤ close ≤ high` 가
반올림 때문에 깨질 수 있다.

## 정지일

`open=high=low=0` 인 행이 거래정지다(종가만 직전 값을 물고 있다). 그 0 에 배율을 곱하면
0 이 되어 **"그 날 가격이 0원이었다"** 로 읽힌다. 시·고·저가는 `None` 으로 두고 종가만
채운다 — FDR 도 정확히 그렇게 준다.
"""

from __future__ import annotations

import sqlite3
from datetime import datetime
from fractions import Fraction
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

from common.corporate_actions import SHARE_RATIO_MIN, factor_series, is_halted
from ingest.clients.fdr_data import SOURCE_FDR
from ingest.store.sqlite_db import write_lock

#: 우리가 계수로 이어 붙인 행의 표시. `daily_price.adj_source` 에 들어간다.
SOURCE_CHAIN = "chain"

#: 소스 뒤에 붙는 표시 — **FDR 이 안 편 자본변동을 우리가 폈다** (`scale_series` ⑤).
#: `fdr+ca_fix` 처럼 붙는다. 값을 정한 것은 여전히 FDR 이지만 그 값을 우리가 고쳤다는
#: 사실이 남아야, 나중에 "이 종목 왜 FDR 과 다르지" 에 답할 수 있다.
SOURCE_CA_FIX = "+ca_fix"

#: 수집 대장에 남길 출처 이름. 종목 하나가 대장 한 줄이다.
#:
#: 왜 대장에 남기나: 후방조정 값은 **새 분할이 생기면 과거 전체가 바뀐다.** 그래서
#: "이 값을 언제 계산했나" 가 값 자체만큼 중요하다. 그렇다고 9.2M 행마다 시각을 적으면
#: 그것만 180MB 다 — 종목별로 한 줄이면 3,677줄이고 같은 질문에 답한다.
COLLECT_SOURCE = "adj_price"

#: `daily_price` 에서 계수 계산에 필요한 칸. `corporate_actions` 가 요구하는 것과 같다.
PRICE_COLUMNS = ("bas_dd", "open", "high", "low", "close", "change",
                 "volume", "listed_shares")


def load_rows(conn: sqlite3.Connection, code: str) -> List[Dict]:
    """한 종목의 전 구간 원가격 행. `bas_dd` 오름차순.

    ⚠️ **그 종목의 전부**여야 한다. 계수는 앞 행과의 관계로 정해지므로 중간을 잘라
       읽으면 잘린 자리에서 조정이 하나 사라지고, 그 뒤가 통째로 어긋난다.
    """
    cursor = conn.execute(
        f"SELECT {','.join(PRICE_COLUMNS)} FROM daily_price WHERE code = ? "
        "ORDER BY bas_dd",
        (code,),
    )
    names = [d[0] for d in cursor.description]
    return [dict(zip(names, row, strict=True)) for row in cursor.fetchall()]


#: ⑤ 판정 임계 둘. 실측으로 정했다 (2026-09-04 · 자본변동 크기 41자리 전수).
#:
#:   원가격이 계수만큼 튀었나   `close` 비율이 `factor` 에서 이만큼 안쪽
#:   FDR 이 안 폈나             `scale` 비율이 1 에서 이만큼 안쪽
#:
#: 안 편 17자리는 `|scale비 - 1|` 이 전부 **0.011 이하**였고, 이미 편 24자리는 그 값이
#: 0.24 이상이거나(폈다) 계수 자체가 틀린 경우였다(아래).
CA_PRICE_TOLERANCE = 0.30
CA_SCALE_TOLERANCE = 0.10

#: ③ 판정 임계. `실제` 가 **KRX 기준가비**와 이만큼 안쪽이면 FDR 이 이미 편 것이다.
#:
#: 2026-09-05 에 전 종목을 정본 코드로 다시 깔았더니 관문 ①②만으로는 못 거르는 자리가
#: 하나 나왔다 — 삼양제넥스(003940) 2013-03-25. 인적분할인데 감자로 오인해 과거
#: **803행을 1.567배 부풀렸다.**
#:
#:     주식수 2,985,917 → 1,905,907  (0.638배)  → 계수 1.5667
#:     그런데 KRX 기준가는 70,500 — 직전 종가 68,900 의 **1.0232배**밖에 안 된다
#:     FDR 이 그 날 실제로 편 폭도 **1.0232** — 소수 다섯 자리까지 같다
#:
#: 즉 FDR 은 사건을 알고 KRX 기준가만큼 정확히 폈다. 인적분할이니 그게 맞는 조정이고,
#: **틀린 것은 관문이 아니라 계수**다 — 재개일이라 주식수 배율에서 얻는데, 인적분할은
#: 주식수가 줄어도 가격은 그대로다. 관문 ②는 "FDR 이 안 폈나" 를 1 에서 ±0.10 으로
#: 보는데 FDR 이 편 폭이 2.3% 라 그 창 안에 들어왔다.
#:
#: 임계는 재고 나서 적었다. 발동한 22자리 전부에서 `|실제/기준가비 - 1|` 을 쟀다.
#:
#:     003940              0.0000   ← FDR 이 기준가만큼 폈다
#:     가장 가까운 정상 자리  0.5000   (009415 태영건설우 20240712)
#:
#: 0.0 과 0.5 사이가 통째로 비어 있다. 양쪽에 5배씩 여유를 두고 낮은 쪽에 붙였다 —
#: 정상 자리를 잃는 것이 오검출보다 비싸기 때문이다(조정을 놓치면 수익률이 틀리지만,
#: 없는 조정을 만들면 멀쩡한 과거 전체를 망친다).
#:
#: ⚠️ 관문 ①②로는 못 가른다. 실측에서 여유가 각각 1.02배·1.36배뿐이라, 임계를 어디에
#:    둬도 004555(진짜 감자인데 정지 중이라 기준가가 무의미한 자리)를 같이 잃는다.
CA_BASIS_TOLERANCE = 0.10


def _fix_unadjusted_actions(rows: Sequence[Mapping],
                            scales: List[Optional[Fraction]],
                            factors: Sequence[Fraction],
                            fixed: set) -> None:
    """🔴 **FDR 이 안 편 자본변동**을 편다. `scales` 를 제자리에서 고친다.

    ## 왜 필요한가 — FDR 은 감자를 조정하지 않는다 (실측 2026-09-04)

    `scale_series` ①이 FDR 값을 심고 ③④가 **빈 자리만** 채우므로, 자본변동 **양쪽에
    FDR 값이 다 있으면** 그 불연속이 그대로 남는다. FDR 은 액면분할은 조정하는데
    감자는 조정하지 않아 이 자리가 생긴다.

        전 종목 대조: 주식수가 2배 이상 변한 2,048자리 중 **17자리**에서
        `adj_close` 가 원가격과 똑같이 뛰었다 (`scripts/verify_base_info.py` §9)

    ⚠️ **액면가로는 안 보인다.** 액면분할·병합은 액면가가 같이 바뀌지만 감자는 액면가가
       그대로다. 그래서 판정은 액면가가 아니라 **주식수 배율**로 한다.

    ## 관문이 셋이다 — 하나만 보면 멀쩡한 값을 망친다

    **① 원가격이 계수만큼 실제로 튀었나** (`CA_PRICE_TOLERANCE`)

    `factors[i]` 를 그대로 믿으면 안 된다. 재개일에는 계수를 **상장주식수 배율**에서
    얻는데(`adjustment_factor`), 주식수가 움직여도 가격은 연속인 사건이 있다.

        009415 태영건설우 20200922  주식수 ×0.51 인데 가격은 32,500 → 기준가 16,250
                                    인적분할이라 가격도 같이 쪼개졌다. 조정할 것이 없다
        009410 태영건설   20240722  주식수 ×24.9 인데 KRX 등락률 0.00% — 출자전환이다

    둘 다 계수가 틀렸다. 그래서 **원문 종가가 그 계수만큼 튀었는지**를 먼저 본다.
    진짜 자본변동이면 `close` 가 계수를 따라간다(감자 10:1 이면 가격이 10배).

    **② 그런데 `scale` 은 그것을 반영하지 않았나** (`CA_SCALE_TOLERANCE`)

    ③이 전파하는 규칙이 `scales[i-1] == scales[i] * factors[i]` 다. 그러므로

        scale 비율이 1 에 가깝다  → FDR 이 **아무것도 안 했다** — 우리가 편다
        그 밖                     → 이미 폈거나 다른 값을 썼다. 건드리지 않는다

    실측에서 안 편 17자리는 `|scale비 - 1|` 이 전부 0.011 이하였고, 이미 편 자리는
    0.24 이상이었다. 둘 사이가 20배 넘게 벌어져 있어 임계를 어디에 두든 같은 답이 된다.

    **③ FDR 이 편 폭이 마침 KRX 기준가가 움직인 폭과 같지는 않나** (`CA_BASIS_TOLERANCE`)

    ②는 "1 에서 얼마나 먼가" 만 본다. 그런데 **KRX 가 조정한 폭 자체가 작은 사건**이
    있다 — 인적분할은 주식수가 줄어도 가격이 거의 안 움직인다. 그러면 FDR 이 제대로
    폈는데도 `실제` 가 1 근처라 ②를 통과한다.

        003940 삼양제넥스 20130325  주식수 ×0.638 → 계수 1.5667
                                    그런데 KRX 기준가는 70,500 (직전 종가의 1.0232배)
                                    FDR 이 편 폭도 1.0232 — 다섯 자리까지 같다

    FDR 이 KRX 기준가만큼 폈다면 **틀린 것은 FDR 이 아니라 우리 계수**다. 건드리지
    않는다. 이 관문이 없던 2026-09-05 새벽 전 종목 적재에서 이 한 자리가 통과해
    과거 803행이 1.567배로 부풀었다.

    ## 자본변동 크기만 본다

    권리락·주식배당(×0.9~0.99)은 FDR 이 제대로 편다. 여기서 다루는 것은 `≥1.5배` 또는
    `≤2/3배` 뿐이다 — `SHARE_RATIO_MIN` 을 `corporate_actions` 와 공유한다.

    ## 오름차순이라 여러 번이 누적된다

    앞에서부터 고치므로 자본변동이 둘 이상인 종목도 자연히 누적된다. 아센디오(012170)는
    2025-03-06 에 10:1, 2026-08-28 에 5:1 을 겪어 가장 오래된 구간이 ×50 이 된다.

    🔴 이 누적이 손으로는 틀리는 자리다. 2026-09-04 에 스크래치 스크립트로 한 번 보정한
       적이 있는데, 두 번째 자리를 처리하며 **첫 자리 당일(20250306) 하루를 빠뜨려**
       그 하루만 배율이 어긋났다(다음날 대비 +312.6%). 그 결함이 이 함수를 쓰는 이유다 —
       경계를 손으로 잡지 않고 규칙 하나로 전파한다.
    """
    for i in range(1, len(scales)):
        factor = factors[i]
        if factor == 1:
            continue
        if not (factor >= SHARE_RATIO_MIN or factor <= 1 / SHARE_RATIO_MIN):
            continue                        # 권리락 같은 잔 조정은 FDR 이 편다

        # ① 원문 종가가 그 계수만큼 튀었나 — 계수 자체를 원가격으로 검산한다.
        앞종가, 종가 = rows[i - 1]["close"], rows[i]["close"]
        if not 앞종가 or not 종가:
            continue
        원가격비 = Fraction(int(종가), int(앞종가))
        if abs(float(원가격비 / factor) - 1) > CA_PRICE_TOLERANCE:
            continue                        # 가격이 안 따라갔다 — 조정 사건이 아니다

        # ② FDR 이 그것을 반영했나.
        앞, 뒤 = scales[i - 1], scales[i]
        if 앞 is None or not 뒤:
            continue                        # 모르는 자리는 만들지 않는다
        실제 = 앞 / 뒤
        if abs(float(실제) - 1) > CA_SCALE_TOLERANCE:
            continue                        # 이미 폈다 (또는 우리가 모르는 값을 썼다)

        # ③ 🔴 FDR 이 이미 **KRX 기준가만큼** 폈나 — 그러면 틀린 것은 계수다.
        #    ②는 "1 에서 얼마나 먼가" 만 보므로, KRX 가 조정한 폭 자체가 작으면
        #    (인적분할처럼 가격이 거의 안 움직이는 사건) 그 창 안에 들어와 버린다.
        전일대비 = rows[i].get("change")
        if 전일대비 is not None:
            기준가 = 종가 - 전일대비
            if 기준가 > 0:
                기준가비 = Fraction(int(기준가), int(앞종가))
                if (기준가비 != 1
                        and abs(float(실제 / 기준가비) - 1) <= CA_BASIS_TOLERANCE):
                    continue                # 조정은 이미 끝났다 — 계수가 틀린 것이다

        보정 = factor / 실제
        for j in range(i):
            if scales[j] is not None:
                scales[j] *= 보정
                fixed.add(j)


def scale_series(rows: Sequence[Mapping],
                 fdr_close: Mapping[str, Optional[float]],
                 *, fixed: Optional[set] = None) -> List[Optional[Fraction]]:
    """행마다 `수정가격 / 원가격` 배율. FDR 이 아는 날에서 시작해 양쪽으로 퍼뜨린다.

    `fdr_close` 는 `{YYYYMMDD: 수정종가}`. 값이 `None` 이거나 그 날이 없으면 모르는 날이다.

    **FDR 이 한 날도 없으면** 마지막 행을 배율 1 로 잡고 뒤로만 퍼뜨린다 — 그 종목을
    자기 마지막 가격 기준으로 이어 붙인 것이 된다. 2014년 이전에 사라진 종목이 그렇다.
    이때도 분할은 제대로 펴지고, 다만 **스케일의 절대 수준이 FDR 종목과 다르다** —
    종목 간 가격 비교가 아니라 종목 안의 수익률을 위한 값이라 문제가 되지 않는다.

    `fixed` 를 주면 ⑤에서 **우리가 고친 행의 인덱스**를 그 집합에 넣는다. 부르는 쪽이
    `adj_source` 에 `+ca_fix` 를 붙이는 데 쓴다.
    """
    n = len(rows)
    if n == 0:
        return []

    factors = factor_series(rows)
    scales: List[Optional[Fraction]] = [None] * n

    # ① FDR 이 아는 날을 그대로 심는다.
    for i, row in enumerate(rows):
        close = row["close"]
        if not close:                       # 0 이나 None 이면 배율을 정의할 수 없다
            continue
        adj = fdr_close.get(row["bas_dd"])
        if adj is None:
            continue
        # 둘 다 정수 자릿수를 가진 값이라 유리수로 정확히 잡을 수 있다.
        scales[i] = Fraction(adj).limit_denominator(10 ** 12) / Fraction(int(close))

    # ② 한 날도 없으면 마지막 행을 1 로 잡는다 (자기 기준 후방조정).
    if not any(s is not None for s in scales):
        scales[n - 1] = Fraction(1)

    # ③ 과거 방향 — 그 날의 조정은 **그 앞** 행들에 적용된다. 그래서 곱한다.
    for i in range(n - 2, -1, -1):
        if scales[i] is None and scales[i + 1] is not None:
            scales[i] = scales[i + 1] * factors[i + 1]

    # ④ 미래 방향 — 반대이므로 나눈다. 계수가 0 이면 나눌 수 없으니 그대로 물려준다
    #    (기준가가 0 인 행은 조정이 아니라 자료 결손이다 — 없는 조정을 만들지 않는다).
    for i in range(1, n):
        if scales[i] is None and scales[i - 1] is not None:
            factor = factors[i]
            scales[i] = scales[i - 1] / factor if factor else scales[i - 1]

    # ⑤ FDR 이 **안 편** 자본변동을 편다 (감자 17자리 · 2026-09-04).
    #    ③④는 빈 자리만 채우므로 FDR 값 사이의 불연속은 여기서만 고쳐진다.
    _fix_unadjusted_actions(rows, scales, factors,
                            fixed if fixed is not None else set())

    return scales


def build_rows(rows: Sequence[Mapping],
               adjusted: Mapping[str, Mapping[str, Optional[float]]]
               ) -> List[Tuple]:
    """DB 에 쓸 `(adj_open, adj_high, adj_low, adj_close, adj_source, bas_dd)` 목록.

    **네 칸 전부 `원가격 × 배율` 로 만든다.** FDR 이 준 날도 마찬가지다 — 다만 그 날의
    배율을 FDR 종가가 정하므로 `adj_close` 는 FDR 값과 정확히 같아진다
    (`scale = fdr종가 / 원종가` 이므로 `원종가 × scale = fdr종가`).

    🔴 **FDR 의 시·고·저가를 그대로 싣지 않는 이유 — 실측으로 드러났다.**
    -------------------------------------------------------------------
    FDR 은 네 칸을 **각각 따로 반올림**한다. 그래서 원문에서 `close == high` 인 날에
    수정값이 `adj_high < adj_close` 로 뒤집힌다.

        20150127 삼성전자  원문 high 1,400,000  close 1,400,000   (같다)
                           FDR  adj_high  27,999  adj_close 28,000  ← 1원 뒤집힘

    표본 3종에서만 68행이 이렇게 걸렸고 **전부 `fdr` 행**이었다(원가격 위반은 0행).
    `true_range`·`parkinson_20` 처럼 고저 폭을 쓰는 피처는 여기서 음수가 나온다.

    배율 하나를 넷에 똑같이 곱하면 그 날의 고저 폭과 시종 관계가 **정확히 보존된다.**
    위 예에서 `1,400,000 × 0.02 = 28,000` 이라 `adj_high` 가 `adj_close` 와 같아진다.
    시가·저가는 FDR 값과 어차피 일치했다 — 뒤집히던 칸만 제자리를 찾는다.

    ⚠️ 그래도 `adj_source` 는 `fdr` 이다. 값을 정한 것이 FDR 종가이기 때문이다.
       "FDR 이 준 네 칸" 이 아니라 **"FDR 이 정한 배율"** 이 이 행의 출처다.

    🔴 FDR 이 **안 편 자본변동**(감자)을 우리가 편 행은 `+ca_fix` 가 붙는다
       (`fdr+ca_fix` · `chain+ca_fix`). 판정과 이유는 `_fix_unadjusted_actions`.
    """
    fdr_close = {day: value.get("adj_close") for day, value in adjusted.items()}
    fixed: set = set()
    scales = scale_series(rows, fdr_close, fixed=fixed)

    out: List[Tuple] = []
    for i, row in enumerate(rows):
        scale = scales[i]
        if scale is None:
            continue                        # 배율을 모르는 행은 **비워 둔다**

        bas_dd = row["bas_dd"]
        factor = float(scale)
        # 정지일은 시·고·저가가 0 이다. 0 × 배율 = 0 을 실으면 "그 날 0원" 이 된다.
        halted = is_halted(row)
        values = tuple(
            None if (halted and column != "close") or not row[column] or row[column] <= 0
            else row[column] * factor
            for column in ("open", "high", "low", "close")
        )
        source = SOURCE_FDR if fdr_close.get(bas_dd) is not None else SOURCE_CHAIN
        if i in fixed:
            # 값을 정한 것은 FDR·계수지만 그 뒤 우리가 자본변동을 폈다 — 둘 다 남긴다.
            source += SOURCE_CA_FIX
        out.append(values + (source, bas_dd))
    return out


def save(conn: sqlite3.Connection, code: str, rows: Sequence[Tuple]) -> int:
    """계산한 수정주가를 그 종목의 행에 채운다. **원가격 칸은 건드리지 않는다.**

    `UPDATE` 를 쓰는 이유: 행은 이미 있다. `INSERT OR REPLACE` 로 넣으면 행을 지우고
    새로 만들어 원가격이 통째로 날아간다 (`krx_store.UPSERT_SQL` 주석과 같은 함정).

    ⚠️ **쓰기 자물쇠는 부르는 쪽이 쥔다.** 여기서 잡으면 종목 3,677개를 한 트랜잭션으로
       묶으려는 적재기가 자물쇠를 두 번 잡게 되고, `threading.Lock` 은 재진입이 안 되므로
       거기서 멈춰 선다 — 예외도 없이 그냥 멈춘다.
    """
    if not rows:
        return 0
    conn.executemany(
        "UPDATE daily_price SET adj_open=?, adj_high=?, adj_low=?, adj_close=?, "
        "adj_source=? WHERE bas_dd=? AND code=?",
        [row + (code,) for row in rows],
    )
    return len(rows)


def build_and_save(conn: sqlite3.Connection, code: str,
                   adjusted: Mapping[str, Mapping[str, Optional[float]]]) -> int:
    """한 종목을 계산해 저장한다. 채운 행 수를 돌려준다."""
    rows = load_rows(conn, code)
    if not rows:
        return 0
    return save(conn, code, build_rows(rows, adjusted))


# ==================================================
# 실측 거래일 달력
# ==================================================
def rebuild_calendar(conn: sqlite3.Connection) -> int:
    """`daily_price` 를 세어 `trading_calendar` 를 다시 깐다. 넣은 행 수를 돌려준다.

    **거래일을 계산으로 맞히지 않는다.** 주말만 걸러 세면 개발구간 평일 3,042일 중
    162일(5.3%)이 어긋나고, 그 162일은 명절·공휴일이라 하필 실적 발표와 뉴스가 몰린다.
    우리가 실제로 받은 날이 거래일이다 — 추정이 아니라 기록이다.

    `DELETE` 후 다시 넣는 이유: 시장 구분이 바뀌거나 어떤 날을 다시 받아 종목 수가
    달라졌을 때 **낡은 줄이 남지 않게** 하기 위해서다. 4,102행이라 비용이 없다.

    ⚠️ 이 함수를 **시세를 적재한 뒤에 부른다.** 안 부르면 달력이 조용히 낡고,
       그 낡음은 "다음 거래일" 을 물었을 때 틀린 날짜로만 드러난다.
    """
    built_at = datetime.now().isoformat(timespec="seconds")
    with write_lock:
        conn.execute("DELETE FROM trading_calendar")
        # 시장별 한 줄 + 'ALL' 한 줄. 'ALL' 을 따로 세는 이유는 한쪽 시장만 열린 날을
        # 양쪽 거래일로 읽지 않기 위해서다.
        conn.execute(
            "INSERT INTO trading_calendar (bas_dd, market, stock_count, built_at) "
            "SELECT bas_dd, market, COUNT(*), ? FROM daily_price "
            "WHERE market IS NOT NULL GROUP BY bas_dd, market",
            (built_at,),
        )
        conn.execute(
            "INSERT INTO trading_calendar (bas_dd, market, stock_count, built_at) "
            "SELECT bas_dd, 'ALL', COUNT(*), ? FROM daily_price GROUP BY bas_dd",
            (built_at,),
        )
        return conn.execute("SELECT COUNT(*) FROM trading_calendar").fetchone()[0]
