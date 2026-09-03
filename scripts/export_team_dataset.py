"""팀원에게 건네줄 데이터셋을 만든다 — 개발구간만, 재현 가능하게.

왜 이 스크립트가 정본인가
------------------------
건네준 파일 자체는 저장소에 올리지 않는다(`.gitignore` 의 `data/outbox/`). KRX 이용약관
제11조 ②가 제3자 제공을 금지하고 이 저장소는 PUBLIC 이기 때문이다. 그래서 "누가 무엇을
받았는가" 를 되짚을 수 있는 것은 파일이 아니라 **이 스크립트와 MANIFEST.json 의 SHA-256**
뿐이다. 같은 커밋에서 다시 돌리면 같은 해시가 나와야 한다.

🔴 홀드아웃은 절대 나가지 않는다
------------------------------
모든 산출물은 `bas_dd < HOLDOUT_START` 로 잘린다. 봉인 구간이 한 번이라도
팀원 손에 들어가면 "미리 정해 두고 딱 한 번 열어본다" 는 우리 프레이밍이 그 자리에서
무너지고, 되돌릴 방법이 없다. 그래서 자르는 것을 각 함수에 맡기지 않고 **마지막에 전
파일을 다시 검사**한다(`verify_no_holdout`).

두 벌로 나눠 내는 이유
--------------------
    small/   지수 51종 + 종목 30개 + 피처·라벨 완성본. 수 MB. 받아서 바로 열어 본다
    full/    개발구간 전량 parquet 599만 행. 142MB. 최종 학습용

실측(2026-08-31)으로 CSV 656.7MB → parquet+zstd 142.1MB 였다. 4.6배 작고 쓰는 것도
7배 빠르다. 그래서 큰 벌은 CSV 를 만들지 않는다.

라벨을 여기서 만드는 것에 대하여
-----------------------------
라벨 조립은 원래 피처 계층(신장환 팀원)이 할 일이고 아직 없다. 그래서 **이 스크립트
안에서만** 만든다 — `features/` 에는 한 줄도 쓰지 않는다. 규칙은 새로 정하지 않고
`evaluation.horizon` 의 기존 공개 함수·상수를 그대로 따르며, 값이 같은지 아래
`verify_labels` 가 매번 대조한다. 피처 계층이 제대로 만들어지면 그쪽으로 갈아끼운다.

🔴 종목의 피처·라벨은 수정주가로 잰다
-----------------------------------
2026-09-03 까지 이 스크립트는 종목도 원문 `close`·`open` 으로 피처와 라벨을 계산했다.
원문에는 액면분할이 반영돼 있지 않아 **분할일 하루가 폭락으로 읽힌다.** 삼성전자
2018-05-04(50:1) 의 5거래일 수익률이 -97.90%(하락) 로 나왔는데 실제로는 +5.12%(상승) 다.

표본 30종목 실측: 분할·병합이 있던 종목 18/30, 라벨이 뒤집힌 행 370(0.448%),
`rsi_14` 는 삼성전자 분할 한 건에 198행(5.49%)·`atr_14` 는 186행이 어긋났다.
지수이동평균 계열이 단순이동평균보다 오래 끈다 — `sma_5` 는 4행인데 `rsi_14` 는 198행이다.

가격 칸을 고르는 것은 `price_basis` 하나뿐이고, 종목이면 `adj_*`·지수면 원문이다.
지수는 분할이라는 사건 자체가 없고 `index_price` 에 수정 칸도 없다.
**칸 이름은 바꾸지 않았다** — `sma_5` 는 그대로 `sma_5` 이고 안의 값만 기준이 바뀐다.
받아 쓰던 팀원 코드가 그대로 돌아야 하기 때문이다.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common.export_profile import write_profile  # noqa: E402
from evaluation.horizon import (  # noqa: E402
    HOLDOUT_START,
    NEUTRAL_BAND,
    classify_3,
    returns_5d_open_gapless,
    trading_day_index,
)
from features.indicators import bollinger_bands, ema, macd, rsi, sma  # noqa: E402
from features.volatility import (  # noqa: E402
    atr,
    historical_volatility,
    parkinson_volatility,
    true_range,
)
from features.volume import (  # noqa: E402
    obv,
    volume_ratio,
    volume_roc,
    volume_sma,
    vwap,
)
from ingest.store import krx_store  # noqa: E402
from supply.market import TARGET_INDEX, index_series, price_series  # noqa: E402
from supply.training import market_context, training_frame  # noqa: E402

# ── 예측 대상 ─────────────────────────────────────────────────────────────
#: 진입 t+1 시가 → 청산 t+6 시가. 5거래일.
HORIZON = 5

#: 지수 3분류 중립 밴드 ±1.0%. `evaluation.horizon` 이 못 박은 값을 그대로 쓴다.
BAND_INDEX = NEUTRAL_BAND

#: 종목 3분류 중립 밴드 ±2.0%. 지수보다 넓은 이유는 개별 종목의 변동이 크기 때문이고,
#: 이 값에서 클래스가 30.12/37.90/31.98 로 갈린다(전 종목 실측).
BAND_STOCK = 0.02

#: 개발구간 상한. `HOLDOUT_START` 하루 전까지 담는다 (`bas_dd <= DEV_END`).
#:
#: 🔴 **값을 여기에 적지 않는다.** 전에는 "20210831" 이 박혀 있어서, 봉인 시작을 옮기면
#: 두 값이 조용히 어긋났다 — 경계 하루가 개발구간과 봉인구간 양쪽에 들어가거나
#: 어느 쪽에도 안 들어가는데, 행 수만 세는 검사로는 잡히지 않는다.
#: 봉인 시작은 `evaluation/horizon.HOLDOUT_START` 하나뿐이고 여기서는 빼서 쓴다.
DEV_END = (
    datetime.strptime(HOLDOUT_START, "%Y%m%d") - timedelta(days=1)
).strftime("%Y%m%d")

# ── 표본 ──────────────────────────────────────────────────────────────────
#: 시가총액 3층에서 고르게 뽑는 수. 8 × 3 = 24.
PER_TIER = 8

#: 일부러 섞어 넣는 예외 사례 수. 팀원이 정상 종목만 보고 코드를 짜지 않게 한다.
N_DELISTED = 3
N_HALTED = 3

#: 상장폐지로 볼 기준. 개발구간 끝보다 이만큼 앞에서 자료가 끊기면 중도 소멸로 본다.
#: 달력일이지 거래일이 아니다 — 표본을 고르는 눈금이라 이 정도 어림으로 충분하고,
#: 거래일로 정확히 세려면 달력을 끌어와야 해서 표본 선정이 무거워진다.
DELISTED_GAP_DAYS = 120


def _utcnow() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def sha256_of(path: Path) -> str:
    """파일 해시. 팀원이 받은 것과 내가 보낸 것이 같은지 맞춰 보는 유일한 수단이다."""
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def size_mb(path: Path) -> float:
    return round(path.stat().st_size / 1024 / 1024, 3)


# ══════════════════════════════════════════════════════════════════════════
# 라벨 — 행 대응을 잃지 않는다
# ══════════════════════════════════════════════════════════════════════════


def forward_returns_aligned(rows: List[Dict], day_index: Dict[str, int],
                            horizon: int = HORIZON) -> List[float]:
    """행마다 미래 수익률. **길이가 `rows` 와 같고 순서가 살아 있다.**

    🔴 `evaluation.horizon.returns_5d_open_gapless` 를 그대로 쓸 수 없는 이유.
       그 함수는 조건에 맞지 않는 행을 `continue` 로 건너뛰고 값만 이어 붙인다.
       분포를 세거나 평균을 낼 때는 그걸로 충분하지만, **표에 라벨 칸을 붙이려면
       몇 번째 행의 값인지가 필요하다.** 건너뛴 자리를 모르면 라벨이 한 칸씩
       밀려 붙고, 그래도 에러는 나지 않는다.

    규칙은 그 함수와 똑같다 — 건너뛰는 대신 `nan` 을 남길 뿐이다.

      · 진입은 `t+1` 시가, 청산은 `t+1+horizon` 시가
      · 거래일 달력으로 잰 거리가 정확히 `horizon` 일 때만 센다
        (종목엔 거래정지 구멍이 있어 행 번호로 세면 3개월 차이가 5일로 둔갑한다)
      · 진입일 시가가 0 이거나 없으면 못 센다 (거래정지 표시행)
    """
    out: List[float] = [float("nan")] * len(rows)
    for i in range(len(rows) - horizon - 1):
        entry, exit_ = rows[i + 1], rows[i + 1 + horizon]
        ei = day_index.get(entry["bas_dd"])
        xi = day_index.get(exit_["bas_dd"])
        if ei is None or xi is None or xi - ei != horizon:
            continue
        if not entry.get("open"):
            continue
        out[i] = exit_["open"] / entry["open"] - 1.0
    return out


def label_3class(returns: Sequence[float], band: float) -> List[Optional[str]]:
    """±`band` 3분류. 못 잰 자리는 `None` 으로 남긴다.

    `evaluation.horizon.classify_3` 과 같은 경계다 — 초과·미만이고 경계값은 중립이다.
    """
    out: List[Optional[str]] = []
    for r in returns:
        if r is None or (isinstance(r, float) and np.isnan(r)):
            out.append(None)
        elif r > band:
            out.append("상승")
        elif r < -band:
            out.append("하락")
        else:
            out.append("중립")
    return out


def verify_labels(rows: List[Dict], day_index: Dict[str, int], band: float) -> Dict:
    """내가 만든 라벨이 기존 공개 함수와 **같은 값인지** 대조한다.

    새로 짠 계산이 맞다고 믿지 않는다. 같은 입력에 대해

      · 남은 수익률의 나열이 `returns_5d_open_gapless` 와 완전히 같은가
      · 3분류 분포가 `classify_3` 와 완전히 같은가

    를 확인하고, 어긋나면 그 자리에서 멈춘다. 조용히 다른 답을 담아 내보내면
    팀원 넷이 그걸 기준으로 몇 주를 쓴다.
    """
    mine = forward_returns_aligned(rows, day_index)
    kept = [v for v in mine if not np.isnan(v)]
    theirs = returns_5d_open_gapless(rows, day_index, HORIZON)

    if len(kept) != len(theirs):
        raise AssertionError(
            f"라벨 대조 실패 — 개수가 다르다. 내 것 {len(kept)} · 기존 {len(theirs)}"
        )
    if kept and not np.allclose(kept, theirs, rtol=0, atol=0):
        어긋난_수 = int(np.sum(~np.isclose(kept, theirs, rtol=0, atol=0)))
        raise AssertionError(f"라벨 대조 실패 — 값이 {어긋난_수}개 다르다")

    labels = label_3class(mine, band)
    mine_dist = {"상승": 0, "중립": 0, "하락": 0}
    for label in labels:
        if label:
            mine_dist[label] += 1
    theirs_dist = classify_3(theirs, band)
    if mine_dist != theirs_dist:
        raise AssertionError(
            f"3분류 대조 실패 — 내 것 {mine_dist} · 기존 {theirs_dist}"
        )

    return {
        "rows": len(rows),
        "labeled": len(kept),
        "unlabeled": len(rows) - len(kept),
        "distribution": mine_dist,
    }


# ══════════════════════════════════════════════════════════════════════════
# 피처 — 공개 지표 14개를 22칸으로 편다
# ══════════════════════════════════════════════════════════════════════════

#: 워밍업이 가장 긴 지표(sma_60)가 필요로 하는 앞구간 길이. 표의 맨 앞 이만큼은
#: 어차피 nan 이라 `dropna` 로 떨어진다. 몇 행이 왜 사라졌는지 세어 보고한다.
LONGEST_WARMUP = 60

#: 피처와 라벨을 **어느 가격으로 재는가**. 칸 이름을 여기서 한 번만 고른다.
#:
#: 🔴 종목은 분할이 조정된 값이 정답이다. 원문 `close` 는 KRX 가 준 그대로라 액면분할이
#:    반영돼 있지 않고, 분할일 하루가 폭락으로 읽힌다. 삼성전자 2018-05-04(50:1) 을
#:    원문으로 재면 5거래일 수익률이 **-97.90%(하락)** 인데 실제로는 **+5.12%(상승)** 다.
#:    라벨이 정반대로 붙는다.
#:
#:    창이 긴 지표는 분할일 하루가 지나간 뒤로도 오래 오염된다. 표본 30종목 실측에서
#:    라벨은 370행(0.448%)이 뒤집혔고, 삼성전자 분할 한 건에 `rsi_14` 는 198행(5.49%)·
#:    `atr_14` 는 186행이 어긋났다. 이동평균보다 지수이동평균 계열이 훨씬 오래 끈다.
#:
#: 지수에는 분할이라는 사건 자체가 없고 `index_price` 에 수정 칸도 없다. 원문이 정답이다.
PRICE_BASIS_ADJ = {"open": "adj_open", "high": "adj_high",
                   "low": "adj_low", "close": "adj_close"}
PRICE_BASIS_RAW = {"open": "open", "high": "high", "low": "low", "close": "close"}


def price_basis(frame: pd.DataFrame) -> Dict[str, str]:
    """이 표를 어느 가격으로 계산할지 고른다. 수정 칸이 있으면 그쪽이다."""
    return PRICE_BASIS_ADJ if "adj_close" in frame.columns else PRICE_BASIS_RAW


def rows_on_basis(frame: pd.DataFrame, basis: Dict[str, str]) -> List[Dict]:
    """`open` 이라는 **이름 자리에** 계산 기준 시가를 앉힌 행 목록을 만든다.

    왜 값을 옮겨 담나 — `evaluation.horizon` 의 공개 함수들이 `rows[i]["open"]` 이라는
    이름으로 읽기 때문이다. 그 계층은 평가 담당(강민석 팀원) 것이라 이쪽에서 고치지
    않는다. 이름만 맞춰 주면 **같은 함수가 그대로 수정주가로 센다.**

    🔴 거래정지일의 수정 시가는 `NaN` 인데, 그쪽 걸러내기는 `if not entry.get("open")`
       이라 **`NaN` 을 못 거른다** (`not float("nan")` 은 `False`). 원문에서 정지일
       시가가 `0` 이라 걸러지던 것과 같은 규약이 되도록 `0` 으로 바꿔 담는다.
       지금은 `training_frame` 이 정지일을 이미 덜어내 결측이 0행이지만(실측),
       그쪽이 바뀌어도 라벨이 조용히 틀리지 않게 여기서 막는다.
    """
    if basis is PRICE_BASIS_RAW:
        return frame.to_dict("records")
    사본 = frame.copy()
    for 표준, 실제 in basis.items():
        사본[표준] = 사본[실제].fillna(0)
    return 사본.to_dict("records")


def build_feature_frame(frame: pd.DataFrame, *, band: float,
                        day_index: Dict[str, int]) -> Tuple[pd.DataFrame, Dict]:
    """시세 표에 피처 22칸과 라벨 2칸을 붙인다.

    피처는 `features/` 의 **공개 함수 14개를 전부** 쓴다. 우리가 가진 지표가 무엇인지
    팀원이 표만 보고도 알 수 있어야 하기 때문이다. 창(window)은 흔히 쓰는 값으로 두되
    고정한다 — 창을 고르는 것 자체가 시도 횟수라, 여기서 여러 개를 흔들면 안 된다.

    가격은 `price_basis` 가 고른다 — 종목이면 분할이 조정된 값, 지수면 원문이다.
    칸 이름(`sma_5` 등)은 그대로 두고 **안의 값만** 그 기준으로 계산한다.
    """
    if frame.empty:
        return frame.copy(), {"rows": 0, "labeled": 0, "unlabeled": 0,
                              "distribution": {}, "dropped_warmup": 0,
                              "price_basis": PRICE_BASIS_RAW["close"]}

    기준 = price_basis(frame)
    close = frame[기준["close"]].astype(float).to_numpy()
    high = frame[기준["high"]].astype(float).to_numpy()
    low = frame[기준["low"]].astype(float).to_numpy()
    vol = frame["volume"].astype(float).to_numpy()

    out = frame.copy()

    # 추세 — 이동평균
    out["sma_5"] = sma(close, 5)
    out["sma_20"] = sma(close, 20)
    out["sma_60"] = sma(close, LONGEST_WARMUP)
    out["ema_12"] = ema(close, 12)
    out["ema_26"] = ema(close, 26)

    # 모멘텀
    out["rsi_14"] = rsi(close, 14)
    m = macd(close)
    out["macd"] = m["macd"]
    out["macd_signal"] = m["signal"]
    out["macd_hist"] = m["hist"]

    # 밴드
    bb = bollinger_bands(close, 20)
    out["bb_mid"] = bb["mid"]
    out["bb_upper"] = bb["upper"]
    out["bb_lower"] = bb["lower"]
    out["bb_bandwidth"] = bb["bandwidth"]

    # 변동성
    out["true_range"] = true_range(high, low, close)
    out["atr_14"] = atr(high, low, close, 14)
    out["hv_20"] = historical_volatility(close, 20)
    out["parkinson_20"] = parkinson_volatility(high, low, 20)

    # 거래량
    out["vol_sma_20"] = volume_sma(vol, 20)
    out["vol_ratio_20"] = volume_ratio(vol, 20)
    out["obv"] = obv(close, vol)
    out["vwap_20"] = vwap(close, vol, 20)
    out["vol_roc_5"] = volume_roc(vol, 5)

    # 라벨 — 먼저 기존 함수와 대조해 보고, 통과한 뒤에만 붙인다
    rows = rows_on_basis(frame, 기준)
    stats = verify_labels(rows, day_index, band)
    out["fwd_return_5d"] = forward_returns_aligned(rows, day_index)
    out["label"] = label_3class(out["fwd_return_5d"].tolist(), band)

    stats["dropped_warmup"] = LONGEST_WARMUP - 1
    stats["price_basis"] = 기준["close"]
    return out, stats


FEATURE_COLUMNS = [
    "sma_5", "sma_20", "sma_60", "ema_12", "ema_26",
    "rsi_14", "macd", "macd_signal", "macd_hist",
    "bb_mid", "bb_upper", "bb_lower", "bb_bandwidth",
    "true_range", "atr_14", "hv_20", "parkinson_20",
    "vol_sma_20", "vol_ratio_20", "obv", "vwap_20", "vol_roc_5",
]


def ready_to_fit(frame: pd.DataFrame) -> pd.DataFrame:
    """워밍업 `nan` 과 라벨 없는 꼬리를 떼어 **바로 `fit` 되는 표**로 만든다.

    팀원이 받자마자 `model.fit(X, y)` 를 돌릴 수 있어야 한다. 앞쪽 `nan` 을 남겨 두면
    각자 다르게 지우고, 그 차이가 나중에 "왜 내 점수만 다르지" 로 돌아온다.
    """
    if frame.empty:
        return frame
    need = FEATURE_COLUMNS + ["fwd_return_5d", "label"]
    return frame.dropna(subset=[c for c in need if c in frame.columns]).reset_index(drop=True)


# ══════════════════════════════════════════════════════════════════════════
# 표본 — 정상만 담지 않는다
# ══════════════════════════════════════════════════════════════════════════


def pick_sample_codes(conn) -> Tuple[List[str], Dict]:
    """종목 30개를 **결정적으로** 고른다. 무작위가 아니라 규칙이다.

    같은 DB 로 다시 돌리면 같은 30개가 나와야 팀원과 같은 표를 보고 이야기할 수 있다.

    구성
    ----
      시총 3층 × 8     대·중·소형이 고르게 섞이게. 대형주만 담으면 유동성이 없는
                      종목에서 무슨 일이 나는지 아무도 못 본다
      중도 소멸 3      개발구간 안에서 자료가 끊긴 종목. 상장폐지·합병이 여기 있고,
                      행 번호로 5일 뒤를 세는 코드가 여기서 조용히 틀린다
      거래정지 3       시고저가가 0 인 행을 가진 종목. 나눗셈이 여기서 터진다

    예외 사례를 일부러 섞는 이유는 팀원이 **정상 종목만 보고 짠 코드**를 나중에 전량
    데이터에 돌렸다가 그때서야 깨지는 일을 막기 위해서다.
    """
    마지막날 = conn.execute(
        "SELECT MAX(bas_dd) FROM daily_price WHERE bas_dd <= ?", (DEV_END,)
    ).fetchone()[0]

    # 시총 층화 — 마지막 거래일에 살아 있고 시총이 잡히는 종목만 순위를 매긴다
    ranked = conn.execute(
        "SELECT code, market_cap, market FROM daily_price "
        "WHERE bas_dd = ? AND market_cap IS NOT NULL AND market_cap > 0 "
        "ORDER BY market_cap DESC",
        (마지막날,),
    ).fetchall()

    tiers: Dict[str, List[str]] = {}
    n = len(ranked)
    경계 = [0, n // 3, 2 * n // 3, n]
    for 층, 이름 in enumerate(("대형", "중형", "소형")):
        구간 = ranked[경계[층]:경계[층 + 1]]
        if not 구간:
            tiers[이름] = []
            continue
        # 구간 안에서 등간격으로 집는다 — 맨 위만 집으면 층을 나눈 뜻이 없다
        step = max(1, len(구간) // PER_TIER)
        tiers[이름] = [row[0] for row in 구간[::step][:PER_TIER]]

    # 중도 소멸 — 개발구간이 끝나기 한참 전에 자료가 끊긴 종목
    소멸선 = (datetime.strptime(DEV_END, "%Y%m%d")
              - timedelta(days=DELISTED_GAP_DAYS)).strftime("%Y%m%d")
    소멸 = [
        r[0]
        for r in conn.execute(
            "SELECT code, MAX(bas_dd) last_dd FROM daily_price WHERE bas_dd <= ? "
            "GROUP BY code HAVING last_dd < ? ORDER BY last_dd DESC, code LIMIT ?",
            (DEV_END, 소멸선, N_DELISTED * 3),
        ).fetchall()
    ]

    # 거래정지 — 시고저가가 0 인 행을 가진 종목 (거래량까지 있는 쪽을 앞세운다)
    정지 = [
        r[0]
        for r in conn.execute(
            "SELECT code, COUNT(*) n FROM daily_price "
            "WHERE bas_dd <= ? AND open = 0 AND high = 0 AND low = 0 "
            "GROUP BY code ORDER BY n DESC, code LIMIT ?",
            (DEV_END, N_HALTED * 3),
        ).fetchall()
    ]

    골라낸: List[str] = []
    사유: Dict[str, str] = {}

    def 넣기(codes: Sequence[str], why: str, limit: int) -> None:
        담은 = 0
        for c in codes:
            if 담은 >= limit or c in 사유:
                continue
            골라낸.append(c)
            사유[c] = why
            담은 += 1

    for 이름 in ("대형", "중형", "소형"):
        넣기(tiers[이름], f"시총 {이름}", PER_TIER)
    넣기(소멸, "중도 소멸", N_DELISTED)
    넣기(정지, "거래정지 이력", N_HALTED)

    골라낸.sort()
    return 골라낸, {
        "기준일": 마지막날,
        "층화_모집단": n,
        "사유별": 사유,
    }


# ══════════════════════════════════════════════════════════════════════════
# 반출
# ══════════════════════════════════════════════════════════════════════════


def _write_csv(frame: pd.DataFrame, path: Path, files: List[Dict], note: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, encoding="utf-8-sig")
    files.append({
        "path": path.name,
        "rows": int(len(frame)),
        "columns": list(frame.columns),
        "size_mb": size_mb(path),
        "sha256": sha256_of(path),
        "note": note,
    })
    print(f"  ✅ {path.name:38s} {len(frame):>9,}행 · {size_mb(path):>7.2f} MB")


def _write_parquet(frame: pd.DataFrame, path: Path, files: List[Dict], note: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, index=False, compression="zstd")
    files.append({
        "path": path.name,
        "rows": int(len(frame)),
        "columns": list(frame.columns),
        "size_mb": size_mb(path),
        "sha256": sha256_of(path),
        "note": note,
    })
    print(f"  ✅ {path.name:38s} {len(frame):>9,}행 · {size_mb(path):>7.2f} MB")


def _price_dev(code: str) -> pd.DataFrame:
    """개발구간 원시 시세. `supply.price_series` 는 `as_of` 를 키워드로 요구한다."""
    return price_series(code, as_of=datetime.now().strftime("%Y%m%d"), end=DEV_END)


def verify_no_holdout(root: Path) -> int:
    """내보낸 **모든** 파일을 다시 열어 홀드아웃이 섞이지 않았는지 확인한다.

    각 함수가 알아서 잘랐을 것이라고 믿지 않는다. 자르는 곳이 여러 군데면 언젠가
    한 곳이 빠지고, 빠진 그 한 번이 봉인을 연다. 되돌릴 수 없는 종류의 사고라
    비용을 더 내고 전수로 검사한다.
    """
    검사한 = 0
    for path in sorted(root.rglob("*")):
        if path.suffix == ".csv":
            df = pd.read_csv(path, usecols=["bas_dd"], dtype={"bas_dd": str})
        elif path.suffix == ".parquet":
            df = pd.read_parquet(path, columns=["bas_dd"])
        else:
            continue
        if df.empty:
            continue
        가장늦은 = str(df["bas_dd"].astype(str).max())
        if 가장늦은 >= HOLDOUT_START:
            raise AssertionError(
                f"🔴 홀드아웃 누수 — {path.name} 의 마지막 날짜가 {가장늦은} 다 "
                f"(봉인 시작 {HOLDOUT_START})"
            )
        검사한 += 1
    return 검사한


def main() -> int:
    parser = argparse.ArgumentParser(description="팀원 반출용 데이터셋을 만든다")
    parser.add_argument("--out", default=None, help="출력 폴더 (기본 data/outbox/<오늘>)")
    parser.add_argument("--skip-full", action="store_true",
                        help="전량 parquet 을 건너뛴다 (작은 벌만 빨리 확인할 때)")
    args = parser.parse_args()

    오늘 = datetime.now().strftime("%Y-%m-%d")
    root = Path(args.out) if args.out else Path("data/outbox") / 오늘
    small, full = root / "small", root / "full"
    root.mkdir(parents=True, exist_ok=True)

    print(f"반출 폴더: {root}")
    print(f"개발구간 상한: {DEV_END} (봉인 시작 {HOLDOUT_START})")
    print()

    files: List[Dict] = []
    stats: Dict = {}
    오늘_as_of = datetime.now().strftime("%Y%m%d")

    # ── 작은 벌 ──────────────────────────────────────────────────────
    print("[작은 벌] 지수")
    with krx_store.connect() as conn:
        지수이름들 = [
            r[0] for r in conn.execute(
                "SELECT DISTINCT index_name FROM index_price ORDER BY index_name"
            ).fetchall()
        ]
        index_all = pd.read_sql_query(
            "SELECT * FROM index_price WHERE bas_dd <= ? ORDER BY index_name, bas_dd",
            conn, params=(DEV_END,),
        )
    _write_csv(index_all, small / "index_all_dev.csv", files,
               f"지수 {len(지수이름들)}종 개발구간 전체")

    kospi200 = index_series(TARGET_INDEX, as_of=오늘_as_of, end=DEV_END)
    _write_csv(kospi200, small / "index_kospi200_dev.csv", files,
               "코스피 200 — 예측 대상 그 자체")

    # 거래일 달력은 시장 전체 기준이다. 종목별 행으로 세면 정지 구간에서 틀린다.
    with krx_store.connect() as conn:
        모든거래일 = [
            r[0] for r in conn.execute(
                "SELECT DISTINCT bas_dd FROM daily_price WHERE bas_dd <= ? ORDER BY bas_dd",
                (DEV_END,),
            ).fetchall()
        ]
    day_index = trading_day_index(모든거래일)

    print()
    print("[작은 벌] 지수 피처·라벨")
    idx_feat, idx_stats = build_feature_frame(kospi200, band=BAND_INDEX,
                                              day_index=day_index)
    idx_fit = ready_to_fit(idx_feat)
    _write_csv(idx_fit, small / "features_labels_kospi200_dev.csv", files,
               f"피처 {len(FEATURE_COLUMNS)}칸 + 라벨. 바로 fit 된다 (밴드 ±{BAND_INDEX:.1%}) "
               f"· 지수는 분할이 없어 원문 가격 기준")
    stats["kospi200"] = {**idx_stats, "fit_rows": int(len(idx_fit))}
    print(f"     분포 {idx_stats['distribution']} · dropna 후 {len(idx_fit):,}행"
          f" · 가격 기준 {idx_stats['price_basis']}")

    print()
    print("[작은 벌] 종목 표본")
    with krx_store.connect() as conn:
        codes, 표본메타 = pick_sample_codes(conn)
    print(f"     {len(codes)}종목: {', '.join(codes[:10])}...")

    ctx = market_context()
    raw_parts, train_parts, feat_parts = [], [], []
    종목분포 = {"상승": 0, "중립": 0, "하락": 0}
    종목기준 = set()
    for code in codes:
        원시 = _price_dev(code)
        if not 원시.empty:
            raw_parts.append(원시)
        정제 = training_frame(code, holdout_start=HOLDOUT_START, context=ctx)
        if 정제.empty:
            continue
        train_parts.append(정제)
        피처, s = build_feature_frame(정제, band=BAND_STOCK, day_index=day_index)
        준비 = ready_to_fit(피처)
        if not 준비.empty:
            feat_parts.append(준비)
        for k, v in s["distribution"].items():
            종목분포[k] += v
        종목기준.add(s["price_basis"])

    # 🔴 종목마다 기준이 갈리면 한 파일 안에서 어떤 행은 원문·어떤 행은 수정주가가 된다.
    #    행 수만 세는 검사로는 절대 안 잡히므로 여기서 못 박는다.
    if len(종목기준) > 1:
        raise AssertionError(f"종목마다 가격 기준이 다르다 — {sorted(종목기준)}")

    stocks_raw = (pd.concat(raw_parts, ignore_index=True) if raw_parts
                  else pd.DataFrame())
    stocks_train = (pd.concat(train_parts, ignore_index=True) if train_parts
                    else pd.DataFrame())
    stocks_feat = (pd.concat(feat_parts, ignore_index=True) if feat_parts
                   else pd.DataFrame())

    _write_csv(stocks_raw, small / "stocks_sample30_raw_dev.csv", files,
               "표본 종목 원시 시세 — 정리매매·거래정지가 그대로 들어 있다")
    _write_csv(stocks_train, small / "stocks_sample30_train_dev.csv", files,
               "표본 종목 학습용 — 정리매매·신규상장·거래정지를 덜어냈다")
    _write_csv(stocks_feat, small / "features_labels_stocks30_dev.csv", files,
               f"표본 종목 피처+라벨. 바로 fit 된다 (밴드 ±{BAND_STOCK:.1%}) "
               f"· 🔴 피처·라벨 모두 수정주가(adj_*) 기준")
    기준 = 종목기준.pop() if 종목기준 else PRICE_BASIS_ADJ["close"]
    stats["stocks30"] = {"codes": codes, "선정": 표본메타,
                         "distribution": 종목분포, "price_basis": 기준}
    print(f"     분포 {종목분포} · 가격 기준 {기준}")

    (small / "sample_codes.json").write_text(
        json.dumps(표본메타, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # ── 큰 벌 ────────────────────────────────────────────────────────
    if not args.skip_full:
        print()
        print("[큰 벌] 전량 parquet")
        with krx_store.connect() as conn:
            conn.execute("PRAGMA cache_size = -1000000")
            daily = pd.read_sql_query(
                "SELECT * FROM daily_price WHERE bas_dd <= ?", conn, params=(DEV_END,)
            )
        _write_parquet(daily, full / "daily_price_dev.parquet", files,
                       "개발구간 전 종목 시세")
        del daily
        with krx_store.connect() as conn:
            idx = pd.read_sql_query(
                "SELECT * FROM index_price WHERE bas_dd <= ?", conn, params=(DEV_END,)
            )
        _write_parquet(idx, full / "index_price_dev.parquet", files,
                       "개발구간 전 지수")
        del idx

    # ── 검사 ─────────────────────────────────────────────────────────
    print()
    print("[검사] 홀드아웃 누수")
    검사수 = verify_no_holdout(root)
    print(f"  ✅ {검사수}개 파일 전수 확인 — 봉인 구간 없음")

    manifest = {
        "generated_at": _utcnow(),
        "holdout_start": HOLDOUT_START,
        "dev_end": DEV_END,
        "horizon": HORIZON,
        "band": {"index": BAND_INDEX, "stock": BAND_STOCK},
        "files": files,
        "stats": stats,
    }
    (root / "MANIFEST.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print()
    print(f"MANIFEST.json 기록 — 파일 {len(files)}개")

    # ── 칸마다 무엇이 들었나 ─────────────────────────────────────────
    #
    # MANIFEST 는 칸 **이름**과 행 수까지다. 받는 사람이 매번 직접 `describe()` 를
    # 돌리지 않도록, 결측률·값 범위·분포를 여기서 재서 남긴다. 데이터셋 카드도 이
    # 파일을 읽어 만들기 때문에, 카드에 손으로 적은 숫자가 자료와 어긋날 자리가 없다.
    print()
    print("[칸별 통계] PROFILE.json")
    프로필 = write_profile(root)
    print(f"  ✅ 파일 {len(프로필['files'])}개 · "
          f"칸 {sum(f['칸수'] for f in 프로필['files'])}개")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
