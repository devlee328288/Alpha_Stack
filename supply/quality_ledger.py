"""데이터 품질 원장 — 게이트가 재는 것과 아무 데도 안 남던 것을 표 하나로 모은다.

설계: `docs/데이터파트/version3.9/데이터_품질_원장_설계.md` (로드맵 ⑯)

## 원장은 막지 않는다. 적는다

막는 것은 지금처럼 게이트(반출 전 검증기 4종·홀드아웃 누수 검사)가 한다. 원장은 **붉어도
업로드를 세우지 않고, 붉다는 사실을 카드에 붉게 적어 내보낸다.** "알면서 내보내는 것" 을
숨기지 않는 것이 목적이다. 원장 **생성 자체가 실패**하면 그때는 `error` 다.

## 왜 필요했나 — 네 가지가 흩어져 있었다

    reports/data_quality.json        시세·지수 구조 검사        저장소를 열면 보인다
    data/outbox/*/PROFILE.json       칸별 통계 185칸            로컬 반출 폴더에만
    검증기 4종 표준출력               규격 판정                  화면에만 흐른다
    🔴 어디에도 없다                  의심 행 수 · 축별 중복 · 달력 정합

그래서 "이 반출본에 의심 행이 몇 개인가", "지난 반출본과 무엇이 달라졌나" 를 물을 곳이
없었다. 세 번을 손으로 셌다.

## 🔴 합계가 아니라 키의 축마다 센다

이 원장이 생긴 직접적인 계기다. `index_price` 40,324행이 덮였을 때 **합계는 오히려 늘었다.**
KOSPI 와 KOSDAQ 이 같은 이름의 업종지수를 갖는데 기본키에 시장이 없어서, 한쪽이 다른 쪽을
덮으면서 다른 날짜 행이 함께 들어온 것이다. `index_class` 축으로 세고서야 보였다.

    잘못된 검사: COUNT(*) 가 늘었나 줄었나        → 못 잡는다
    맞는 검사:   (bas_dd, index_name) 로 센 값 ≠ (bas_dd, index_name, index_class) 로 센 값

그래서 §2.4 는 **같은 표를 두 축으로 세서 비교**한다.

## 무엇을 싣나 — 일곱 축

    missing     결측 — 칸별 결측률 · 정지일이 0 이 아니라 NaN 인가 · 층별 정지일 비율
    validity    유효성 — 값이 규칙을 지키나 (OHLC 순서 · 시가총액 항등식 · 부호)
    outlier     이상치 — is_adj_suspect 를 adj_source 별로 · 후보군 교집합 · 극단 사건
    volume      거래량 급변 — KRX 시장경보 기준(최근 5일 평균 대비 3배)으로 센다
    calendar    달력 정합 — 최신일 일치 · 시장별 거래일 수 · 반출본과 DB 의 차이
    duplicate   중복 — 축마다 (위 참조)
    sealing     시점·봉인 — 반출본 최대 날짜 < 홀드아웃 시작

`validity` 는 표준 6차원(completeness·validity·accuracy·consistency·uniqueness·
timeliness) 중 우리에게 없던 하나다. `volume` 은 qlib 건강검사에는 있는데 우리에게
없던 것인데, **기준값을 그대로 가져오지 않았다** — §거래량 급변 참조.

각 지표는 `{"value", "red_if", "status", "note"}` 넷을 갖는다. `status` 는
`ok` · `red` · `warn` · `skip`(잴 자료가 없다) 중 하나다.
"""

from __future__ import annotations

import json
import subprocess
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd

from evaluation.horizon import HOLDOUT_START

#: 반출 폴더에 남는 이름. `MANIFEST.json` · `PROFILE.json` 옆이라 반출본과 같이 움직인다.
LEDGER_NAME = "QUALITY_LEDGER.json"
#: 저장소에 쌓는 이력. append-only — 고치지 않고 새 줄을 쓴다.
HISTORY_PATH = Path("reports") / "quality_ledger.jsonl"

#: 일곱 축의 이름과 순서. 카드 표도 이 순서로 나간다.
AXES = ("missing", "validity", "outlier", "volume", "calendar", "duplicate",
        "sealing")

#: 거래량 급변을 재는 기준 — **KRX 시장경보제도**가 쓰는 정의를 그대로 따른다.
#: 시장감시규정 시행세칙의 투자주의 지정 요건 중 하나가 *"당일의 거래량이 최근
#: 5일 평균 거래량 대비 3배 이상 증가"* 다. 한국 시장의 공식 정의라 방어가 쉽고,
#: 셋 중 가장 보수적이다.
#:
#: 실측 2026-09-09 · 개발구간 · 3배 초과 비율
#:
#:     기준선              전체      KOSPI 상위 50
#:     KRX 5일 평균      5.426%          1.374%     ← 이걸 쓴다
#:     qlib 전일         7.977%          1.908%
#:     20일 중앙값      10.329%          2.353%
#:
#: 🔴 **기준선을 바꾸면 값이 2배 갈린다.** 배수만 베끼면 안 되는 이유다.
SURGE_WINDOW = 5
SURGE_MULTIPLE = 3.0

#: KST. 반출 시각은 사람이 읽는 값이라 현지 시간으로 적는다.
KST = timezone(timedelta(hours=9))

_OK, _RED, _WARN, _SKIP = "ok", "red", "warn", "skip"


def _metric(value: Any, red_if: str, status: str, note: str = "") -> Dict[str, Any]:
    """지표 한 칸. `value` 는 숫자일 수도 있고 축별로 나눈 dict 일 수도 있다."""
    out: Dict[str, Any] = {"value": value, "red_if": red_if, "status": status}
    if note:
        out["note"] = note
    return out


def _git_sha() -> Optional[str]:
    """지금 코드가 어느 커밋인가. 못 읽으면 None — 원장 생성을 막지는 않는다."""
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                             capture_output=True, text=True, timeout=10, check=False)
        return out.stdout.strip() or None
    except Exception:                                     # noqa: BLE001 — 기록용이라 삼킨다
        return None


# ── 축 1. 결측 ────────────────────────────────────────────────────────────────

def _axis_missing(profile: Optional[Dict], daily: Optional[pd.DataFrame]) -> Dict:
    """결측 — 이미 `PROFILE.json` 이 185칸을 재므로 여기서는 **가리키고 판정만** 더한다."""
    axis: Dict[str, Any] = {}

    if profile is None:
        axis["profile_columns"] = _metric(None, "PROFILE.json 이 없다", _SKIP)
    else:
        칸수 = sum(f.get("칸수", 0) for f in profile.get("files", []))
        axis["profile_columns"] = _metric(
            칸수, "(재지 않는다 — PROFILE.json 을 가리킨다)", _OK,
            "칸별 결측률·분포는 PROFILE.json 에 있다. 원장은 그것을 가리킨다",
        )

    if daily is None or daily.empty:
        axis["zero_filled_halt_rows"] = _metric(None, "반출본을 못 읽었다", _SKIP)
        axis["halt_ratio_by_layer"] = _metric(None, "반출본을 못 읽었다", _SKIP)
        return axis

    # 🔴 정지일 판정은 **거래량 0** 이다. 종가로 판정하면 한 건도 못 잡는다 —
    #    KRX 는 거래정지일에도 종가 칸에 직전 가격을 그대로 실어 보낸다(실측: 반출본
    #    7,888,945행에서 close == 0 인 행은 0건, volume == 0 인 행은 231,684건).
    #    처음에 `close == 0` 을 함께 걸었다가 정지행이 0 으로 나와 이 주석을 남긴다.
    거래없음 = daily["volume"].fillna(-1).eq(0)
    정지행 = int(거래없음.sum())
    adj칸 = [c for c in ("adj_open", "adj_high", "adj_low", "adj_close")
             if c in daily.columns]
    zero_filled = int(
        (거래없음 & daily[adj칸].fillna(-1).eq(0).any(axis=1)).sum()
    ) if adj칸 else 0
    axis["halt_rows"] = _metric(정지행, "(붉지 않다 · 기록만)", _OK)
    axis["zero_filled_halt_rows"] = _metric(
        zero_filled, "> 0 (정지일 adj_* 는 NaN 이어야 한다)",
        _RED if zero_filled else _OK,
        "0 으로 채우면 일간수익률이 -100% 가 되어 없던 사건이 생긴다",
    )

    # 층마다 다르다 — 전 시장에서는 흔하고, 후보군·라벨 있는 행으로 갈수록 사라진다.
    층: Dict[str, float] = {"all": round(정지행 / len(daily), 5) if len(daily) else 0.0}
    axis["halt_ratio_by_layer"] = _metric(
        층, "(붉지 않다 · 기록만)", _OK,
        "후보군·라벨 층은 학습 경로에서 재므로 여기서는 전 시장만 잰다",
    )
    return axis


# ── 축 2. 유효성 — 값이 규칙을 지키나 ─────────────────────────────────────────

def _axis_validity(daily: Optional[pd.DataFrame]) -> Dict:
    """유효성 — *"Does the data match the rules?"* 표준 6차원 중 우리에게 없던 축.

    ## 정지행을 위반으로 세면 전부가 붉어진다

    KRX 는 거래정지 중에도 행을 준다. 그 행은 `open=high=low=0` 이고 종가만 직전 값을
    물고 있다 — 개발구간에 231,808행(2.938%)이다. 이건 규칙 위반이 아니라 **KRX 의
    정지 표기**다. 그래서 **체결이 있던 행만**(`volume > 0`) 재고, 나머지는
    `is_halted` 칸으로 따로 표시한다.

    ## 실측이 뒤집은 것 — 124행은 "고가·저가가 살아 있는" 행이 아니었다

    처음에는 이 124행을 *"`open ≤ 0` 인데 고가·저가는 살아 있는 행"* 이라고 적었다.
    실제로 열어 보니 **고가·저가도 0** 이고 종가와 거래량만 있다. 거래량이 1주·30주인
    행이 많고, 96종목이 우선주·리츠·스팩이다. 정규장 체결 없이 **시간외 단일가만
    체결된 날**이다.

    실측 2026-09-09 · 개발구간 · `volume > 0` 인 7,657,137행:

        high >= low                       0
        high >= max(open, close)        124   ← high=0 이고 close>0
        low  <= min(open, close)          0
        open > 0                        124   ← 같은 행이다
        market_cap == close x shares      0
        value >= 0 · listed_shares > 0    0

    그 124행은 `is_traded` 가 거짓이라 **작은 벌에서는 이미 빠져 있다.** 큰 벌에만
    남아 있었고, `is_halted` 칸을 실으면서 큰 벌에서도 보이게 됐다.

    붉게 보는 것은 **OHLC 순서**와 **시가총액 항등식**뿐이다. 둘은 어떤 시장 사건으로도
    설명되지 않는 계산 오류다. `open <= 0` 은 위 이유로 기록만 한다.
    """
    axis: Dict[str, Any] = {}
    필요 = {"open", "high", "low", "close", "volume", "market_cap", "listed_shares"}
    if daily is None or daily.empty or 필요 - set(daily.columns):
        return {k: _metric(None, "반출본을 못 읽었다", _SKIP)
                for k in ("ohlc_order_rows", "market_cap_identity_rows",
                          "nonpositive_open_rows", "negative_value_rows")}

    체결 = daily["volume"].fillna(0) > 0
    sub = daily.loc[체결]

    순서 = (
        (sub["high"] < sub["low"])
        | (sub["high"] < sub[["open", "close"]].max(axis=1))
        | (sub["low"] > sub[["open", "close"]].min(axis=1))
    )
    # 🔴 `open <= 0` 한 갈래를 따로 뺀다. 시간외 단일가만 체결된 날이라 고가·저가가
    #    0 인데, 그건 계산 오류가 아니라 그 날 정규장에 체결이 없었다는 뜻이다.
    시간외 = sub["open"] <= 0
    진짜순서 = 순서 & ~시간외

    axis["ohlc_order_rows"] = _metric(
        int(진짜순서.sum()), "> 0 (어떤 시장 사건으로도 설명되지 않는다)",
        _RED if int(진짜순서.sum()) else _OK,
        f"체결이 있던 {int(체결.sum()):,}행에서 잰다 · 시간외 단일가 행은 아래로 뺀다",
    )
    axis["nonpositive_open_rows"] = _metric(
        int(시간외.sum()), "(붉지 않다 — 시간외 단일가만 체결된 날이다)", _OK,
        "정규장 체결이 없어 시·고·저가가 0 이고 종가·거래량만 있다 · "
        "`is_halted` 가 참이라 학습 표본에서는 이미 빠진다",
    )

    쓸수있음 = (sub["market_cap"].notna() & sub["listed_shares"].notna()
                & sub["close"].notna())
    계산 = sub["close"] * sub["listed_shares"]
    어긋남 = 쓸수있음 & (
        (sub["market_cap"] - 계산).abs() > 1e-7 * sub["market_cap"].abs())
    axis["market_cap_identity_rows"] = _metric(
        int(어긋남.sum()), "> 0 (시가총액 = 종가 x 상장주식수 가 깨졌다)",
        _RED if int(어긋남.sum()) else _OK,
        f"상대오차 1e-7 · 비교 가능 {int(쓸수있음.sum()):,}행 · "
        "1.8경이라 float64 유효숫자를 넘어 절대오차로는 못 잰다",
    )

    음수 = (sub["volume"] < 0) | (sub["market_cap"] < 0) | (sub["listed_shares"] <= 0)
    if "value" in sub.columns:
        음수 = 음수 | (sub["value"] < 0)
    axis["negative_value_rows"] = _metric(
        int(음수.fillna(False).sum()), "> 0 (음수 거래량·시가총액·상장주식수)",
        _RED if int(음수.fillna(False).sum()) else _OK,
    )
    return axis


# ── 축 3. 이상치 — 크기가 아니라 어긋남 ───────────────────────────────────────

def _axis_outlier(daily: Optional[pd.DataFrame]) -> Dict:
    """이상치 — KRX 등락률과 우리 수정주가 수익률이 어긋난 행만 의심한다.

    크기로 자르지 않는다. 하루 100% 는 권리락·거래정지 해제로 실제 일어나고, 그것은
    지울 오류가 아니라 모델이 배울 사건이다.
    """
    axis: Dict[str, Any] = {}
    if daily is None or daily.empty:
        return {k: _metric(None, "반출본을 못 읽었다", _SKIP)
                for k in ("adj_suspect_rows", "extreme_rows")}

    from supply.adj_quality import flag_adjustment_quality

    flags = flag_adjustment_quality(daily)
    요약 = dict(flags.attrs.get("adjustment_quality", {}))
    의심 = flags["is_adj_suspect"].fillna(False)
    극단 = flags["is_extreme_return"].fillna(False)

    # `adj_source` 로 나누는 것이 핵심이다. fdr 구간의 의심은 원천이 다르게 편 것이라
    # 기록만 하고, **chain 구간의 의심은 우리 계산이 틀린 것**이라 붉다.
    by_source: Dict[str, int] = {}
    if "adj_source" in daily.columns:
        by_source = {
            str(k): int(v)
            for k, v in daily.loc[의심.to_numpy(), "adj_source"]
            .fillna("(없음)").value_counts().items()
        }
    chain의심 = sum(v for k, v in by_source.items() if "chain" in k)

    axis["adj_suspect_rows"] = _metric(
        {"total": int(의심.sum()), "by_source": by_source},
        "chain > 0 (우리가 편 구간이 KRX 와 어긋난 것)",
        _RED if chain의심 else _OK,
        f"gap > {요약.get('gap_tolerance', 1.0)}%p · 비교 가능 "
        f"{요약.get('comparable_rows', 0):,}행",
    )
    axis["extreme_rows"] = _metric(
        int(극단.sum()), "(붉지 않다 — 진짜 사건이다)", _OK,
        f"|일간수익률| > {요약.get('extreme_pct', 30.0)}% · 의심이 아니면 남긴다",
    )
    return axis


# ── 축 4. 거래량 급변 — 기준을 베끼지 않는다 ──────────────────────────────────

def _axis_volume(daily: Optional[pd.DataFrame]) -> Dict:
    """거래량 급변 — **KRX 시장경보 기준**(최근 5일 평균 대비 3배)으로 센다.

    ## 왜 붉게 두지 않나

    qlib 은 `large_step_threshold_volume=3` 으로 3배 초과를 붉게 본다. 우리도 세지만
    **판정은 하지 않는다.** 실측이 그 이유를 준다.

    실측 2026-09-09 · KOSPI 시총 상위 50 · 10배 초과 254행:

        자본변동 ±5거래일 안        38행 (15.0%)   액면분할·증자·감자
        is_extreme_return 과 겹침    0행 ( 0.0%)   완전히 독립된 축이다
        나머지                     216행          수급 사건

    상위를 열어 보면 전부 설명된다 — 2018-05-04 삼성전자 158배(50:1 액면분할 재상장),
    2021-02-24 SK바이오팜 73배(보호예수 해제 · −17.29%), 2023-12-20 HMM 50배(인수전
    · +19.91%), 2024-01-11 카카오페이 37배(+21.59%).

    **데이터 오류가 아니라 시장 사건이다.** KRX 자신도 이걸 "오류" 가 아니라 투자주의
    지정 요건으로 쓴다. 붉게 두면 배포마다 게이트가 막히고, 붉은불이 흔해지면 아무도
    보지 않게 된다 — `is_extreme_return` 을 "빼라" 가 아니라 "남겨라" 로 적은 것과
    같은 이유다.

    ## 기준선을 함께 적는 이유

    배수(3배)는 세 표준이 같은데 **무엇에 대한 3배인지가 다르다.** KRX 는 최근 5일
    평균, qlib 은 전일, 흔한 관행은 20일 중앙값이다. 우리 데이터에서 전체 비율이
    5.426% / 7.977% / 10.329% 로 **2배 가까이 갈린다.** 그래서 값과 함께 기준선을
    남긴다.
    """
    axis: Dict[str, Any] = {}
    if daily is None or daily.empty or {"code", "bas_dd", "volume"} - set(daily.columns):
        return {k: _metric(None, "반출본을 못 읽었다", _SKIP)
                for k in ("surge_rows", "surge_ratio_p999")}

    v = daily[["bas_dd", "code", "volume"]].sort_values(["code", "bas_dd"])
    # 체결이 있던 날만 기준선에 넣는다. 정지·무거래 0 을 평균에 섞으면 재개일이
    # 전부 급변으로 잡힌다.
    vol = v["volume"].where(v["volume"] > 0)
    # 🔴 `shift(1)` — 그 날을 뺀 과거만 본다. 자기 자신을 평균에 넣으면 급변이
    #    스스로를 희석해서 큰 값일수록 덜 잡힌다.
    기준선 = (vol.groupby(v["code"])
                 .transform(lambda s: s.shift(1)
                            .rolling(SURGE_WINDOW, min_periods=3).mean()))
    비율 = (vol / 기준선).replace([float("inf"), float("-inf")], pd.NA)
    잴수있음 = 비율.notna()
    급변 = int((비율 > SURGE_MULTIPLE).fillna(False).sum())

    axis["surge_rows"] = _metric(
        급변, "(붉지 않다 — 시장 사건이다)", _OK,
        f"KRX 시장경보 기준: 최근 {SURGE_WINDOW}일 평균 대비 {SURGE_MULTIPLE:g}배 초과 · "
        f"잴 수 있는 {int(잴수있음.sum()):,}행의 "
        f"{급변 / max(int(잴수있음.sum()), 1):.3%} · "
        "`is_extreme_return` 과 겹치지 않는 독립된 축이다",
    )
    axis["surge_ratio_p999"] = _metric(
        round(float(비율[잴수있음].quantile(0.999)), 2) if 잴수있음.any() else None,
        "(붉지 않다 · 분포를 남긴다)", _OK,
        "99.9% 분위수 — 임계를 옮길 일이 생기면 이 값을 근거로 삼는다",
    )
    return axis


# ── 축 5. 달력 정합 ───────────────────────────────────────────────────────────

def _axis_calendar(conn, daily: Optional[pd.DataFrame]) -> Dict:
    """달력 — `trading_calendar` 는 날짜마다 시장 수만큼 행이 있다. DISTINCT 로 센다."""
    axis: Dict[str, Any] = {}
    if conn is None:
        return {"calendar_head": _metric(None, "DB 를 안 줬다", _SKIP)}

    달력최신 = conn.execute("SELECT MAX(bas_dd) FROM trading_calendar").fetchone()[0]
    시세최신 = conn.execute("SELECT MAX(bas_dd) FROM daily_price").fetchone()[0]
    axis["calendar_head"] = _metric(
        {"calendar": 달력최신, "daily_price": 시세최신},
        "두 값이 다르다 (next_session 이 선다)",
        _OK if 달력최신 == 시세최신 else _RED,
    )

    # 🔴 날짜마다 시장 수만큼 행이 있다. COUNT(*) 로 세면 실제 거래일 수의 몇 배가 나온다.
    시장별 = {
        str(m): int(n) for m, n in conn.execute(
            "SELECT market, COUNT(DISTINCT bas_dd) FROM trading_calendar GROUP BY market"
        ).fetchall()
    }
    고른가 = len(set(시장별.values())) <= 1
    axis["sessions_by_market"] = _metric(
        시장별, "시장마다 거래일 수가 다르다",
        _OK if 고른가 else _RED,
        "COUNT(DISTINCT bas_dd) 로 센다 — 날짜마다 시장 수만큼 행이 있다",
    )

    if daily is not None and not daily.empty:
        반출최신 = str(daily["bas_dd"].max())
        axis["export_head"] = _metric(
            반출최신, "(기록만 — 개발구간 상한이라 달력 최신일과 다른 것이 맞다)", _OK,
        )
    return axis


# ── 축 6. 중복 — 합계가 아니라 축마다 ─────────────────────────────────────────

def _axis_duplicate(conn, daily: Optional[pd.DataFrame]) -> Dict:
    """중복 — **같은 표를 두 축으로 세서 비교**한다. 합계는 덮어쓰기를 못 잡는다."""
    axis: Dict[str, Any] = {}
    if conn is None:
        return {"daily_price_key": _metric(None, "DB 를 안 줬다", _SKIP)}

    총, 고유 = conn.execute(
        "SELECT COUNT(*), COUNT(DISTINCT bas_dd || '|' || code) FROM daily_price"
    ).fetchone()
    axis["daily_price_key"] = _metric(
        {"rows": int(총), "distinct": int(고유)},
        "두 값이 다르다",
        _OK if 총 == 고유 else _RED,
    )

    # 🔴 이 원장이 생긴 이유. 기본키에 시장이 없어 KOSPI·KOSDAQ 이 같은 이름 업종지수로
    #    서로를 덮었다. **시장을 뺀 축**과 **넣은 축**의 값이 다르면 덮어쓰기가 있었다는 뜻이다.
    시장없이, 시장포함 = conn.execute(
        "SELECT COUNT(DISTINCT bas_dd || '|' || index_name), "
        "       COUNT(DISTINCT bas_dd || '|' || index_name || '|' || index_class) "
        "FROM index_price"
    ).fetchone()
    axis["index_price_two_axes"] = _metric(
        {"without_market": int(시장없이), "with_market": int(시장포함)},
        "두 축의 값이 다르다 (한 시장이 다른 시장을 덮었다)",
        _OK if 시장없이 == 시장포함 else _RED,
        "합계로는 못 잡는다 — 09-07 에 40,324행이 덮였을 때 COUNT(*) 는 오히려 늘었다",
    )

    종수 = {
        str(c): int(n) for c, n in conn.execute(
            "SELECT index_class, COUNT(DISTINCT index_name) FROM index_price "
            "GROUP BY index_class"
        ).fetchall()
    }
    axis["index_names_by_class"] = _metric(
        종수, "전 반출본보다 종 수가 줄었다 (덮어쓰기 흔적)", _OK,
        "줄었는지는 previous 와 대조해 판정한다",
    )

    try:
        t총, t고유 = conn.execute(
            "SELECT COUNT(*), COUNT(DISTINCT text_sha || '|' || model_id) FROM text_signal"
        ).fetchone()
        axis["text_signal_key"] = _metric(
            {"rows": int(t총), "distinct": int(t고유)}, "두 값이 다르다",
            _OK if t총 == t고유 else _RED,
        )
    except Exception:                                     # noqa: BLE001 — 표가 없을 수 있다
        axis["text_signal_key"] = _metric(None, "표가 없다", _SKIP)

    if daily is not None and not daily.empty and {"bas_dd", "code"} <= set(daily.columns):
        중복 = int(daily.duplicated(["bas_dd", "code"]).sum())
        axis["export_daily_key"] = _metric(
            중복, "> 0", _RED if 중복 else _OK, "반출본 자체의 키 중복",
        )
    return axis


# ── 축 7. 시점 · 봉인 ─────────────────────────────────────────────────────────

def _axis_sealing(manifest: Optional[Dict], daily: Optional[pd.DataFrame],
                  holdout_start: str) -> Dict:
    """봉인 — 반출본에 홀드아웃 구간 행이 한 줄도 없어야 한다. 반출 게이트가 이미 막지만
    원장은 **그 사실을 남긴다** (막는 것과 적는 것은 다르다)."""
    axis: Dict[str, Any] = {}
    if daily is not None and not daily.empty:
        최대 = str(daily["bas_dd"].max())
        axis["export_max_bas_dd"] = _metric(
            {"max_bas_dd": 최대, "holdout_start": holdout_start},
            ">= holdout_start",
            _OK if 최대 < holdout_start else _RED,
        )
    else:
        axis["export_max_bas_dd"] = _metric(None, "반출본을 못 읽었다", _SKIP)

    if manifest:
        axis["manifest_files"] = _metric(
            len(manifest.get("files", [])), "(기록만)", _OK,
        )
        axis["manifest_holdout_start"] = _metric(
            manifest.get("holdout_start"), "원장과 다르다",
            _OK if manifest.get("holdout_start") == holdout_start else _RED,
        )
    return axis


# ── 원장 ──────────────────────────────────────────────────────────────────────

def build_quality_ledger(
    outbox_dir,
    *,
    conn=None,
    holdout_start: str = HOLDOUT_START,
    previous: Optional[Dict] = None,
    run_id: Optional[str] = None,
    daily: Optional[pd.DataFrame] = None,
) -> Dict[str, Any]:
    """반출 폴더 하나를 재서 원장 dict 을 만든다.

    `outbox_dir` 는 `MANIFEST.json` · `PROFILE.json` 이 있는 폴더다. `conn` 을 주면 DB 축
    (달력·중복)까지 재고, 안 주면 그 축은 `skip` 이다. `daily` 를 주면 반출본 parquet 을
    다시 읽지 않는다 — 320MB 라 부르는 쪽이 이미 갖고 있으면 넘기는 편이 빠르다.

    **막지 않는다.** `status` 가 `red` 여도 예외를 던지지 않는다 — 붉다는 사실을 실어
    돌려줄 뿐이다. 막는 것은 반출 게이트의 몫이다.
    """
    outbox = Path(outbox_dir)
    if not outbox.is_dir():
        raise FileNotFoundError(f"반출 폴더가 없다: {outbox}")

    manifest = _read_json(outbox / "MANIFEST.json")
    profile = _read_json(outbox / "PROFILE.json")

    if daily is None:
        daily = _read_export_daily(outbox)

    axes = {
        "missing": _axis_missing(profile, daily),
        "validity": _axis_validity(daily),
        "outlier": _axis_outlier(daily),
        "volume": _axis_volume(daily),
        "calendar": _axis_calendar(conn, daily),
        "duplicate": _axis_duplicate(conn, daily),
        "sealing": _axis_sealing(manifest, daily, holdout_start),
    }

    붉은 = [f"{축}.{이름}" for 축, 표 in axes.items()
           for 이름, m in 표.items() if m.get("status") == _RED]

    ledger: Dict[str, Any] = {
        "generated_at": datetime.now(KST).isoformat(timespec="seconds"),
        "run_id": run_id or outbox.name,
        "git_sha": _git_sha(),
        "outbox": str(outbox).replace("\\", "/"),
        "holdout_start": holdout_start,
        "axes": axes,
        "red": 붉은,
        "status": _RED if 붉은 else _OK,
    }
    if previous:
        ledger["previous"] = _diff_against(previous, ledger)
    return ledger


def _read_json(path: Path) -> Optional[Dict]:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:                                     # noqa: BLE001
        return None


def _read_export_daily(outbox: Path) -> Optional[pd.DataFrame]:
    """반출본 시세를 읽는다 — 품질 판정에 쓰는 칸만. 전량은 320MB 라 칸을 좁힌다."""
    path = outbox / "full" / "daily_price_dev.parquet"
    if not path.exists():
        return None
    칸 = ["bas_dd", "code", "close", "adj_close", "change_rate", "volume",
         "adj_open", "adj_high", "adj_low", "adj_source", "market"]
    try:
        return pd.read_parquet(path, columns=칸)
    except Exception:                                     # noqa: BLE001 — 칸이 다를 수 있다
        try:
            return pd.read_parquet(path)
        except Exception:                                 # noqa: BLE001
            return None


def _previous_values(previous: Dict) -> Dict[str, Any]:
    """지난 것의 값만 `축.지표 -> 값` 으로 편다.

    원장 전체(`axes` 안에 `{"value": …}`)로 올 수도 있고 이력 한 줄(`summary` 안에
    값이 바로)로 올 수도 있다. 둘 다 받는다 — 부르는 쪽이 어느 것을 갖고 있든
    "지난 반출본" 이라는 뜻은 같기 때문이다.
    """
    값: Dict[str, Any] = {}
    for 축, 표 in (previous.get("axes") or {}).items():
        for 이름, m in (표 or {}).items():
            값[f"{축}.{이름}"] = m.get("value") if isinstance(m, dict) else m
    for 축, 표 in (previous.get("summary") or {}).items():
        for 이름, v in (표 or {}).items():
            값.setdefault(f"{축}.{이름}", v)
    return 값


def _diff_against(previous: Dict, now: Dict) -> Dict[str, Any]:
    """지난 원장과 달라진 지표만 추린다. 카드의 "변화" 열이 이것을 읽는다."""
    옛값 = _previous_values(previous)
    바뀐: Dict[str, Any] = {}
    for 축, 표 in now.get("axes", {}).items():
        for 이름, m in 표.items():
            키 = f"{축}.{이름}"
            if 키 not in 옛값 or 옛값[키] == m.get("value"):
                continue
            바뀐[키] = {"before": 옛값[키], "after": m.get("value")}
    return {
        "run_id": previous.get("run_id"),
        "generated_at": previous.get("generated_at"),
        "status": previous.get("status"),
        "changed": 바뀐,
    }


def write_quality_ledger(outbox_dir, ledger: Dict[str, Any]) -> Path:
    """원장을 `MANIFEST.json` 옆에 쓴다. 반출본과 같이 움직여야 하기 때문이다."""
    path = Path(outbox_dir) / LEDGER_NAME
    path.write_text(json.dumps(ledger, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def append_history(ledger: Dict[str, Any], path: Path = HISTORY_PATH) -> Optional[Path]:
    """이력에 한 줄 더한다 — **append-only.** 같은 `run_id` 는 두 번 쓰지 않는다."""
    path.parent.mkdir(parents=True, exist_ok=True)
    run_id = ledger.get("run_id")
    if path.exists():
        for 줄 in path.read_text(encoding="utf-8").splitlines():
            if not 줄.strip():
                continue
            try:
                if json.loads(줄).get("run_id") == run_id:
                    return None                           # 이미 있다 — 고치지 않는다
            except json.JSONDecodeError:
                continue
    한줄 = {
        "run_id": run_id,
        "generated_at": ledger.get("generated_at"),
        "git_sha": ledger.get("git_sha"),
        "status": ledger.get("status"),
        "red": ledger.get("red", []),
        "summary": {축: {이름: m.get("value") for 이름, m in 표.items()}
                    for 축, 표 in ledger.get("axes", {}).items()},
    }
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(한줄, ensure_ascii=False) + "\n")
    return path


def read_last_history(path: Path = HISTORY_PATH) -> Optional[Dict]:
    """이력의 마지막 줄. 다음 반출본이 "무엇이 달라졌나" 를 재는 기준이 된다."""
    if not path.exists():
        return None
    줄들 = [줄 for 줄 in path.read_text(encoding="utf-8").splitlines() if 줄.strip()]
    for 줄 in reversed(줄들):
        try:
            return json.loads(줄)
        except json.JSONDecodeError:
            continue
    return None


# ── 카드 ──────────────────────────────────────────────────────────────────────

_축이름 = {
    "missing": "결측",
    "validity": "유효성",
    "outlier": "이상치",
    "volume": "거래량 급변",
    "calendar": "달력 정합",
    "duplicate": "중복",
    "sealing": "시점·봉인",
}
_표시 = {_OK: "✅", _RED: "🔴", _WARN: "⚠️", _SKIP: "—"}

#: 카드에 반드시 함께 나가는 플래그 사용법. 신장환 님이 이슈 #168 에서 짚은 위험이다 —
#: `is_extreme_return` 은 이름만 보면 "극단이니 빼라" 로 읽히지만 **반대다.** 거르라는
#: 플래그가 아니라 "진짜 사건이니 남겨라" 는 표시다. 한 줄이 없으면 거꾸로 거른다.
FLAG_GUIDE = """### 품질 플래그 4칸을 어떻게 쓰나

`daily_price_dev.parquet` 에 품질 플래그 네 칸이 함께 실려 있습니다. 같은 표본을 쓰려면
**아래 한 줄만** 지키면 됩니다.

```python
df = df[~df["is_adj_suspect"]]        # 걸러야 하는 것은 이것 하나뿐입니다
```

| 칸 | 무엇 | 어떻게 쓰나 |
|---|---|---|
| `is_adj_suspect` | KRX 등락률과 우리 수정주가 수익률이 **1%p 넘게 어긋난** 행 | 🔴 **거른다** |
| `is_extreme_return` | 일간수익률 절댓값이 **30% 를 넘는** 행 | ✅ **남긴다.** 거르지 않습니다 |
| `adj_return_1d` | 수정종가로 계산한 일간수익률 | 위 두 플래그의 근거값 |
| `adj_change_rate_gap` | KRX 등락률과의 차이(%p) | `is_adj_suspect` 의 근거값 |

> ⚠️ **`is_extreme_return` 을 거르지 마십시오.** 이름과 달리 "빼라" 가 아니라 **"진짜
> 일어난 사건이니 남겨라"** 는 표시입니다. 무상증자 권리락·거래정지 해제·상한가 연속은
> 하루 30% 넘게 움직이지만 오류가 아니라 **모델이 배워야 할 사건**입니다. 크기로 자르면
> 그것까지 지워집니다. 걸러야 하는 것은 **KRX 와 어긋난** `is_adj_suspect` 뿐입니다.

팀 기준선은 `is_adj_suspect` 만 제외한 표본입니다. 이 칸들이 파일에 함께 있는 이유가
그것입니다 — 각자 같은 함수를 다시 돌리지 않아도 **같은 표본**이 되도록.
"""

#: 🔴 작은 벌과 큰 벌이 **다른 규칙 위에 서 있다**는 사실. 이슈 #186 ① 이 짚은 것이다.
#: 칸 구성만 보고는 알 수 없고, 큰 벌로 학습하면 작은 벌과 결과가 갈리는데 경고가 없다.
SAMPLE_GUIDE = """### 🔴 큰 벌과 작은 벌은 **다른 표본**입니다

같은 자료를 두 벌로 냅니다. **둘은 표본이 다릅니다.**

| | 파일 | 정리매매·거래정지·신규상장 |
|---|---|---|
| **큰 벌** | `full/daily_price_dev.parquet` | **안 뺐습니다** — 그대로 있습니다 |
| **작은 벌** | `small/stocks_sample30_train_dev.csv` | **뺐습니다** |

큰 벌로 그냥 학습하면 작은 벌·팀 기준선과 **표본이 달라집니다.** 개발구간 7,888,945행
중 **251,282행(3.185%)** 이 그 차이입니다.

같은 표본으로 맞추려면 한 줄입니다.

```python
df = df[~(df.is_liquidation | df.is_halted | df.is_first_listing)]
```

| 칸 | 무엇 | 개발구간 |
|---|---|---:|
| `is_liquidation` | 정리매매 — 체결이 끊기기 직전 10체결일 | 17,973 (0.228%) |
| `is_halted` | 거래정지 — 그 행에 체결이 없었다 | 231,808 (2.938%) |
| `is_first_listing` | 신규상장 첫 거래일 — 등락률이 공모가 기준 | 1,501 (0.019%) |

> 🔴 **이 셋을 피처로 넣지 마십시오.** `is_liquidation` 은 *"이 뒤로 체결이 끊긴다"* 를
> 보고 매깁니다. 그 시점에는 알 수 없는 사실이라 피처로 쓰면 곧 미래참조입니다.
> **표본을 고르는 데만** 쓰십시오.

### 유니버스를 좁히려면 — 주권종류 3칸

`kind_stkcert_tp_nm` · `secugrp_nm` · `sect_tp_nm` 이 그 날의 KRX 판정을 그대로 담고
있습니다. **무엇을 뺄지는 정해 두지 않았습니다** — 쓰는 쪽이 고르십시오.

```python
# 보통주만 (KOSPI200 방법론 · CRSP share code 10/11 에 해당)
df = df[df.kind_stkcert_tp_nm == "보통주"]

# KOSPI200 지수 방법론에 맞추려면 — 리츠·선박투자회사·SPAC·관리종목도 뺍니다
빼기 = (
    df.secugrp_nm.isin(["부동산투자회사", "선박투자회사", "투자회사",
                        "사회간접자본투융자회사", "외국주권",
                        "주식예탁증권", "주식예탁증서"])
    | df.sect_tp_nm.astype(str).str.contains("SPAC|관리종목|투자주의환기|외국기업")
)
df = df[~빼기]
```

> ⚠️ 이렇게 걸러도 **KOSPI200 과 똑같아지지는 않습니다.** 방법론은 "유동주식비율 10%
> 미만"과 "상장 후 6개월 미경과"도 제외하는데, 유동주식비율은 우리가 수집하지 않습니다.
> 그래서 판정을 한 칸으로 뭉치지 않고 **원문 값 그대로** 실어 보냅니다.
"""


def _fmt_value(값: Any) -> str:
    """지표 값을 표 한 칸에 넣는다. 축별로 나눈 dict 은 `키 값` 을 가운뎃점으로 잇는다.

    `by_source` 처럼 dict 안에 dict 이 오는 자리가 있어 한 단계 더 편다 — 파이썬
    표기(`{'fdr': 1193}`)가 그대로 카드에 나가면 읽는 사람이 코드를 보게 된다.
    """
    if 값 is None:
        return "—"
    if isinstance(값, bool):
        return "예" if 값 else "아니오"
    if isinstance(값, int):
        return f"{값:,}"
    if isinstance(값, dict):
        조각 = []
        for k, v in 값.items():
            if isinstance(v, dict):
                안 = " · ".join(f"{kk} {vv:,}" if isinstance(vv, int) else f"{kk} {vv}"
                               for kk, vv in v.items())
                조각.append(f"{k}({안})" if 안 else f"{k} —")
            else:
                조각.append(f"{k} {v:,}" if isinstance(v, int) else f"{k} {v}")
        return " · ".join(조각)
    return str(값)


def render_card_section(ledger: Dict[str, Any]) -> str:
    """HF 데이터셋 카드에 붙일 "품질 원장" 절을 만든다.

    붉은 것은 붉게 적어 내보낸다 — 숨기지 않는 것이 이 원장의 목적이다.
    """
    머리 = ledger.get("status", _OK)
    줄: list[str] = [
        "## 데이터 품질 원장",
        "",
        f"`run_id` **{ledger.get('run_id')}** · 잰 시각 {ledger.get('generated_at')} · "
        f"코드 `{ledger.get('git_sha') or '(모름)'}` · 판정 **{_표시.get(머리, '')} {머리}**",
        "",
        "> 이 표는 **막지 않고 적습니다.** 반출을 막는 것은 검증기와 게이트이고, 원장은 "
        "붉은 것을 붉게 적어 함께 내보냅니다 — 알면서 내보내는 것을 숨기지 않기 위해서입니다.",
        "",
        # 🔴 `note` 를 빼면 값이 무슨 뜻인지 알 수 없다. 거래량 급변의 "3배" 는
        #    기준선(KRX 5일 평균 / qlib 전일 / 20일 중앙값)에 따라 전체 비율이
        #    5.426% ~ 10.329% 로 갈린다. 근거를 같이 실어야 다음 사람이 배수만
        #    베끼지 않는다.
        "| 축 | 지표 | 값 | 붉은 기준 | 판정 | 근거 |",
        "|---|---|---|---|---|---|",
    ]
    변화 = (ledger.get("previous") or {}).get("changed", {})
    for 축 in AXES:
        표 = ledger.get("axes", {}).get(축, {})
        for 이름, m in 표.items():
            값글 = _fmt_value(m.get("value"))
            줄.append(
                f"| {_축이름.get(축, 축)} | `{이름}` | {값글} | {m.get('red_if', '')} | "
                f"{_표시.get(m.get('status'), '')} | {m.get('note', '')} |"
            )
    if 변화:
        줄 += ["", "### 지난 반출본에서 달라진 것", "",
              "| 지표 | 전 | 후 |", "|---|---|---|"]
        for 이름, d in 변화.items():
            줄.append(f"| `{이름}` | {d.get('before')} | {d.get('after')} |")
    붉은 = ledger.get("red", [])
    if 붉은:
        줄 += ["", f"🔴 **붉은 항목 {len(붉은)}개** — " + " · ".join(f"`{x}`" for x in 붉은)]
    줄 += ["", SAMPLE_GUIDE, "", FLAG_GUIDE]
    return "\n".join(줄) + "\n"


__all__ = [
    "AXES",
    "FLAG_GUIDE",
    "SAMPLE_GUIDE",
    "HISTORY_PATH",
    "LEDGER_NAME",
    "append_history",
    "build_quality_ledger",
    "read_last_history",
    "render_card_section",
    "write_quality_ledger",
]
