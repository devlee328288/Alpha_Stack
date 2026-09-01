"""규격의 `lookahead` 가 적어 둔 "언제부터 알 수 있었나" 를 실제로 계산한다.

**왜 반입 단계에서 하는가.** 이 값들은 규격상 required 이거나(거시의 `known_from`) 조인 열쇠가
되는데(뉴스의 `eff_dd`), 팀원이 손으로 채우기에는 규칙이 까다롭다. 안 채워 오면 통째로
격리되고, 채워 오면 그 값이 맞는지 우리가 확인해야 한다. 그래서 **비었으면 채우고, 채워져
있으면 검사한다.**

🔴 **채워 온 값을 그대로 믿지 않는다.** 규격에 그렇게 적힌 이유가 있다 — `eff_dd` 를 발행일과
같게 적어 오면 장중 기사를 그날 시가에 쓰게 되고, 그건 정확히 미래참조다. 그런데 그 파일은
겉보기에 아무 문제가 없다. 값이 있고, 형식도 맞고, 규칙도 통과한다. 우리가 같은 계산을 해서
**대조하지 않으면 잡을 방법이 없다.**

대조의 방향이 한쪽뿐이라는 점이 중요하다. 채워 온 값이 우리 계산보다 **이르면** 격리하고,
**늦으면** 통과시킨다. 늦게 잡은 것은 자료를 조금 버릴 뿐이지만 이르게 잡은 것은 성능을
부풀린다. 틀리더라도 **언제나 늦는 쪽으로만** 틀리게 둔다.

무엇을 파생하나
---------------
| 규격 | 칸 | 근거 |
|------|-----|------|
| `news` | `eff_dd` | 발행 시각 + 08:30 경계 + 실측 거래일 달력 |
| `macro` | `known_from` · `known_from_basis` | 발표일이 있으면 그것, 없으면 참조기간 시작 + 지연 |

`ohlcv_stock`·`ohlcv_index` 는 파생하지 않는다 — 규격이 `enforcedAt: "supply 계층"` 이라고
못 박아 두었고, 실제로 `supply/clock.py` 의 `known_at()` 하나가 둘을 함께 자른다. 여기서 또
계산하면 같은 규칙이 두 군데 살게 되고, 두 군데는 언젠가 어긋난다.

`financial` 도 파생하지 않는다. 규격의 `onMissingTimeField` 는 `rcept_no` 로 DART 를 다시
불러 `rcept_dt` 를 채우라는 것인데, **반입 검사기가 네트워크를 타면 안 된다** — 팀원 파일을
검사하는 일이 하루 한도를 태우고, 한도가 떨어지면 검사 자체가 실패한다. 그건 수집기의 일이다.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import pandas as pd

from common.trading_calendar import CalendarOutOfRange, next_session

#: 장 시작 전으로 보는 경계. 규격 `news.lookahead.cutoffKST` 와 같은 값이다.
#: 08:30 인 이유는 시가 단일가 호가가 그때부터 쌓이기 때문이다 — 09:00 이 아니다.
NEWS_CUTOFF = "083000"


class DeriveError(ValueError):
    """규격이 파생에 필요한 것을 안 갖췄다 — 자료가 아니라 규격의 문제다."""


@dataclass
class DeriveEntry:
    """파생 한 칸의 결과 요약."""

    column: str
    filled: int = 0                 # 비어 있어서 우리가 채운 수
    verified: int = 0               # 채워져 있고 우리 계산과 맞은 수
    too_early: int = 0              # 채워져 있는데 우리 계산보다 이른 수 → 격리
    undecidable: int = 0            # 달력 밖 등으로 계산조차 못 한 수 → 격리
    samples: List[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "column": self.column,
            "filled": self.filled,
            "verified": self.verified,
            "too_early": self.too_early,
            "undecidable": self.undecidable,
            "samples": self.samples,
        }


@dataclass
class DeriveResult:
    """파생 전체의 결과. `blocked` 가 참인 행은 부르는 쪽이 격리한다."""

    frame: pd.DataFrame
    blocked: pd.Series
    reasons: pd.Series
    entries: List[DeriveEntry] = field(default_factory=list)

    def to_list(self) -> List[dict]:
        return [e.to_dict() for e in self.entries]


def _sample(entry: DeriveEntry, limit: int = 20, **fields) -> None:
    """표본은 정제기와 같은 20건까지만 — 보고서 크기를 입력 크기와 무관하게 둔다."""
    if len(entry.samples) < limit:
        entry.samples.append(fields)


# ==================================================
# 1. 뉴스 — 그 기사를 처음 쓸 수 있는 거래일
# ==================================================
_ISO_DATETIME = re.compile(r"^(\d{4})-(\d{2})-(\d{2})[T ](\d{2}):(\d{2}):(\d{2})")


def _split_kst(value) -> Optional[Tuple[str, str]]:
    """KST ISO 문자열을 `(YYYYMMDD, HHMMSS)` 로 가른다. 못 읽으면 None."""
    if not isinstance(value, str):
        return None
    match = _ISO_DATETIME.match(value.strip())
    if match is None:
        return None
    year, month, day, hour, minute, second = match.groups()
    return year + month + day, hour + minute + second


def derive_news_eff_dd(frame: pd.DataFrame, *, db_path=None) -> DeriveResult:
    """`pub_dt` 로부터 `eff_dd` 를 정한다.

    배정 규칙(규격 `news.lookahead.배정규칙` 그대로):

    - 발행일 D 가 거래일이고 시각이 08:30 **미만**이면 → `eff_dd = D` (그날 시가에 반영)
    - D 가 거래일인데 08:30 **이상**이면 → D 의 다음 거래일 (장중·장마감 후 기사가 여기로)
    - D 가 비거래일이면 → D 이후 첫 거래일
    - 시각이 정확히 `00:00:00` 이면 **날짜만 있는 자료**로 보고 다음 거래일로 민다

    마지막 규칙이 있는 이유: 날짜만 적힌 자료를 자정으로 읽으면 "08:30 미만" 에 걸려 그날
    시가에 쓰이게 된다. 실제로는 그 기사가 몇 시에 나왔는지 모르는 것이라, 모르는 쪽을
    **늦게** 잡는다.
    """
    entry = DeriveEntry(column="eff_dd")
    blocked = pd.Series(False, index=frame.index)
    reasons = pd.Series("", index=frame.index, dtype="object")

    if "pub_dt" not in frame.columns:
        raise DeriveError("뉴스 규격에 pub_dt 가 없다 — eff_dd 를 정할 근거가 없다.")

    existing = frame["eff_dd"] if "eff_dd" in frame.columns else pd.Series(None, index=frame.index)
    computed: List[Optional[str]] = []

    for position, raw in enumerate(frame["pub_dt"]):
        index = frame.index[position]
        parts = _split_kst(raw)
        if parts is None:
            computed.append(None)
            entry.undecidable += 1
            blocked.loc[index] = True
            reasons.loc[index] = "pub_dt 를 시각으로 읽을 수 없어 eff_dd 를 정할 수 없다"
            _sample(entry, pub_dt=str(raw), verdict="undecidable")
            continue

        published_day, clock = parts
        # 장 시작 전이면서 자정 표기가 아닐 때만 그날을 쓴다.
        before_open = clock < NEWS_CUTOFF and clock != "000000"
        try:
            effective = next_session(published_day, db_path, inclusive=before_open)
        except CalendarOutOfRange as error:
            computed.append(None)
            entry.undecidable += 1
            blocked.loc[index] = True
            head = error.args[0].splitlines()[0]
            reasons.loc[index] = f"거래일 달력 밖이라 eff_dd 를 정할 수 없다: {head}"
            _sample(entry, pub_dt=str(raw), verdict="out_of_calendar")
            continue

        computed.append(effective)
        given = existing.iloc[position]
        if given is None or (isinstance(given, float) and pd.isna(given)) or given is pd.NA:
            entry.filled += 1
            _sample(entry, pub_dt=str(raw), filled=effective)
        elif str(given) == effective:
            entry.verified += 1
        elif str(given) < effective:
            # 🔴 우리 계산보다 이르다 = 아직 못 쓸 자료를 쓸 수 있다고 적어 온 것이다.
            entry.too_early += 1
            blocked.loc[index] = True
            reasons.loc[index] = (
                f"eff_dd 가 규칙보다 이르다 (적힌 값 {given} · 규칙 {effective}) — 미래참조"
            )
            _sample(entry, pub_dt=str(raw), given=str(given), rule=effective, verdict="too_early")
        else:
            # 늦게 잡아 온 것은 통과시킨다. 자료를 조금 버릴 뿐 새는 방향이 아니다.
            entry.verified += 1

    out = frame.copy()
    out["eff_dd"] = computed if "eff_dd" not in frame.columns else [
        c if _is_blank(existing.iloc[i]) else existing.iloc[i]
        for i, c in enumerate(computed)
    ]
    return DeriveResult(out, blocked, reasons, [entry])


def _is_blank(value) -> bool:
    if value is None or value is pd.NA:
        return True
    if isinstance(value, float) and pd.isna(value):
        return True
    return isinstance(value, str) and not value.strip()


# ==================================================
# 2. 거시 — 그 통계를 처음 알 수 있었던 날
# ==================================================
def derive_macro_known_from(frame: pd.DataFrame, spec: dict) -> DeriveResult:
    """`known_from` 과 `known_from_basis` 를 정한다.

    우선순위는 규격 그대로다.

    1. `release_date` 가 있으면 그것을 쓴다 → `basis = "release"`.
       실제 공표일이라 추정보다 언제나 낫다.
    2. 없으면 `period_start + delayFromPeriodStart[출처][주기]` → `basis = "estimate"`.

    🔴 **지연을 `period_end` 가 아니라 `period_start` 에 더한다.** 규격이 못 박은 대로다 —
    2026년 7월 CPI 는 7월 1일 기준으로 재야 실제 공표 시점과 맞는다. `period_end`(7월 31일)에
    같은 일수를 더하면 한 달을 통째로 더 기다려, 누설은 없지만 쓸 수 있는 자료를 버린다.

    ⚠️ 지연표가 **출처마다 다르다.** 미국 CPI 는 익월 11~13일에 나오는데 한국 관행(익월 2일)에
    맞춘 32일을 쓰면 9~11일을 미리 보게 된다. 그래서 `(출처, 주기)` 로 나눠 찾는다.
    """
    lookahead = (spec.get("x-alphastack") or {}).get("lookahead") or {}
    delays: Dict[str, Dict[str, int]] = lookahead.get("delayFromPeriodStart") or {}
    if not delays:
        raise DeriveError(
            "거시 규격에 delayFromPeriodStart 가 없다 — known_from 을 추정할 근거가 없다.\n"
            "  할 일: ingest/inbox/schemas/macro.json 의 lookahead 를 확인한다."
        )

    entry = DeriveEntry(column="known_from")
    basis_entry = DeriveEntry(column="known_from_basis")
    blocked = pd.Series(False, index=frame.index)
    reasons = pd.Series("", index=frame.index, dtype="object")

    for column in ("source", "freq", "period_start"):
        if column not in frame.columns:
            raise DeriveError(f"거시 규격의 {column} 칸이 없어 known_from 을 정할 수 없다.")

    existing = (frame["known_from"] if "known_from" in frame.columns
                else pd.Series(None, index=frame.index))
    release = (frame["release_date"] if "release_date" in frame.columns
               else pd.Series(None, index=frame.index))

    computed: List[Optional[str]] = []
    bases: List[Optional[str]] = []

    for position in range(len(frame)):
        index = frame.index[position]
        source = frame["source"].iloc[position]
        freq = frame["freq"].iloc[position]
        start = frame["period_start"].iloc[position]
        published = release.iloc[position]

        if not _is_blank(published):
            computed.append(str(published))
            bases.append("release")
            basis_entry.filled += 1
        else:
            delay = (delays.get(str(source)) or {}).get(str(freq))
            if delay is None or _is_blank(start):
                computed.append(None)
                bases.append(None)
                entry.undecidable += 1
                blocked.loc[index] = True
                reasons.loc[index] = (
                    f"known_from 을 정할 수 없다 — 출처 {source!r}·주기 {freq!r} 의 "
                    "지연을 모르거나 period_start 가 비었다"
                )
                _sample(entry, source=str(source), freq=str(freq), verdict="undecidable")
                continue
            try:
                anchor = datetime.strptime(str(start), "%Y%m%d")
            except ValueError:
                computed.append(None)
                bases.append(None)
                entry.undecidable += 1
                blocked.loc[index] = True
                reasons.loc[index] = f"period_start 를 날짜로 읽을 수 없다: {start!r}"
                _sample(entry, period_start=str(start), verdict="undecidable")
                continue
            computed.append((anchor + timedelta(days=int(delay))).strftime("%Y%m%d"))
            bases.append("estimate")
            basis_entry.filled += 1

        given = existing.iloc[position]
        derived = computed[-1]
        if _is_blank(given):
            entry.filled += 1
            _sample(entry, source=str(source), freq=str(freq), filled=derived, basis=bases[-1])
        elif str(given) == derived:
            entry.verified += 1
        elif str(given) < derived:
            entry.too_early += 1
            blocked.loc[index] = True
            reasons.loc[index] = (
                f"known_from 이 규칙보다 이르다 (적힌 값 {given} · 규칙 {derived}) — 미래참조"
            )
            _sample(entry, given=str(given), rule=derived, verdict="too_early")
        else:
            entry.verified += 1

    out = frame.copy()
    out["known_from"] = [
        c if _is_blank(existing.iloc[i]) else existing.iloc[i] for i, c in enumerate(computed)
    ]
    # basis 는 우리가 정한 근거이므로 **적혀 온 값보다 우리 판정을 쓴다** — 팀원이 추정치에
    # "release" 라고 적어 오면 그 자료가 실제보다 믿을 만해 보인다.
    out["known_from_basis"] = bases
    return DeriveResult(out, blocked, reasons, [entry, basis_entry])


# ==================================================
# 3. 갈래
# ==================================================
#: 종류별 파생기. 여기 없는 종류는 파생하지 않는다.
DERIVERS = {
    "news": "derive_news_eff_dd",
    "macro": "derive_macro_known_from",
}


def derive(kind: str, frame: pd.DataFrame, spec: dict, *, db_path=None) -> DeriveResult:
    """종류에 맞는 파생을 돌린다. 파생할 것이 없으면 그대로 돌려준다."""
    if kind == "news":
        return derive_news_eff_dd(frame, db_path=db_path)
    if kind == "macro":
        return derive_macro_known_from(frame, spec)
    empty = pd.Series(False, index=frame.index)
    return DeriveResult(frame, empty, pd.Series("", index=frame.index, dtype="object"), [])


__all__ = [
    "NEWS_CUTOFF",
    "DeriveError",
    "DeriveEntry",
    "DeriveResult",
    "DERIVERS",
    "derive",
    "derive_news_eff_dd",
    "derive_macro_known_from",
]
