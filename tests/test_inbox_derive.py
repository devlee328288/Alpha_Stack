"""시점 파생이 **미래를 당겨 보지 않는지** 확인한다.

이 파일의 시험은 대부분 *"일부러 미래참조를 심고 잡히는지"* 다. 파생은 값을 채우는 일이라
안 채우면 티가 나지만(required 위반) **틀리게 채우면 아무 데도 티가 안 난다** — 형식도 맞고
규칙도 통과하고 성능만 좋아진다. 그래서 잡는 쪽을 먼저 잰다.

거래일 달력은 **DB 를 읽지 않고** 작은 가짜 달력을 끼워 쓴다. 진짜 달력(4,097일)을 쓰면
테스트가 수집 상태에 따라 달라지고, `conftest.py` 가 막아 둔 진짜 DB 를 건드리게 된다.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from common import trading_calendar  # noqa: E402
from ingest.inbox.derive import (  # noqa: E402
    DeriveError,
    derive,
    derive_macro_known_from,
    derive_news_eff_dd,
)

#: 2021년 첫 주. 01-01(신정)과 주말이 빠져 있다 — 주말 규칙만으로는 못 만드는 달력이다.
가짜달력 = frozenset({
    "20201231", "20210104", "20210105", "20210106", "20210107", "20210108",
    "20210111", "20210112",
})


@pytest.fixture(autouse=True)
def 달력을_끼운다(monkeypatch):
    """실측 달력 대신 위의 작은 집합을 쓴다."""
    monkeypatch.setattr(trading_calendar, "_SESSION_CACHE", 가짜달력)
    monkeypatch.setattr(trading_calendar, "_SESSION_SPAN",
                        (min(가짜달력), max(가짜달력)))


@pytest.fixture
def 거시규격():
    path = ROOT / "ingest" / "inbox" / "schemas" / "macro.json"
    return json.loads(path.read_text(encoding="utf-8"))


# ==================================================
# 1. 뉴스 — 배정 규칙 다섯 가지
# ==================================================
def 배정(pub_dt: str, eff_dd=None):
    frame = pd.DataFrame({"pub_dt": [pub_dt], "eff_dd": [eff_dd]})
    result = derive_news_eff_dd(frame)
    return result.frame["eff_dd"].iloc[0], bool(result.blocked.iloc[0]), result


def test_장_시작_전_기사는_그날_시가에_반영된다():
    assert 배정("2021-01-04T08:15:00+09:00")[0] == "20210104"


def test_장중_기사는_다음_거래일로_민다():
    assert 배정("2021-01-04T09:30:00+09:00")[0] == "20210105"


def test_장_마감_후_기사도_다음_거래일로_모인다():
    """실적 발표가 장 마감 뒤에 몰리는데, 발행일로 조인하면 이 뭉치가 통째로 하루 빨라진다."""
    assert 배정("2021-01-04T16:00:00+09:00")[0] == "20210105"


def test_비거래일_기사는_그_이후_첫_거래일로_간다():
    # 01-02 는 토요일, 01-03 은 일요일, 01-01 은 신정이라 휴장.
    assert 배정("2021-01-02T10:00:00+09:00")[0] == "20210104"


def test_자정_표기는_날짜만_있는_자료로_보고_다음_거래일로_민다():
    """🔴 자정으로 읽으면 "08:30 미만" 에 걸려 그날 시가에 쓰인다. 모르는 쪽을 늦게 잡는다."""
    assert 배정("2021-01-04T00:00:00+09:00")[0] == "20210105"


def test_주말_규칙만으로는_못_잡는_공휴일을_실측_달력이_잡는다():
    """01-01 은 **금요일**이다. 주말만 거르는 달력은 이 날을 거래일로 본다."""
    assert 배정("2021-01-01T08:00:00+09:00")[0] == "20210104"


# ==================================================
# 2. 뉴스 — 채워 온 값 검사
# ==================================================
def test_발행일과_같게_적어_온_eff_dd_는_격리된다():
    """장 마감 후 기사에 발행일을 적으면 그날 시가에 쓰게 된다 — 정확히 미래참조다."""
    value, blocked, result = 배정("2021-01-04T16:00:00+09:00", eff_dd="20210104")
    assert blocked is True
    assert "미래참조" in result.reasons.iloc[0]
    assert result.entries[0].too_early == 1


def test_규칙과_같은_값을_적어_오면_통과한다():
    _, blocked, result = 배정("2021-01-04T16:00:00+09:00", eff_dd="20210105")
    assert blocked is False
    assert result.entries[0].verified == 1


def test_늦게_잡아_온_값은_통과시킨다():
    """늦게 잡은 것은 자료를 조금 버릴 뿐 새는 방향이 아니다."""
    _, blocked, result = 배정("2021-01-04T16:00:00+09:00", eff_dd="20210111")
    assert blocked is False
    assert result.entries[0].verified == 1


def test_달력_밖_날짜는_지어내지_않고_격리한다():
    _, blocked, result = 배정("2030-01-04T09:00:00+09:00")
    assert blocked is True
    assert result.entries[0].undecidable == 1


def test_읽을_수_없는_발행시각은_격리한다():
    _, blocked, result = 배정("어제쯤")
    assert blocked is True
    assert result.entries[0].undecidable == 1


def test_pub_dt_가_없으면_규격_잘못으로_세운다():
    with pytest.raises(DeriveError, match="pub_dt"):
        derive_news_eff_dd(pd.DataFrame({"title": ["기사"]}))


# ==================================================
# 3. 거시 — 지연표
# ==================================================
def 거시(거시규격, **columns):
    frame = pd.DataFrame({k: [v] for k, v in columns.items()})
    result = derive_macro_known_from(frame, 거시규격)
    return (result.frame["known_from"].iloc[0],
            result.frame["known_from_basis"].iloc[0],
            bool(result.blocked.iloc[0]), result)


def test_발표일이_있으면_그것을_쓴다(거시규격):
    value, basis, blocked, _ = 거시(
        거시규격, source="FRED", freq="M", period_start="20210701",
        release_date="20210813", known_from=None)
    assert (value, basis, blocked) == ("20210813", "release", False)


def test_발표일이_없으면_참조기간_시작에_지연을_더한다(거시규격):
    value, basis, _, _ = 거시(
        거시규격, source="ECOS", freq="M", period_start="20210701",
        release_date=None, known_from=None)
    assert (value, basis) == ("20210802", "estimate"), "07-01 + 32일"


def test_지연은_출처마다_다르다(거시규격):
    """🔴 미 CPI 는 익월 11~13일에 나온다. 한국 관행(32일)을 쓰면 9~11일을 미리 본다."""
    한국, _, _, _ = 거시(거시규격, source="ECOS", freq="M", period_start="20210701",
                       release_date=None, known_from=None)
    미국, _, _, _ = 거시(거시규격, source="FRED", freq="M", period_start="20210701",
                       release_date=None, known_from=None)
    assert 한국 == "20210802"
    assert 미국 == "20210815"
    assert 미국 > 한국, "FRED 쪽이 더 늦어야 한다"


def test_지연을_period_end_가_아니라_period_start_에_더한다(거시규격):
    """period_end 에 더하면 한 달을 통째로 더 기다려 쓸 수 있는 자료를 버린다."""
    value, _, _, _ = 거시(거시규격, source="ECOS", freq="M", period_start="20210701",
                        period_end="20210731", release_date=None, known_from=None)
    assert value == "20210802", "07-31 기준이면 09-01 이 됐을 것이다"


def test_한국_관행으로_미리_적어_온_known_from_은_격리된다(거시규격):
    _, _, blocked, result = 거시(
        거시규격, source="FRED", freq="M", period_start="20210701",
        release_date=None, known_from="20210802")
    assert blocked is True
    assert result.entries[0].too_early == 1
    assert "미래참조" in result.reasons.iloc[0]


def test_모르는_출처는_지어내지_않고_격리한다(거시규격):
    _, _, blocked, result = 거시(거시규격, source="블룸버그", freq="M",
                               period_start="20210701", release_date=None, known_from=None)
    assert blocked is True
    assert result.entries[0].undecidable == 1


def test_basis_는_우리_판정을_쓴다(거시규격):
    """추정치에 "release" 라고 적어 오면 그 자료가 실제보다 믿을 만해 보인다."""
    _, basis, _, _ = 거시(거시규격, source="ECOS", freq="M", period_start="20210701",
                        release_date=None, known_from=None, known_from_basis="release")
    assert basis == "estimate"


# ==================================================
# 4. 갈래
# ==================================================
@pytest.mark.parametrize("kind", ["ohlcv_stock", "ohlcv_index", "financial"])
def test_시세와_재무는_파생하지_않는다(kind, 거시규격):
    """시세는 supply 계층이 자르고, 재무의 시점 보정은 네트워크가 필요해 수집기의 일이다."""
    frame = pd.DataFrame({"bas_dd": ["20210104"]})
    result = derive(kind, frame, 거시규격)
    assert result.entries == []
    assert not result.blocked.any()
