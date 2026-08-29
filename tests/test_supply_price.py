"""종목 공급 경로 테스트 — `price_series` 는 순수하고 `training_frame` 만 미래를 본다.

**무엇을 지키려는 테스트인가.** 이 PR 이 문을 둘로 갈랐다.

    price_series(as_of=...)   그 시점에 알 수 있었던 것만. 플래그 없음
    training_frame(...)       전 구간을 보고 정리매매·신규상장을 덜어낸다

가른 이유는 정리매매 판정이 *"이 뒤로 체결이 끊긴다"* 를 보고 정해지기 때문이다 —
그 시점에는 알 수 없는 사실이다. 손잡이 하나로 켜고 끄게 두면 언젠가 켜진 채로 예측
경로에 들어가고, **그때도 예외는 나지 않는다.** 그래서 여기서 잠그는 것은 기능이
아니라 **어느 함수가 무엇을 볼 수 있는가** 다.

    수용 기준
    - `price_series` 는 정리매매 행을 **그대로 준다** (판정하지 않는다)
    - `training_frame` 은 같은 행을 **덜어낸다** 그리고 몇 개 뺐는지 말한다
    - `holdout_start` 를 빠뜨리면 **터진다** (기본값이 없다)
    - 빈 결과에도 칸이 남는다
"""

from __future__ import annotations

from datetime import date, timedelta

import pytest

import supply
from ingest.store import krx_store


def _평일들(n: int) -> list:
    out, day = [], date(2024, 1, 2)
    while len(out) < n:
        if day.weekday() < 5:
            out.append(day.strftime("%Y%m%d"))
        day += timedelta(days=1)
    return out


# ⚠️ `SUSPENSION_GAP_DAYS`(20)보다 길게 잡는다. 짧으면 어떤 공백도 문턱에 못 닿아서
#    "정리매매가 안 잡힌다" 는 결과가 나오는데, 그건 코드가 아니라 시험대 탓이다.
DAYS = _평일들(30)

살아있음 = "000001"      # 마지막 거래일까지 체결된다
소멸 = "000002"          # 20일째까지만 있다 → 마지막 10체결일이 정리매매
신규 = "000003"          # 6일째에 처음 나타난다 → 그 날이 신규상장 첫 거래일


@pytest.fixture()
def 종목저장소(tmp_path, monkeypatch):
    """세 종목이 든 임시 저장소. 각각 다른 플래그가 붙도록 만들었다."""
    db = tmp_path / "price.db"
    monkeypatch.setattr(krx_store, "DB_PATH", db)
    krx_store.init_db()

    행 = []
    for i, day in enumerate(DAYS):
        행.append((day, 살아있음))
        if i < 20:
            행.append((day, 소멸))
        if i >= 5:
            행.append((day, 신규))

    with krx_store.connect() as conn:
        conn.executemany(
            "INSERT INTO daily_price (bas_dd, code, name, market, open, high, low, "
            "close, change, change_rate, volume, value, market_cap, listed_shares) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            [(d, c, f"종목{c}", "KOSPI", 1000, 1010, 990, 1000, 0, 0.0,
              5000, 5_000_000, 100_000, 100) for d, c in 행],
        )
    return db


# ── price_series 는 판정하지 않는다 ─────────────────────────────────────────

def test_as_of_없이_부르면_터진다(종목저장소):
    with pytest.raises(TypeError):
        supply.price_series(살아있음)              # type: ignore[call-arg]


def test_아직_몰랐던_거래일은_나오지_않는다(종목저장소):
    df = supply.price_series(살아있음, as_of="2024-01-10")

    assert df["bas_dd"].max() == "20240109", "당일 자료가 새어 나왔다"


def test_정리매매_행을_그대로_준다(종목저장소):
    """🔴 `price_series` 가 정리매매를 덜어내면, 문이 지키려던 경계를 문이 넘는다.

    "이 뒤로 체결이 끊긴다" 는 그 시점에 알 수 없는 사실이다. 그걸 보고 행을 빼면
    그 자체가 미래 참조다.
    """
    df = supply.price_series(소멸, as_of="2099-01-01")

    assert len(df) == 20, "정리매매 구간이 조용히 빠졌다"
    assert "liquidation" not in df.columns, "예측 경로에 플래그가 새어 나왔다"


def test_신규상장_첫_행도_그대로_준다(종목저장소):
    df = supply.price_series(신규, as_of="2099-01-01")

    assert df["bas_dd"].iloc[0] == DAYS[5]


def test_빈_결과에도_칸이_남는다(종목저장소):
    df = supply.price_series(살아있음, as_of="2000-01-01")

    assert len(df) == 0
    assert list(df.columns) == list(supply.PRICE_COLUMNS)


def test_칸_목록이_저장소_반환을_덮는다(종목저장소):
    df = supply.price_series(살아있음, as_of="2099-01-01")

    assert list(df.columns) == list(supply.PRICE_COLUMNS), (
        "저장소가 칸을 바꿨다. supply/market.py 의 PRICE_COLUMNS 도 맞춰야 한다.\n"
        f"  실제: {list(df.columns)}"
    )


def test_ISO_로_준_end_가_결과를_0행으로_만들지_않는다(종목저장소):
    """지수와 같은 함정이다 — `min('2024-01-05','20240110')` 은 하이픈 쪽이 이긴다."""
    여덟자리 = supply.price_series(살아있음, as_of="2099-01-01", end="20240105")
    ISO = supply.price_series(살아있음, as_of="2099-01-01", end="2024-01-05")

    assert len(ISO) > 0, "ISO 로 준 end 가 0행을 만들었다"
    assert list(ISO["bas_dd"]) == list(여덟자리["bas_dd"])


def test_기본은_전_구간이다(종목저장소):
    """`days` 기본이 250 이면 라벨과 정리매매 판정이 잘린 자리에서 틀린다."""
    df = supply.price_series(살아있음, as_of="2099-01-01")

    assert len(df) == len(DAYS)


# ── training_frame 만 미래를 본다 ───────────────────────────────────────────

def test_holdout_start_를_빠뜨리면_터진다(종목저장소):
    """`as_of` 와 같은 수법이다. 기본이 '자르지 않음' 이면 빠뜨린 한 번이 봉인을 연다."""
    with pytest.raises(TypeError):
        supply.training_frame(살아있음)             # type: ignore[call-arg]


def test_정리매매_구간을_덜어낸다(종목저장소):
    """소멸 종목의 마지막 10체결일이 빠져야 한다. 그 구간은 가격제한폭이 없다."""
    df = supply.training_frame(소멸, holdout_start=None)

    assert len(df) == 10
    assert df.attrs["dropped"]["liquidation"] == 10
    assert df.attrs["input_rows"] == 20


def test_신규상장_첫_행을_덜어낸다(종목저장소):
    """그 행의 등락률은 전일종가가 아니라 공모가 기준이라 수익률이 아니다."""
    df = supply.training_frame(신규, holdout_start=None)

    assert df["bas_dd"].iloc[0] == DAYS[6]
    assert df.attrs["dropped"]["first_listing"] == 1


def test_상장중_종목은_아무것도_덜어내지_않는다(종목저장소):
    """멀쩡한 종목의 최근 10일을 자르면 그건 정리매매가 아니라 자료를 버리는 것이다."""
    df = supply.training_frame(살아있음, holdout_start=None)

    assert len(df) == len(DAYS)
    assert df.attrs["dropped"] == {}


def test_덜어낸_양을_말해_준다(종목저장소):
    """조용히 빠지면 표본 수가 왜 줄었는지 아무도 모른다."""
    df = supply.training_frame(소멸, holdout_start=None)

    뺀것 = df.attrs["dropped"]
    assert sum(뺀것.values()) + len(df) == df.attrs["input_rows"]


def test_홀드아웃을_열지_않는다(종목저장소):
    """봉인 구간이 학습 자료에 섞이면 그 순간 사전등록이 무의미해진다."""
    df = supply.training_frame(살아있음, holdout_start=DAYS[10])

    assert df["bas_dd"].max() < DAYS[10]
    assert df.attrs["dropped"]["holdout"] == len(DAYS) - 10


def test_손잡이를_끄면_남는다(종목저장소):
    """비교해 보고 싶을 때가 있다. 다만 **기본값은 제외**다."""
    남김 = supply.training_frame(소멸, holdout_start=None, drop_liquidation=False)

    assert len(남김) == 20


def test_없는_종목은_빈_표를_준다(종목저장소):
    df = supply.training_frame("999999", holdout_start=None)

    assert len(df) == 0
    assert list(df.columns) == list(supply.PRICE_COLUMNS)
    assert df.attrs["input_rows"] == 0


# ── 여러 종목 ───────────────────────────────────────────────────────────────

def test_시장_사실을_한_번만_만들어도_답이_같다(종목저장소):
    """🔴 준비 비용을 아끼려다 답이 달라지면 아낀 의미가 없다.

    종목마다 달력을 다시 만들면 종목당 2.71초라 3,677종목이면 166분이다.
    한 번 만들어 나눠 쓰면 16.3분인데(실측 10배), **결과가 같아야** 쓸 수 있다.
    """
    따로 = {c: supply.training_frame(c, holdout_start=None)
            for c in (살아있음, 소멸, 신규)}
    같이 = dict(supply.training_frames((살아있음, 소멸, 신규), holdout_start=None))

    for code in (살아있음, 소멸, 신규):
        assert list(따로[code]["bas_dd"]) == list(같이[code]["bas_dd"])
        assert 따로[code].attrs["dropped"] == 같이[code].attrs["dropped"]


def test_시장_사실을_밖에서_넘길_수_있다(종목저장소):
    """모듈 전역에 몰래 캐시하지 않는다 — 수집이 새 행을 넣어도 낡은 달력을 쓰게 된다."""
    ctx = supply.market_context()

    assert ctx.collect_start == DAYS[0]
    assert ctx.market_last_index == len(DAYS) - 1
    assert 살아있음 in ctx.listed_codes
    assert 소멸 not in ctx.listed_codes, "20일째에 사라진 종목이 상장중으로 잡혔다"

    df = supply.training_frame(소멸, holdout_start=None, context=ctx)
    assert df.attrs["dropped"]["liquidation"] == 10
