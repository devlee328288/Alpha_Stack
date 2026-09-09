"""데이터 품질 원장 계약 — `supply/quality_ledger.py`.

## 이 파일이 지키는 것

1. **원장은 막지 않는다** — 붉어도 예외를 던지지 않는다. 막는 것은 게이트의 몫이다.
2. **축별 중복이 덮어쓰기를 잡는다** — 합계로는 못 잡는 사고를 여기서 재현한다.
3. **정지일은 거래량으로 판정한다** — 종가로 판정하면 한 건도 못 잡는다.
4. **이력은 append-only** — 같은 `run_id` 를 두 번 쓰지 않는다.
5. **카드에 플래그 사용법이 함께 나간다** — `is_extreme_return` 을 거꾸로 거르지 않도록.
6. **유효성은 체결이 있던 행만 잰다** — 정지행을 위반으로 세면 2.9%가 통째로 붉어진다.
7. **거래량 급변은 붉지 않고, 기준선을 함께 남긴다** — 배수만 베끼면 값이 2배 갈린다.

실제 DB·반출본을 쓰지 않는다. 인메모리 sqlite 와 손으로 만든 폴더로 잰다.
"""

from __future__ import annotations

import json
import sqlite3

import pandas as pd
import pytest

from supply import quality_ledger as ql

# ── 손으로 만드는 DB · 반출본 ────────────────────────────────────────────────

def _conn(*, 덮어썼나: bool = False) -> sqlite3.Connection:
    """`index_price` 를 두 시장으로 채운다.

    `덮어썼나=True` 면 **KOSDAQ 이 KOSPI 를 덮은 상태**를 만든다 — 같은
    `(bas_dd, index_name)` 인데 `index_class` 만 다른 행이 사라진 모양이다.
    """
    conn = sqlite3.connect(":memory:")
    conn.executescript(
        """
        CREATE TABLE daily_price (bas_dd TEXT, code TEXT, close REAL);
        CREATE TABLE index_price (bas_dd TEXT, index_name TEXT, index_class TEXT);
        CREATE TABLE trading_calendar (bas_dd TEXT, market TEXT);
        CREATE TABLE text_signal (text_sha TEXT, model_id TEXT);
        """
    )
    날들 = ["20240102", "20240103"]
    conn.executemany("INSERT INTO daily_price VALUES (?,?,?)",
                     [(d, c, 100.0) for d in 날들 for c in ("000010", "000020")])
    # 같은 이름의 업종지수가 두 시장에 있다 — 기본키에 시장이 없으면 서로를 덮는다
    행 = [(d, "전기전자", "KOSPI") for d in 날들]
    if not 덮어썼나:
        행 += [(d, "전기전자", "KOSDAQ") for d in 날들]
    conn.executemany("INSERT INTO index_price VALUES (?,?,?)", 행)
    conn.executemany("INSERT INTO trading_calendar VALUES (?,?)",
                     [(d, m) for d in 날들 for m in ("ALL", "KOSPI", "KOSDAQ")])
    conn.executemany("INSERT INTO text_signal VALUES (?,?)",
                     [("sha1", "m1"), ("sha2", "m1")])
    return conn


def _daily(*, 정지행: int = 1, adj0: bool = False) -> pd.DataFrame:
    """반출본 시세를 흉내 낸다. 정지일은 **거래량 0** 이고 종가는 직전값이 남는다."""
    행 = [
        {"bas_dd": "20240102", "code": "000010", "close": 100.0, "adj_close": 100.0,
         "change_rate": 0.0, "volume": 1000, "adj_open": 99.0, "adj_high": 101.0,
         "adj_low": 98.0, "adj_source": "fdr", "market": "KOSPI",
         "is_first_listing": False},
        {"bas_dd": "20240103", "code": "000010", "close": 110.0, "adj_close": 110.0,
         "change_rate": 10.0, "volume": 1200, "adj_open": 109.0, "adj_high": 111.0,
         "adj_low": 108.0, "adj_source": "fdr", "market": "KOSPI",
         "is_first_listing": False},
    ]
    if 정지행:
        행.append(
            {"bas_dd": "20240104", "code": "000010", "close": 110.0,   # 직전값이 남는다
             "adj_close": 0.0 if adj0 else 110.0, "change_rate": 0.0,
             "volume": 0,                                              # ← 정지일 표시
             "adj_open": 0.0 if adj0 else None, "adj_high": None, "adj_low": None,
             "adj_source": "fdr", "market": "KOSPI", "is_first_listing": False}
        )
    return pd.DataFrame(행)


@pytest.fixture
def 반출폴더(tmp_path):
    root = tmp_path / "2026-09-08"
    root.mkdir()
    (root / "MANIFEST.json").write_text(
        json.dumps({"holdout_start": "20240901", "files": [{"path": "a"}], "stats": {}}),
        encoding="utf-8",
    )
    (root / "PROFILE.json").write_text(
        json.dumps({"files": [{"path": "a", "칸수": 20}]}), encoding="utf-8"
    )
    return root


# ── 1. 원장은 막지 않는다 ────────────────────────────────────────────────────

def test_붉어도_예외를_던지지_않는다(반출폴더):
    """막는 것은 게이트의 몫이다. 원장은 붉다는 사실을 실어 돌려주기만 한다."""
    daily = _daily(adj0=True)                      # 정지일 adj_* 가 0 — 붉은 자리
    led = ql.build_quality_ledger(반출폴더, daily=daily)

    assert led["status"] == "red"
    assert any("zero_filled" in r for r in led["red"]), led["red"]
    # 예외가 아니라 값으로 돌아왔다는 것 자체가 이 시험의 요지다


def test_반출폴더가_없으면_그건_원장의_잘못이라_멈춘다(tmp_path):
    with pytest.raises(FileNotFoundError, match="반출 폴더"):
        ql.build_quality_ledger(tmp_path / "없는폴더")


# ── 2. 🔴 축별 중복 — 합계가 못 잡는 것을 잡는다 ─────────────────────────────

def test_시장이_빠진_축과_들어간_축이_다르면_덮어쓰기를_잡는다(반출폴더):
    """v3.8 사고 재현 — KOSPI·KOSDAQ 이 같은 이름 업종지수로 서로를 덮은 자리.

    합계(`COUNT(*)`)로는 못 잡는다. 실제 사고 때 합계는 **오히려 늘었다.**
    `(bas_dd, index_name)` 과 `(bas_dd, index_name, index_class)` 를 **따로 세서
    비교**해야 보인다.
    """
    with _conn(덮어썼나=False) as 정상:
        led = ql.build_quality_ledger(반출폴더, conn=정상, daily=_daily())
    m = led["axes"]["duplicate"]["index_price_two_axes"]
    assert m["value"]["without_market"] == 2       # 날짜 2 × 이름 1
    assert m["value"]["with_market"] == 4          # 날짜 2 × 이름 1 × 시장 2
    # 🔴 두 축이 다르면 붉다 — 한 시장이 다른 시장을 덮었다는 뜻이다
    assert m["status"] == "red"
    assert "duplicate.index_price_two_axes" in led["red"]


def test_한_시장만_남으면_두_축이_같아져_초록이다(반출폴더):
    """KOSDAQ 을 아예 안 들인 상태 — 덮어쓸 짝이 없으니 두 축이 같다."""
    with _conn(덮어썼나=True) as 한시장:
        led = ql.build_quality_ledger(반출폴더, conn=한시장, daily=_daily())
    m = led["axes"]["duplicate"]["index_price_two_axes"]
    assert m["value"]["without_market"] == m["value"]["with_market"] == 2
    assert m["status"] == "ok"


def test_행_수만_세면_덮어쓰기를_못_잡는다(반출폴더):
    """이 시험이 없으면 "합계로 충분하지 않나" 로 되돌아간다. 왜 안 되는지를 못박는다."""
    with _conn(덮어썼나=False) as 정상, _conn(덮어썼나=True) as 덮인:
        정상합계 = 정상.execute("SELECT COUNT(*) FROM index_price").fetchone()[0]
        덮인합계 = 덮인.execute("SELECT COUNT(*) FROM index_price").fetchone()[0]
        정상원장 = ql.build_quality_ledger(반출폴더, conn=정상, daily=_daily())
        덮인원장 = ql.build_quality_ledger(반출폴더, conn=덮인, daily=_daily())

    assert 정상합계 != 덮인합계                     # 합계는 달라지긴 한다
    # 그러나 합계만 보면 "줄었다" 인지 "덮였다" 인지 가릴 수 없다. 축 비교는 가린다.
    assert 정상원장["axes"]["duplicate"]["index_price_two_axes"]["status"] == "red"
    assert 덮인원장["axes"]["duplicate"]["index_price_two_axes"]["status"] == "ok"


# ── 3. 정지일은 거래량으로 판정한다 ──────────────────────────────────────────

def test_정지일은_거래량_0_으로_센다_종가로는_한_건도_못_잡는다(반출폴더):
    """KRX 는 정지일에도 종가 칸에 직전 가격을 실어 보낸다.

    처음에 `close == 0` 을 함께 걸었다가 실제 반출본에서 정지행이 0 으로 나왔다
    (7,888,945행 중 `close == 0` 은 0건, `volume == 0` 은 231,684건).
    """
    daily = _daily(정지행=1)
    assert (daily["close"] == 0).sum() == 0        # 종가로는 아무것도 못 잡는다
    led = ql.build_quality_ledger(반출폴더, daily=daily)
    assert led["axes"]["missing"]["halt_rows"]["value"] == 1


def test_정지일_수정가가_0_이면_붉다(반출폴더):
    """0 으로 채우면 일간수익률이 −100% 가 되어 없던 사건이 생긴다."""
    깨끗 = ql.build_quality_ledger(반출폴더, daily=_daily(adj0=False))
    더러움 = ql.build_quality_ledger(반출폴더, daily=_daily(adj0=True))
    assert 깨끗["axes"]["missing"]["zero_filled_halt_rows"]["status"] == "ok"
    assert 더러움["axes"]["missing"]["zero_filled_halt_rows"]["status"] == "red"


# ── 4. 이력은 append-only ────────────────────────────────────────────────────

def test_같은_run_id_는_두_번_쓰지_않는다(반출폴더, tmp_path):
    이력 = tmp_path / "quality_ledger.jsonl"
    led = ql.build_quality_ledger(반출폴더, daily=_daily())

    assert ql.append_history(led, path=이력) is not None
    assert ql.append_history(led, path=이력) is None       # 두 번째는 안 쓴다
    assert len(이력.read_text(encoding="utf-8").strip().splitlines()) == 1


def test_이력의_마지막_줄이_다음_반출본의_기준이_된다(반출폴더, tmp_path):
    이력 = tmp_path / "quality_ledger.jsonl"
    첫판 = ql.build_quality_ledger(반출폴더, daily=_daily(정지행=0), run_id="r1")
    ql.append_history(첫판, path=이력)

    지난 = ql.read_last_history(path=이력)
    assert 지난["run_id"] == "r1"

    둘째 = ql.build_quality_ledger(반출폴더, daily=_daily(정지행=1), run_id="r2",
                                 previous=지난)
    바뀐 = 둘째["previous"]["changed"]
    assert "missing.halt_rows" in 바뀐                     # 0 → 1 로 달라졌다
    assert 바뀐["missing.halt_rows"] == {"before": 0, "after": 1}


# ── 5. 카드 ──────────────────────────────────────────────────────────────────

def test_카드에_플래그_사용법이_함께_나간다(반출폴더):
    """신장환 님이 #168 에서 짚은 것 — `is_extreme_return` 을 거꾸로 거를 위험.

    이름만 보면 "극단이니 빼라" 로 읽히지만 반대다. 카드에 그 한 줄이 없으면
    받는 사람이 진짜 사건을 지운다.
    """
    카드 = ql.render_card_section(ql.build_quality_ledger(반출폴더, daily=_daily()))

    assert "is_adj_suspect" in 카드 and "is_extreme_return" in 카드
    assert "거르지" in 카드 or "거르는 용도가" in 카드     # 거르지 말라고 분명히 적는다
    assert "df[~df[\"is_adj_suspect\"]]" in 카드          # 그대로 복사할 한 줄


def test_카드는_붉은_것을_붉게_적는다(반출폴더):
    """숨기지 않는 것이 이 원장의 목적이다."""
    카드 = ql.render_card_section(
        ql.build_quality_ledger(반출폴더, daily=_daily(adj0=True))
    )
    assert "🔴" in 카드
    assert "zero_filled_halt_rows" in 카드
    assert "막지 않고 적습니다" in 카드


def test_원장을_MANIFEST_옆에_쓴다(반출폴더):
    led = ql.build_quality_ledger(반출폴더, daily=_daily())
    경로 = ql.write_quality_ledger(반출폴더, led)
    assert 경로.name == ql.LEDGER_NAME
    assert 경로.parent == 반출폴더                          # 반출본과 같이 움직인다
    다시 = json.loads(경로.read_text(encoding="utf-8"))
    assert 다시["status"] == led["status"]


def test_일곱_축이_모두_있다(반출폴더):
    led = ql.build_quality_ledger(반출폴더, daily=_daily())
    assert tuple(led["axes"]) == ql.AXES
    assert led["run_id"] and led["generated_at"]


# ── 6. 유효성 — 정지행을 위반으로 세지 않는다 ────────────────────────────────
#
# KRX 는 거래정지 중에도 행을 준다. `open=high=low=0` 이고 종가만 직전 값이 남는다.
# 그 행을 "OHLC 순서 위반" 으로 세면 개발구간 231,808행(2.938%)이 통째로 붉어진다.
# 규칙 위반이 아니라 **KRX 의 정지 표기**다.

def _가격행(**덮어쓰기):
    행 = {"bas_dd": "20240102", "code": "000010", "market": "KOSPI",
          "open": 100.0, "high": 110.0, "low": 95.0, "close": 105.0,
          "volume": 1000, "value": 105_000.0,
          "market_cap": 1000.0, "listed_shares": 10.0,
          "adj_close": 105.0, "change_rate": 0.0, "adj_source": "fdr",
          "is_first_listing": False}
    행.update(덮어쓰기)
    return 행


def test_정지행은_유효성_위반이_아니다():
    """`open=high=low=0` 이고 거래량 0 인 행. 재면 안 된다."""
    daily = pd.DataFrame([
        _가격행(),
        _가격행(bas_dd="20240103", open=0.0, high=0.0, low=0.0, close=105.0,
                volume=0, value=0.0),
    ])
    축 = ql._axis_validity(daily)
    assert 축["ohlc_order_rows"]["value"] == 0
    assert 축["ohlc_order_rows"]["status"] == "ok"


def test_시간외_단일가만_체결된_행은_따로_센다():
    """개발구간 124행이 이 모양이다 — OHL 이 0 인데 종가·거래량이 있다.

    처음에는 "고가·저가는 살아 있는 행" 이라고 적었는데 실제로는 고가·저가도 0 이다.
    계산 오류가 아니라 정규장 체결이 없었던 날이므로 붉게 보지 않는다.
    """
    daily = pd.DataFrame([
        _가격행(),
        _가격행(bas_dd="20240103", open=0.0, high=0.0, low=0.0, close=105.0,
                volume=1),
    ])
    축 = ql._axis_validity(daily)
    assert 축["nonpositive_open_rows"]["value"] == 1
    assert 축["nonpositive_open_rows"]["status"] == "ok"
    # 그 행이 순서 위반으로 이중 계상되면 안 된다
    assert 축["ohlc_order_rows"]["value"] == 0


def test_고가가_종가보다_낮으면_붉다():
    """어떤 시장 사건으로도 설명되지 않는다. 계산이 틀린 것이다."""
    daily = pd.DataFrame([_가격행(high=100.0, close=105.0)])
    축 = ql._axis_validity(daily)
    assert 축["ohlc_order_rows"]["value"] == 1
    assert 축["ohlc_order_rows"]["status"] == "red"


def test_시가총액_항등식이_깨지면_붉다():
    """`market_cap == close x listed_shares`. 1.8경이라 상대오차로 잰다."""
    daily = pd.DataFrame([_가격행(market_cap=9999.0)])   # 105 x 10 = 1050 이어야 한다
    축 = ql._axis_validity(daily)
    assert 축["market_cap_identity_rows"]["value"] == 1
    assert 축["market_cap_identity_rows"]["status"] == "red"


# ── 7. 거래량 급변 — 붉지 않고, 기준선을 남긴다 ──────────────────────────────

def _거래량시계열(마지막: int) -> pd.DataFrame:
    """5일 평균이 100 이 되도록 앞을 채우고, 마지막 날 거래량만 바꾼다."""
    행 = [_가격행(bas_dd=f"2024010{i}", volume=100) for i in range(1, 6)]
    행.append(_가격행(bas_dd="20240106", volume=마지막))
    return pd.DataFrame(행)


def test_거래량_급변은_붉지_않다():
    """상위가 전부 설명되는 사건이다 — 액면분할·보호예수 해제·인수전.

    붉게 두면 배포마다 게이트가 막히고, 붉은불이 흔해지면 아무도 보지 않는다.
    """
    축 = ql._axis_volume(_거래량시계열(100_000))
    assert 축["surge_rows"]["value"] == 1
    assert 축["surge_rows"]["status"] == "ok"


def test_거래량_급변의_기준선이_노트에_적힌다():
    """배수(3배)는 세 표준이 같지만 **무엇에 대한 3배인지**가 다르다.

    우리 데이터에서 KRX 5일 평균 5.426% · qlib 전일 7.977% · 20일 중앙값 10.329% 로
    2배 가까이 갈린다. 기준선이 없으면 다음 사람이 배수만 보고 베낀다.
    """
    노트 = ql._axis_volume(_거래량시계열(100_000))["surge_rows"]["note"]
    assert "KRX" in 노트
    assert f"{ql.SURGE_WINDOW}일 평균" in 노트
    assert f"{ql.SURGE_MULTIPLE:g}배" in 노트


def test_그날_거래량은_자기_기준선에_들어가지_않는다():
    """`shift(1)` 이 빠지면 급변이 스스로를 희석해 큰 값일수록 덜 잡힌다."""
    작음 = ql._axis_volume(_거래량시계열(301))["surge_rows"]["value"]
    assert 작음 == 1, "5일 평균 100 의 3배를 갓 넘겼는데 안 잡혔다"


def test_정지일_0_은_기준선에_섞이지_않는다():
    """0 을 평균에 넣으면 기준선이 내려가 재개일이 전부 급변으로 잡힌다."""
    행 = [_가격행(bas_dd=f"2024010{i}", volume=100) for i in range(1, 4)]
    행 += [_가격행(bas_dd="20240104", open=0.0, high=0.0, low=0.0, volume=0),
           _가격행(bas_dd="20240105", open=0.0, high=0.0, low=0.0, volume=0)]
    행.append(_가격행(bas_dd="20240106", volume=250))     # 100 의 2.5배 — 급변이 아니다
    축 = ql._axis_volume(pd.DataFrame(행))
    assert 축["surge_rows"]["value"] == 0


# ── 8. 원장이 읽는 칸이 모든 축을 덮는다 ─────────────────────────────────────
#
# 🔴 `_read_export_daily` 는 376MB 를 다 읽지 않으려고 칸을 좁힌다. 그런데 축이
#    요구하는 칸이 그 목록에 없으면 **그 축은 조용히 `skip` 된다** — 예외도 경고도
#    없이 값이 `None` 으로 나갈 뿐이다.
#
#    실제로 그렇게 됐다. `validity` 축을 넣고 첫 반출을 돌렸더니 네 지표가 전부
#    `None` 이었다. 원장 파일에는 축 이름이 멀쩡히 있고 판정도 실려 있어서, 이력
#    한 줄을 열어 보기 전까지 아무도 몰랐다. 축은 늘었는데 재는 것은 없었다.

def test_원장이_읽는_칸이_모든_축을_덮는다(반출폴더):
    """읽는 칸만 준 표로 원장을 만들면 `skip` 이 하나도 없어야 한다.

    새 축을 넣으면서 `EXPORT_DAILY_COLUMNS` 를 안 늘리면 여기서 걸린다.
    """
    행 = []
    for i in range(1, 7):
        r = {c: 0 for c in ql.EXPORT_DAILY_COLUMNS}
        r.update({"bas_dd": f"2024010{i}", "code": "000010", "market": "KOSPI",
                  "open": 100.0, "high": 110.0, "low": 95.0, "close": 105.0,
                  "volume": 1000, "value": 105_000.0,
                  "market_cap": 1050.0, "listed_shares": 10.0,
                  "change_rate": 0.0, "adj_open": 100.0, "adj_high": 110.0,
                  "adj_low": 95.0, "adj_close": 105.0, "adj_source": "fdr"})
        행.append(r)
    daily = pd.DataFrame(행)[list(ql.EXPORT_DAILY_COLUMNS)]

    led = ql.build_quality_ledger(반출폴더, daily=daily)

    건너뛴 = [f"{축}.{이름}"
             for 축, 표 in led["axes"].items()
             for 이름, m in 표.items()
             if m.get("status") == "skip" and 축 not in ("calendar", "duplicate")]
    assert not 건너뛴, (
        f"이 축들이 조용히 건너뛰어졌다: {건너뛴}. "
        "`EXPORT_DAILY_COLUMNS` 에 그 축이 쓰는 칸을 더해라."
    )


def test_새_축을_넣으면_이력에도_따라_들어간다(반출폴더, tmp_path):
    """이력 한 줄에 일곱 축이 다 있어야 나중에 추세를 볼 수 있다."""
    led = ql.build_quality_ledger(반출폴더, daily=_daily())
    경로 = ql.append_history(led, path=tmp_path / "h.jsonl")
    줄 = json.loads(경로.read_text(encoding="utf-8").splitlines()[-1])
    assert tuple(줄["summary"]) == ql.AXES


# ── 카드의 표본 안내 숫자는 손으로 적지 않는다 (이슈 #195) ───────────────────
#
# `build_dataset_card` 의 docstring 이 *"숫자는 MANIFEST.json 과 PROFILE.json 에서만
# 가져온다 — 카드에 손으로 적지 않는다"* 라고 못박아 두었는데 이 절만 예외였다.
# 2026-09-09 에 코드 재사용 판정으로 `is_first_listing` 이 1,501 → 1,502 ·
# 셋 중 하나가 251,282 → 251,283 으로 늘었는데 **카드는 옛 숫자를 계속 적었다.**

def test_표본_안내_숫자는_넘긴_값으로_채운다():
    글 = ql.sample_guide(
        {"is_liquidation": 17_973, "is_halted": 231_808,
         "is_first_listing": 1_502, "any": 251_283},
        total_rows=7_888_945,
    )
    assert "1,502" in 글 and "251,283" in 글
    assert "17,973" in 글 and "231,808" in 글
    assert "1,501" not in 글 and "251,282" not in 글, "옛 숫자가 남아 있다"
    assert "(0.019%)" in 글, "비율도 함께 적는다"


def test_표본_안내_템플릿에는_행수가_박혀_있지_않다():
    """행 수를 템플릿에 다시 박으면 이 시험이 잡는다.

    쉼표가 든 수(`251,282` · `1,501`)만 본다 — `KOSPI200` 처럼 이름에 든 숫자는
    행 수가 아니다.
    """
    import re
    남은 = re.findall(r"\d{1,3}(?:,\d{3})+", ql.SAMPLE_GUIDE_TEMPLATE)
    assert not 남은, f"템플릿에 박힌 행 수: {남은}"
    assert "{차이}" in ql.SAMPLE_GUIDE_TEMPLATE and "{표}" in ql.SAMPLE_GUIDE_TEMPLATE


def test_숫자를_못_읽으면_옛_숫자를_쓰지_않고_비운다():
    """조용히 옛 숫자를 쓰는 것보다 "여기 숫자가 없다" 가 낫다."""
    글 = ql.sample_guide(None, None)
    assert "MANIFEST 에서 못 읽었다" in 글
    assert "1,501" not in 글 and "1,502" not in 글


def test_원장_카드_절이_표본_숫자를_받아_넘긴다():
    led = {"status": "ok", "run_id": "t", "generated_at": "now", "axes": {}, "red": []}
    글 = ql.render_card_section(
        led, corporate_action_flags={"is_first_listing": 1_502, "any": 251_283},
        total_rows=7_888_945)
    assert "1,502" in 글 and "251,283" in 글
