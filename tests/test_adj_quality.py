"""수정주가 품질 플래그 시험 — 판별자는 크기가 아니라 KRX 등락률과의 어긋남이다 (이슈 #132).

실제 모양 둘을 그대로 쓴다.
  경남에너지 2016-05-11  +153.66% 인데 KRX 등락률도 +153.66%  → 진짜 사건. 남긴다
  신세계     2011-06-10  우리 −60.6% 인데 KRX 등락률은 +14.95% → 조정 오류 의심. 뺀다
"""

import numpy as np
import pandas as pd
import pytest

from supply.adj_quality import (
    EXTREME_RETURN_PCT,
    FLAG_COLUMNS,
    SUSPECT_GAP_TOLERANCE,
    attach_adjustment_quality,
    flag_adjustment_quality,
)


def _frame(rows, *, first_listing=None):
    """(bas_dd, code, close, adj_close, change_rate) 튜플 목록 → 원천 모양의 프레임.

    `is_first_listing` 은 필수 칸이라(코드 재사용 자리를 알아야 한다 · 이슈 #195)
    기본값 False 로 채운다. 그 칸을 켜서 시험할 때는 `first_listing=[...]` 로 준다.
    """
    frame = pd.DataFrame(
        rows, columns=["bas_dd", "code", "close", "adj_close", "change_rate"]
    )
    frame["is_first_listing"] = (
        [False] * len(frame) if first_listing is None else list(first_listing)
    )
    return frame


# 경남에너지 — 10,250 → 26,000 (+153.66%), KRX 도 +153.66
경남에너지 = [
    ("20160510", "008020", 10250, 10250.0, 0.99),
    ("20160511", "008020", 26000, 26000.0, 153.66),
    ("20160512", "008020", 22800, 22800.0, -12.31),
]

# 신세계 — 인적분할 재상장일. 우리 chain 이 과거를 3.83배 부풀려 −60.6% 가 됐다
신세계 = [
    ("20110609", "004170", 270000, 1034483.0, 0.0),
    ("20110610", "004170", 407500, 407500.0, 14.95),
    ("20110613", "004170", 366500, 366500.0, -10.06),
]


def test_KRX와_일치하는_극단수익률은_진짜_사건이라_남긴다():
    out = flag_adjustment_quality(_frame(경남에너지))
    row = out.iloc[1]
    assert row["adj_return_1d"] == pytest.approx(153.66, abs=0.01)
    assert row["adj_change_rate_gap"] < 0.01
    assert bool(row["is_adj_suspect"]) is False
    assert bool(row["is_extreme_return"]) is True


def test_KRX와_어긋나면_조정_오류_의심이고_극단으로_치지_않는다():
    out = flag_adjustment_quality(_frame(신세계))
    row = out.iloc[1]
    assert row["adj_return_1d"] == pytest.approx(-60.61, abs=0.01)
    assert row["adj_change_rate_gap"] == pytest.approx(75.56, abs=0.01)
    assert bool(row["is_adj_suspect"]) is True
    # 크기로는 극단이지만 원인이 오류이므로 "진짜 극단" 이 아니다
    assert bool(row["is_extreme_return"]) is False


def test_반올림_잡음은_의심이_아니다():
    """저가주는 FDR 원 단위 반올림으로 0.2~0.5%p 가 흔하다. 1%p 안이면 정상이다."""
    rows = [
        ("20200102", "000001", 500, 500.0, 0.0),
        ("20200103", "000001", 503, 503.0, 0.6),    # 우리 +0.60 · KRX +0.6 → gap 0
        ("20200106", "000001", 505, 504.0, 0.4),    # 우리 +0.20 · KRX +0.4 → gap 0.2
    ]
    out = flag_adjustment_quality(_frame(rows))
    assert out["adj_change_rate_gap"].iloc[2] == pytest.approx(0.2, abs=0.01)
    assert not out["is_adj_suspect"].any()
    assert not out["is_extreme_return"].any()


def test_허용폭_경계에서_갈린다():
    rows = [
        ("20200102", "000001", 1000, 1000.0, 0.0),
        ("20200103", "000001", 1020, 1020.0, 0.9),    # gap 1.1 → 의심
        ("20200106", "000001", 1030, 1030.0, 0.5),    # 우리 +0.98 → gap 0.48 → 정상
    ]
    out = flag_adjustment_quality(_frame(rows))
    assert out["is_adj_suspect"].tolist() == [False, True, False]
    wider = flag_adjustment_quality(_frame(rows), gap_tolerance=2.0)
    assert not wider["is_adj_suspect"].any()


def test_비교할_수_없는_행은_NaN_이고_플래그는_False_다():
    """첫 행 · 전일 adj 없음 · 종가 0(정지) · 등락률 없음 — 모르는 것을 오류로 치지 않는다."""
    rows = [
        ("20200102", "000001", 1000, 1000.0, 0.0),        # 첫 행
        ("20200103", "000001", 1500, np.nan, 50.0),       # adj 없음
        ("20200106", "000001", 1500, 1500.0, 0.0),        # 전일 adj 없음
        ("20200107", "000001", 0, 0.0, 0.0),              # 종가 0
        ("20200108", "000001", 1500, 1500.0, np.nan),     # 등락률 없음 (전일 adj 0 이기도)
    ]
    out = flag_adjustment_quality(_frame(rows))
    assert out["adj_return_1d"].isna().all()
    assert out["adj_change_rate_gap"].isna().all()
    assert not out["is_adj_suspect"].any()
    assert not out["is_extreme_return"].any()
    assert out.attrs["adjustment_quality"]["comparable_rows"] == 0


def test_입력_순서와_인덱스를_보존한다():
    frame = _frame(경남에너지 + 신세계)
    shuffled = frame.sample(frac=1.0, random_state=7)
    out = flag_adjustment_quality(shuffled)
    assert out.index.equals(shuffled.index)
    assert list(out.columns) == list(FLAG_COLUMNS)
    # 같은 행은 순서와 무관하게 같은 값을 받는다
    ordered = flag_adjustment_quality(frame)
    pd.testing.assert_frame_equal(out.sort_index(), ordered.sort_index())


def test_수익률은_후보가_아니라_종목_전체_시계열에서_계산한다():
    """후보만 잘라 넘기면 전날이 빠져 며칠짜리 수익률이 된다. 원천 전체를 넘겨야 한다."""
    rows = [
        ("20200102", "000001", 1000, 1000.0, 0.0),
        ("20200103", "000001", 1100, 1100.0, 10.0),
        ("20200106", "000001", 1210, 1210.0, 10.0),
    ]
    whole = flag_adjustment_quality(_frame(rows))
    assert whole["adj_return_1d"].iloc[2] == pytest.approx(10.0)
    cut = flag_adjustment_quality(_frame([rows[0], rows[2]]))     # 가운데 날을 빼면
    assert cut["adj_return_1d"].iloc[1] == pytest.approx(21.0)    # 이틀치가 되고
    assert bool(cut["is_adj_suspect"].iloc[1]) is True             # 없는 오류가 생긴다


def test_뒤_행을_잘라도_앞_행의_플래그는_같다():
    """시점 규칙 — T−1·T 만 쓴다. 나중 행이 있든 없든 T 의 판정은 같다 (홀드아웃 동일 적용)."""
    frame = _frame(경남에너지 + 신세계)
    full = flag_adjustment_quality(frame)
    truncated = flag_adjustment_quality(frame.iloc[:-1])     # 마지막 행을 뗀다
    pd.testing.assert_frame_equal(full.iloc[:-1], truncated)


def test_같은_날짜_종목이_두_번이면_거부한다():
    with pytest.raises(ValueError, match="두 번 이상"):
        flag_adjustment_quality(_frame(경남에너지 + [경남에너지[0]]))


def test_필요한_열이_없으면_거부한다():
    frame = _frame(경남에너지).drop(columns=["change_rate"])
    with pytest.raises(ValueError, match="change_rate"):
        flag_adjustment_quality(frame)


def test_허용폭과_극단_기준은_양수여야_한다():
    frame = _frame(경남에너지)
    with pytest.raises(ValueError):
        flag_adjustment_quality(frame, gap_tolerance=0.0)
    with pytest.raises(ValueError):
        flag_adjustment_quality(frame, extreme_pct=-1.0)


def test_날짜는_문자열_정수_datetime_을_모두_받는다():
    base = _frame(경남에너지)
    as_int = base.assign(bas_dd=base["bas_dd"].astype(int))
    as_dt = base.assign(bas_dd=pd.to_datetime(base["bas_dd"]))
    expected = flag_adjustment_quality(base)
    pd.testing.assert_frame_equal(flag_adjustment_quality(as_int), expected)
    pd.testing.assert_frame_equal(flag_adjustment_quality(as_dt), expected)


def test_후보_프레임에_붙이면_후보_행만_받고_원천에_없는_키는_False_다():
    daily = _frame(경남에너지 + 신세계)
    candidates = pd.DataFrame(
        {
            "bas_dd": ["20160511", "20110610", "20240102"],
            "code": ["008020", "004170", "005930"],       # 마지막은 원천에 없다
            "candidate_rank": [1, 2, 3],
        },
        index=[10, 20, 30],
    )
    candidates.attrs["candidate_rule"] = {"sector_count": 10}

    out = attach_adjustment_quality(candidates, daily)

    assert out.index.tolist() == [10, 20, 30]
    assert out["candidate_rank"].tolist() == [1, 2, 3]
    assert out["is_extreme_return"].tolist() == [True, False, False]
    assert out["is_adj_suspect"].tolist() == [False, True, False]
    assert np.isnan(out["adj_return_1d"].iloc[2])
    assert out.attrs["candidate_rule"] == {"sector_count": 10}
    summary = out.attrs["adjustment_quality"]
    assert summary["candidate_rows"] == 3
    assert summary["candidate_suspect_rows"] == 1
    assert summary["candidate_extreme_rows"] == 1
    assert summary["candidate_unmatched_rows"] == 1
    assert summary["gap_tolerance"] == SUSPECT_GAP_TOLERANCE
    assert summary["extreme_pct"] == EXTREME_RETURN_PCT


def test_후보에_같은_이름의_열이_있으면_거부한다():
    daily = _frame(경남에너지)
    candidates = pd.DataFrame(
        {"bas_dd": ["20160511"], "code": ["008020"], "is_adj_suspect": [False]}
    )
    with pytest.raises(ValueError, match="is_adj_suspect"):
        attach_adjustment_quality(candidates, daily)


def test_후보에_같은_날짜_종목이_두_번이면_거부한다():
    daily = _frame(경남에너지)
    candidates = pd.DataFrame(
        {"bas_dd": ["20160511", "20160511"], "code": ["008020", "008020"]}
    )
    with pytest.raises(ValueError, match="후보에 같은 날짜"):
        attach_adjustment_quality(candidates, daily)


def test_학습에서는_의심_행만_빼고_진짜_사건은_남긴다():
    """모델 파트가 쓸 한 줄 — `cand[~cand.is_adj_suspect]` — 이 경남에너지를 남기는지."""
    daily = _frame(경남에너지 + 신세계)
    candidates = daily[["bas_dd", "code"]].copy()
    out = attach_adjustment_quality(candidates, daily)
    kept = out.loc[~out["is_adj_suspect"], ["bas_dd", "code"]]
    assert ("20160511", "008020") in set(map(tuple, kept.to_numpy()))
    assert ("20110610", "004170") not in set(map(tuple, kept.to_numpy()))


# ── 코드 재사용 — "전일" 은 shift(1) 이 아니다 (이슈 #195) ──────────────────

# 036220 — 인포피아(~2016-05-04) 코드를 오상헬스케어(2024-03-13~)가 다시 받았다.
# 실제 값이다: shift(1) 로 재면 +738.57% 인데 KRX 등락률은 +46.75% 다.
코드재사용 = [
    ("20160503", "036220", 3100, 3100.0, -1.90),
    ("20160504", "036220", 3500, 3500.0, 12.90),      # 인포피아 마지막 거래일
    ("20240313", "036220", 29350, 29350.0, 46.75),    # 오상헬스케어 첫 거래일
    ("20240314", "036220", 26200, 26200.0, -10.73),
]


def test_새_시계열의_첫_행은_전일이_없다():
    """8년 전 다른 회사의 종가로 수익률을 만들지 않는다.

    이 판정을 안 하면 `adj_return_1d` 가 +738.57% 로 나오고 `is_adj_suspect` 가 켜져
    검증기가 붉어진다(2026-09-09 실제로 겪었다). 켜는 근거는
    `common.corporate_actions.is_series_restart` 다.
    """
    끔 = flag_adjustment_quality(_frame(코드재사용))
    켬 = flag_adjustment_quality(
        _frame(코드재사용, first_listing=[False, False, True, False]))

    # 끄면 8년을 가로지르는 가짜 수익률이 생기고 오류로 표시된다
    assert 끔["adj_return_1d"].iloc[2] == pytest.approx(738.57, abs=0.01)
    assert bool(끔["is_adj_suspect"].iloc[2]) is True

    # 켜면 비교 자체를 하지 않는다 — 모르는 것을 오류로 치지 않는다
    assert np.isnan(켬["adj_return_1d"].iloc[2])
    assert np.isnan(켬["adj_change_rate_gap"].iloc[2])
    assert bool(켬["is_adj_suspect"].iloc[2]) is False
    assert bool(켬["is_extreme_return"].iloc[2]) is False

    # 그 다음 날부터는 새 회사 안에서 정상적으로 이어진다 — 둘 다 같다
    assert 켬["adj_return_1d"].iloc[3] == pytest.approx(-10.73, abs=0.01)
    assert 켬["adj_return_1d"].iloc[3] == pytest.approx(끔["adj_return_1d"].iloc[3])
    # 앞 회사 구간도 건드리지 않는다
    assert 켬["adj_return_1d"].iloc[1] == pytest.approx(끔["adj_return_1d"].iloc[1])
    assert 켬.attrs["adjustment_quality"]["comparable_rows"] == 2
    assert 끔.attrs["adjustment_quality"]["comparable_rows"] == 3


def test_종목의_진짜_첫_행에_켜져도_결과는_같다():
    """`is_first_listing` 은 신규상장 1,502행 전부에 켜져 있다.

    그 중 코드 재사용 2행을 뺀 나머지는 애초에 전일 adj 가 없어 비교 불가다. 즉 이
    조건은 **그 행들에 아무것도 안 바꾼다** — 바꾸면 멀쩡한 판정이 조용히 사라진다.
    """
    끔 = flag_adjustment_quality(_frame(경남에너지))
    켬 = flag_adjustment_quality(
        _frame(경남에너지, first_listing=[True, False, False]))   # 첫 행이 신규상장
    pd.testing.assert_frame_equal(끔, 켬)


def test_is_first_listing_이_없으면_무엇을_해야_하는지_알려_준다():
    """가드를 막다른 길로 만들지 않는다 — 어디서 그 칸을 얻는지까지 말한다."""
    frame = _frame(경남에너지).drop(columns=["is_first_listing"])
    with pytest.raises(ValueError, match="is_first_listing") as e:
        flag_adjustment_quality(frame)
    말 = str(e.value)
    assert "daily_price_dev.parquet" in 말                    # 어디서 받나
    assert "attach_corporate_action_flags" in 말              # DB 경로라면 무엇을 부르나
