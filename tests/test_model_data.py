import pandas as pd
import pytest

from supply.model_data import holdout_safe_frame


def test_홀드아웃과_경계_직전_5거래일을_모두_제거한다():
    dates = [f"202408{day:02d}" for day in range(20, 32)] + ["20240902", "20240903"]
    frame = pd.DataFrame({"bas_dd": dates, "value": range(len(dates))})

    result = holdout_safe_frame(frame, holdout_start="20240901", label_horizon=5)

    assert result["bas_dd"].tolist() == [
        "20240820",
        "20240821",
        "20240822",
        "20240823",
        "20240824",
        "20240825",
        "20240826",
    ]
    assert result.attrs["holdout_filter"]["purged_dates"] == [
        "20240827",
        "20240828",
        "20240829",
        "20240830",
        "20240831",
    ]
    assert result.attrs["holdout_filter"]["holdout_rows_removed"] == 2


def test_같은_거래일의_여러_행은_날짜_단위로_함께_제거한다():
    dates = [f"202408{day:02d}" for day in range(20, 32)]
    frame = pd.DataFrame(
        {
            "bas_dd": [date for date in dates for _ in range(2)] + ["20240902"],
            "code": ["A", "B"] * len(dates) + ["A"],
        }
    )

    result = holdout_safe_frame(frame, holdout_start="20240901", label_horizon=5)

    assert result["bas_dd"].max() == "20240826"
    assert result.attrs["holdout_filter"]["boundary_rows_removed"] == 10


def test_원본_프레임은_바꾸지_않는다():
    frame = pd.DataFrame(
        {"bas_dd": [f"202408{day:02d}" for day in range(20, 32)], "value": range(12)}
    )
    original = frame.copy(deep=True)

    holdout_safe_frame(frame, holdout_start="20240901", label_horizon=5)

    pd.testing.assert_frame_equal(frame, original)


def test_이미_잘린_dev_전용_파일은_마지막_5일을_다시_버리지_않는다():
    dates = [f"202108{day:02d}" for day in range(10, 24)]
    frame = pd.DataFrame({"bas_dd": dates, "value": range(len(dates))})

    result = holdout_safe_frame(frame, holdout_start="20240901", label_horizon=5)

    assert result["bas_dd"].tolist() == dates
    assert result.attrs["holdout_filter"]["boundary_rows_removed"] == 0


def test_기준일_열이_없으면_즉시_중단한다():
    with pytest.raises(ValueError, match="기준일 열"):
        holdout_safe_frame(pd.DataFrame({"date": ["20240801"]}))


def test_개발구간이_지평보다_짧으면_즉시_중단한다():
    frame = pd.DataFrame(
        {"bas_dd": ["20240829", "20240830", "20240831", "20240902"]}
    )

    with pytest.raises(ValueError, match="레이블 지평보다 짧습니다"):
        holdout_safe_frame(frame, holdout_start="20240901", label_horizon=5)
