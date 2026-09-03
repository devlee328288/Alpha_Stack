import hashlib

import pandas as pd
import pytest

from supply.hf_model_data import validate_daily_snapshot, validate_index_snapshot


def _write_snapshot(tmp_path, dates):
    frame = pd.DataFrame(
        {
            "bas_dd": dates,
            "index_name": ["코스피 200"] * len(dates),
            "open": [100.0] * len(dates),
        }
    )
    path = tmp_path / "index_price_dev.parquet"
    frame.to_parquet(path, index=False)
    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    manifest = {
        "holdout_start": "20240901",
        "dev_end": "20240831",
        "files": [
            {
                "path": path.name,
                "rows": len(frame),
                "columns": list(frame.columns),
                "sha256": sha,
            }
        ],
    }
    return path, manifest


def test_MANIFEST와_같은_HF_지수_Parquet만_통과한다(tmp_path):
    path, manifest = _write_snapshot(tmp_path, ["20240829", "20240830"])

    frame, actual_sha = validate_index_snapshot(path, manifest)

    assert frame["bas_dd"].tolist() == ["20240829", "20240830"]
    assert actual_sha == manifest["files"][0]["sha256"]


def test_HF_지수_Parquet에_홀드아웃_행이_있으면_중단한다(tmp_path):
    path, manifest = _write_snapshot(tmp_path, ["20240830", "20240902"])

    with pytest.raises(RuntimeError, match="홀드아웃 행"):
        validate_index_snapshot(path, manifest)


def _write_daily_snapshot(tmp_path, dates):
    frame = pd.DataFrame(
        {
            "bas_dd": dates,
            "code": ["005930"] * len(dates),
            "name": ["삼성전자"] * len(dates),
            "market": ["KOSPI"] * len(dates),
            "sector": ["전기전자"] * len(dates),
            "open": [100.0] * len(dates),
            "high": [101.0] * len(dates),
            "low": [99.0] * len(dates),
            "close": [100.0] * len(dates),
            "change": [0.0] * len(dates),
            "change_rate": [0.0] * len(dates),
            "volume": [1_000.0] * len(dates),
            "value": [100_000.0] * len(dates),
            "market_cap": [1_000_000.0] * len(dates),
            "listed_shares": [10_000.0] * len(dates),
            "adj_open": [100.0] * len(dates),
            "adj_high": [101.0] * len(dates),
            "adj_low": [99.0] * len(dates),
            "adj_close": [100.0] * len(dates),
            "adj_source": ["none"] * len(dates),
        }
    )
    path = tmp_path / "daily_price_dev.parquet"
    frame.to_parquet(path, index=False)
    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    manifest = {
        "holdout_start": "20240901",
        "dev_end": "20240831",
        "files": [
            {
                "path": "full/daily_price_dev.parquet",
                "rows": len(frame),
                "columns": list(frame.columns),
                "sha256": sha,
            }
        ],
    }
    return path, manifest


def test_HF_전종목_Parquet는_검증후_피처원천열만_읽는다(tmp_path):
    path, manifest = _write_daily_snapshot(tmp_path, ["20240829", "20240830"])

    frame, actual_sha = validate_daily_snapshot(path, manifest)

    assert list(frame.columns) == [
        "bas_dd",
        "code",
        "market",
        "volume",
        "value",
        "market_cap",
        "adj_close",
    ]
    assert actual_sha == manifest["files"][0]["sha256"]


def test_HF_전종목_Parquet에_홀드아웃_행이_있으면_중단한다(tmp_path):
    path, manifest = _write_daily_snapshot(tmp_path, ["20240830", "20240902"])

    with pytest.raises(RuntimeError, match="홀드아웃 행"):
        validate_daily_snapshot(path, manifest)
