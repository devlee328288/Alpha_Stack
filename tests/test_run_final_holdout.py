import pytest

from evaluation.horizon import HOLDOUT_START
from scripts.run_final_holdout import validate_run_mode

SHA = {name: name * 8 for name in ("index_dev", "index_holdout", "stock_dev", "stock_holdout")}


def _authorization():
    return {
        "schema_version": 1,
        "authorized": True,
        "run_id": "final-20260911",
        "holdout_start": HOLDOUT_START,
        "config_sha256": "config-sha",
        "source_sha256": SHA,
    }


def test_드라이런은_실제홀드아웃날짜를읽지않는다():
    with pytest.raises(RuntimeError, match="실제 홀드아웃"):
        validate_run_mode(
            mode="dry-run",
            evaluation_dates=[HOLDOUT_START],
            config_sha256="config-sha",
            source_sha256=SHA,
            authorization=None,
        )


def test_공식평가는_승인된입력해시와설정만허용한다():
    assert validate_run_mode(
        mode="official",
        evaluation_dates=[HOLDOUT_START, "20240902"],
        config_sha256="config-sha",
        source_sha256=SHA,
        authorization=_authorization(),
    ) == "final-20260911"


def test_공식평가는_입력파일이바뀌면중단한다():
    authorization = _authorization()
    authorization["source_sha256"] = {**SHA, "stock_holdout": "changed"}

    with pytest.raises(RuntimeError, match="입력 파일 SHA-256"):
        validate_run_mode(
            mode="official",
            evaluation_dates=[HOLDOUT_START],
            config_sha256="config-sha",
            source_sha256=SHA,
            authorization=authorization,
        )


def test_공식평가는_승인파일없이는중단한다():
    with pytest.raises(RuntimeError, match="개봉 승인 JSON"):
        validate_run_mode(
            mode="official",
            evaluation_dates=[HOLDOUT_START],
            config_sha256="config-sha",
            source_sha256=SHA,
            authorization=None,
        )
