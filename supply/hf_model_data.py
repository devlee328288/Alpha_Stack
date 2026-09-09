"""모델 입력용 HF 지수 Parquet를 고정 리비전에서 검증해 읽는다."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow.parquet as pq

from common import secrets
from evaluation.horizon import HOLDOUT_START

REPO_ID = "qurious-quant/alphastack-krx-dev"
INDEX_FILE = "full/index_price_dev.parquet"
DAILY_FILE = "full/daily_price_dev.parquet"
ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CACHE_DIR = ROOT / "data" / "raw" / "hf_cache"

DAILY_FEATURE_COLUMNS = (
    "bas_dd",
    "code",
    "market",
    "volume",
    "value",
    "market_cap",
    "adj_close",
)


@dataclass(frozen=True)
class HfIndexSnapshot:
    """검증을 통과한 HF 지수 원시표와 출처 정보."""

    frame: pd.DataFrame
    repo_sha: str
    file_sha256: str
    generated_at: str
    dev_end: str


@dataclass(frozen=True)
class HfMarketSnapshot:
    """같은 HF 커밋에서 검증한 지수표와 전 종목 피처 원천표."""

    index_frame: pd.DataFrame
    daily_frame: pd.DataFrame
    repo_sha: str
    index_sha256: str
    daily_sha256: str
    generated_at: str
    dev_end: str


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_index_snapshot(
    parquet_path: Path,
    manifest: dict[str, Any],
    *,
    holdout_start: str = HOLDOUT_START,
) -> tuple[pd.DataFrame, str]:
    """MANIFEST와 실제 Parquet가 같고 홀드아웃 행이 없는지 확인한다."""

    if str(manifest.get("holdout_start")) != holdout_start:
        raise ValueError(
            "HF MANIFEST의 홀드아웃 시작일이 코드와 다릅니다: "
            f"{manifest.get('holdout_start')} != {holdout_start}"
        )
    records = [
        record
        for record in manifest.get("files", [])
        if Path(str(record.get("path", ""))).name == parquet_path.name
    ]
    if len(records) != 1:
        raise ValueError(f"MANIFEST에서 {parquet_path.name} 기록 하나를 찾지 못했습니다.")
    record = records[0]
    actual_sha = _sha256(parquet_path)
    if actual_sha != record.get("sha256"):
        raise ValueError(f"HF {parquet_path.name}의 SHA-256이 MANIFEST와 다릅니다.")

    frame = pd.read_parquet(parquet_path)
    if len(frame) != int(record["rows"]):
        raise ValueError(
            f"HF {parquet_path.name} 행 수가 MANIFEST와 다릅니다: "
            f"{len(frame)} != {record['rows']}"
        )
    if list(frame.columns) != list(record["columns"]):
        raise ValueError(f"HF {parquet_path.name} 열 구성이 MANIFEST와 다릅니다.")

    dates = (
        frame["bas_dd"]
        .astype("string")
        .str.replace(r"\.0$", "", regex=True)
        .str.zfill(8)
    )
    if (dates >= holdout_start).any():
        first = str(dates.loc[dates >= holdout_start].min())
        raise RuntimeError(f"HF 지수 Parquet에 홀드아웃 행이 들어 있습니다: {first}")
    dev_end = str(manifest.get("dev_end", ""))
    if not dev_end or str(dates.max()) > dev_end:
        raise ValueError(f"HF 지수 Parquet 최종일이 MANIFEST dev_end를 넘습니다: {dates.max()}")

    frame = frame.copy()
    frame["bas_dd"] = dates
    return frame, actual_sha


def validate_daily_snapshot(
    parquet_path: Path,
    manifest: dict[str, Any],
    *,
    holdout_start: str = HOLDOUT_START,
) -> tuple[pd.DataFrame, str]:
    """전 종목 Parquet를 검증하고 시장 집계에 필요한 열만 읽는다."""

    if str(manifest.get("holdout_start")) != holdout_start:
        raise ValueError(
            "HF MANIFEST의 홀드아웃 시작일이 코드와 다릅니다: "
            f"{manifest.get('holdout_start')} != {holdout_start}"
        )
    records = [
        record
        for record in manifest.get("files", [])
        if Path(str(record.get("path", ""))).name == parquet_path.name
    ]
    if len(records) != 1:
        raise ValueError(f"MANIFEST에서 {DAILY_FILE} 기록 하나를 찾지 못했습니다.")
    record = records[0]
    actual_sha = _sha256(parquet_path)
    if actual_sha != record.get("sha256"):
        raise ValueError(f"HF {parquet_path.name}의 SHA-256이 MANIFEST와 다릅니다.")

    metadata = pq.ParquetFile(parquet_path).metadata
    if metadata.num_rows != int(record["rows"]):
        raise ValueError(
            f"HF {parquet_path.name} 행 수가 MANIFEST와 다릅니다: "
            f"{metadata.num_rows} != {record['rows']}"
        )
    schema_columns = pq.ParquetFile(parquet_path).schema_arrow.names
    if schema_columns != list(record["columns"]):
        raise ValueError(f"HF {parquet_path.name} 열 구성이 MANIFEST와 다릅니다.")
    missing = set(DAILY_FEATURE_COLUMNS) - set(schema_columns)
    if missing:
        raise ValueError(f"시장 집계에 필요한 열이 없습니다: {sorted(missing)}")

    frame = pd.read_parquet(parquet_path, columns=list(DAILY_FEATURE_COLUMNS))
    dates = (
        frame["bas_dd"]
        .astype("string")
        .str.replace(r"\.0$", "", regex=True)
        .str.zfill(8)
    )
    if (dates >= holdout_start).any():
        first = str(dates.loc[dates >= holdout_start].min())
        raise RuntimeError(f"HF 전 종목 Parquet에 홀드아웃 행이 들어 있습니다: {first}")
    dev_end = str(manifest.get("dev_end", ""))
    if not dev_end or str(dates.max()) > dev_end:
        raise ValueError(f"HF 전 종목 Parquet 최종일이 MANIFEST dev_end를 넘습니다: {dates.max()}")

    frame = frame.copy()
    frame["bas_dd"] = dates
    return frame, actual_sha


def _hf_revision_and_token() -> tuple[str, str]:
    """private 여부를 확인하고 한 번의 실행에서 쓸 HF 커밋을 고정한다."""

    from huggingface_hub import HfApi

    token, source = secrets.load_key(
        ("HUGGINGFACE_ACCESS_TOKEN", "HF_TOKEN", "HUGGINGFACE_TOKEN")
    )
    if not token:
        raise RuntimeError(f"Hugging Face 토큰을 찾지 못했습니다: {source}")
    info = HfApi(token=token).repo_info(repo_id=REPO_ID, repo_type="dataset")
    if not info.private:
        raise RuntimeError("KRX 데이터가 있는 HF 저장소가 private 상태가 아닙니다.")
    return str(info.sha), token


def _download_hf_file(
    filename: str,
    *,
    revision: str,
    token: str,
    cache_dir: Path,
) -> Path:
    """고정한 HF 커밋에서 파일 하나를 내려받는다."""

    from huggingface_hub import hf_hub_download

    return Path(
        hf_hub_download(
            repo_id=REPO_ID,
            repo_type="dataset",
            filename=filename,
            revision=revision,
            token=token,
            cache_dir=cache_dir,
        )
    )


def load_hf_index_prices(*, cache_dir: Path | None = None) -> HfIndexSnapshot:
    """HF private 데이터셋의 같은 커밋에서 MANIFEST와 지수 Parquet를 받는다."""

    target_cache = Path(cache_dir) if cache_dir is not None else DEFAULT_CACHE_DIR
    target_cache.mkdir(parents=True, exist_ok=True)
    revision, token = _hf_revision_and_token()
    manifest_path = _download_hf_file(
        "MANIFEST.json",
        revision=revision,
        token=token,
        cache_dir=target_cache,
    )
    parquet_path = _download_hf_file(
        INDEX_FILE,
        revision=revision,
        token=token,
        cache_dir=target_cache,
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    frame, file_sha = validate_index_snapshot(parquet_path, manifest)
    return HfIndexSnapshot(
        frame=frame,
        repo_sha=revision,
        file_sha256=file_sha,
        generated_at=str(manifest.get("generated_at", "")),
        dev_end=str(manifest.get("dev_end", "")),
    )


def load_hf_market_prices(*, cache_dir: Path | None = None) -> HfMarketSnapshot:
    """한 HF 커밋에서 지수와 전 종목 Parquet를 함께 고정·검증해 읽는다."""

    target_cache = Path(cache_dir) if cache_dir is not None else DEFAULT_CACHE_DIR
    target_cache.mkdir(parents=True, exist_ok=True)
    revision, token = _hf_revision_and_token()
    manifest_path = _download_hf_file(
        "MANIFEST.json",
        revision=revision,
        token=token,
        cache_dir=target_cache,
    )
    index_path = _download_hf_file(
        INDEX_FILE,
        revision=revision,
        token=token,
        cache_dir=target_cache,
    )
    daily_path = _download_hf_file(
        DAILY_FILE,
        revision=revision,
        token=token,
        cache_dir=target_cache,
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    index_frame, index_sha = validate_index_snapshot(index_path, manifest)
    daily_frame, daily_sha = validate_daily_snapshot(daily_path, manifest)
    return HfMarketSnapshot(
        index_frame=index_frame,
        daily_frame=daily_frame,
        repo_sha=revision,
        index_sha256=index_sha,
        daily_sha256=daily_sha,
        generated_at=str(manifest.get("generated_at", "")),
        dev_end=str(manifest.get("dev_end", "")),
    )
