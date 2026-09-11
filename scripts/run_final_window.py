"""개봉된 HF 원본 두 개로 마지막 T+1→T+6 구간의 C/K 모델을 실행한다."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from evaluation.horizon import HOLDOUT_START  # noqa: E402
from features.model_dataset import KOSPI200_NAME, build_model_dataset  # noqa: E402
from features.stock_model_dataset import build_sector_stock_model_dataset  # noqa: E402
from models.final_holdout import (  # noqa: E402
    FinalHoldoutResult,
    config_from_dict,
    run_final_holdout,
)
from models.final_window import (  # noqa: E402
    FinalWindow,
    derive_final_window,
    split_index_window,
    split_stock_window,
)
from supply.adj_quality import attach_adjustment_quality  # noqa: E402
from supply.stock_training_universe import build_sector_candidate_frame  # noqa: E402

DEFAULT_INDEX_PATH = ROOT / "data" / "raw" / "hf_snapshot" / "full" / "index_price_dev.parquet"
DEFAULT_DAILY_PATH = ROOT / "data" / "raw" / "hf_snapshot" / "full" / "daily_price_dev.parquet"
DEFAULT_CONFIG_PATH = ROOT / "config" / "final_holdout_model.json"
DEFAULT_UNSEAL_LOG = ROOT / "reports" / "unseal.log"
DEFAULT_SHARED_OUTPUT_DIR = ROOT / "notebooks" / "04-모델" / "최종결과"

LABEL_NAMES = {-1: "하락", 0: "보합", 1: "상승"}
SHARED_STOCK_METADATA = (
    "bas_dd",
    "code",
    "name",
    "industry_index_name",
    "sector_market_cap_rank",
    "industry_stock_rank",
)

DAILY_COLUMNS = (
    "bas_dd",
    "code",
    "name",
    "market",
    "kind_stkcert_tp_nm",
    "is_liquidation",
    "is_halted",
    "is_first_listing",
    "market_cap",
    "value",
    "industry",
    "adj_open",
    "adj_high",
    "adj_low",
    "adj_close",
    "close",
    "change_rate",
    "volume",
)
SAMPLE_EXCLUSION_COLUMNS = ("is_liquidation", "is_halted", "is_first_listing")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _normalize_dates(values: pd.Series) -> pd.Series:
    dates = (
        values.astype("string")
        .str.strip()
        .str.replace(r"\.0$", "", regex=True)
        .str.zfill(8)
    )
    if dates.isna().any() or (~dates.str.fullmatch(r"\d{8}")).any():
        raise ValueError("HF 원본 bas_dd에 YYYYMMDD가 아닌 값이 있습니다.")
    return dates


def require_single_unseal(log_path: Path) -> None:
    """공식 개봉 기록이 정확히 한 줄일 때만 새 실행 경로를 연다."""

    if not log_path.exists():
        raise FileNotFoundError(f"홀드아웃 개봉 기록이 없습니다: {log_path}")
    rows = [
        line
        for line in log_path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    if len(rows) != 1:
        raise RuntimeError(f"홀드아웃 개봉 기록은 정확히 한 줄이어야 합니다: {len(rows)}줄")


def _exclude_corporate_action_samples(candidates: pd.DataFrame) -> pd.DataFrame:
    missing = set(SAMPLE_EXCLUSION_COLUMNS) - set(candidates.columns)
    if missing:
        raise ValueError(f"기업행위 표본 선택 열이 없습니다: {sorted(missing)}")
    flags = candidates.loc[:, SAMPLE_EXCLUSION_COLUMNS]
    if flags.isna().any().any():
        raise ValueError("기업행위 표본 선택 열에 결측이 있습니다.")
    selected = candidates.loc[~flags.astype(bool).any(axis=1)].copy()
    selected.attrs.update(candidates.attrs)
    return selected


def _write_outputs(
    output_dir: Path,
    *,
    shared_output_dir: Path,
    result: FinalHoldoutResult,
    stock_test: pd.DataFrame,
    window: FinalWindow,
    index_path: Path,
    daily_path: Path,
    unseal_log: Path,
) -> None:
    if output_dir.resolve() == shared_output_dir.resolve():
        raise ValueError("로컬 원본 결과와 팀 공유 결과 디렉터리는 달라야 합니다.")
    for target in (output_dir, shared_output_dir):
        if target.exists():
            raise FileExistsError(f"기존 결과를 덮어쓰지 않습니다: {target}")
    # 공유 열 계약도 실제 파일을 만들기 전에 검사해 실패 시 반쪽 결과가 남지 않게 한다.
    shared_rows = _build_shared_stock_rows(result, stock_test)
    shared_index = _shared_index_result(result.index_predictions)

    output_dir.mkdir(parents=True)
    result.index_predictions.to_parquet(output_dir / "index_predictions.parquet", index=False)
    result.stock_predictions.to_parquet(output_dir / "stock_predictions.parquet", index=False)
    result.combined_predictions.to_parquet(
        output_dir / "combined_predictions.parquet", index=False
    )
    report = {
        "schema_version": 2,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "mode": "unsealed_final_window",
        "models": {"index": "C LogisticRegression", "stock": "K LogisticRegression"},
        "window": {
            "decision_date": window.decision_date,
            "entry_date": window.entry_date,
            "exit_date": window.exit_date,
            "rule": "T일 종가 판단 → T+1 시가 진입 → T+6 시가 청산",
        },
        "source": {
            "index": {"path": str(index_path), "sha256": sha256_file(index_path)},
            "daily": {"path": str(daily_path), "sha256": sha256_file(daily_path)},
            "unseal_log": {"path": str(unseal_log), "sha256": sha256_file(unseal_log)},
        },
        "config_sha256": result.config_sha256,
        "rows": {
            "index": len(result.index_predictions),
            "stock": len(result.stock_predictions),
            "combined": len(result.combined_predictions),
            "buy_candidates": int(result.combined_predictions["buy_candidate"].sum()),
        },
        "metrics": result.metrics,
        "note": (
            "개봉 후 목적을 마지막 T+1→T+6 한 구간 예측으로 바로잡았다. "
            "최종 목표 구간을 사용하는 라벨은 학습에서 제외했다."
        ),
    }
    report_text = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    (output_dir / "report.json").write_text(report_text, encoding="utf-8")

    shared_output_dir.mkdir(parents=True)
    (shared_output_dir / "report.json").write_text(report_text, encoding="utf-8")
    shared_payload = {
        "schema_version": 1,
        "window": report["window"],
        "index": shared_index,
        "stocks": shared_rows.to_dict(orient="records"),
    }
    (shared_output_dir / "final_signals.json").write_text(
        json.dumps(shared_payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (shared_output_dir / "README.md").write_text(
        _shared_readme(report, shared_payload, shared_rows), encoding="utf-8"
    )


def _label_name(value: object) -> str:
    return LABEL_NAMES[int(value)]


def _shared_index_result(predictions: pd.DataFrame) -> dict[str, object]:
    if len(predictions) != 1:
        raise ValueError("공유 결과의 KOSPI200 예측은 정확히 한 행이어야 합니다.")
    row = predictions.iloc[0]
    return {
        "actual": _label_name(row["actual"]),
        "prediction": _label_name(row["predicted"]),
        "p_down": float(row["p_down"]),
        "p_flat": float(row["p_neutral"]),
        "p_up": float(row["p_up"]),
        "hit": "O" if int(row["actual"]) == int(row["predicted"]) else "X",
    }


def _build_shared_stock_rows(
    result: FinalHoldoutResult,
    stock_test: pd.DataFrame,
) -> pd.DataFrame:
    missing = set(SHARED_STOCK_METADATA) - set(stock_test.columns)
    if missing:
        raise ValueError(f"공유 종목표 메타데이터가 없습니다: {sorted(missing)}")
    metadata = stock_test.loc[:, list(SHARED_STOCK_METADATA)].copy()
    metadata["code"] = metadata["code"].astype("string")
    if metadata.duplicated(["bas_dd", "code"]).any():
        raise ValueError("공유 종목표 메타데이터의 날짜·종목코드가 중복됐습니다.")

    rows = result.combined_predictions.merge(
        metadata,
        on=["bas_dd", "code"],
        how="left",
        validate="one_to_one",
    )
    if rows.loc[:, list(SHARED_STOCK_METADATA[2:])].isna().any().any():
        raise ValueError("예측 결과와 공유 종목표 메타데이터를 전부 연결하지 못했습니다.")
    rows = rows.sort_values(
        ["sector_market_cap_rank", "industry_stock_rank", "code"], kind="stable"
    )
    shared = pd.DataFrame(
        {
            "sector_rank": rows["sector_market_cap_rank"].astype(int),
            "industry": rows["industry_index_name"].astype(str),
            "name": rows["name"].astype(str),
            "code": rows["code"].astype(str),
            "industry_market_cap_rank": rows["industry_stock_rank"].astype(int),
            "stock_prediction": rows["stock_predicted"].map(_label_name),
            "p_up": rows["p_up"].astype(float),
            "p_flat": rows["p_neutral"].astype(float),
            "p_down": rows["p_down"].astype(float),
            "actual": rows["stock_actual"].map(_label_name),
            "hit": rows.apply(
                lambda row: "O"
                if int(row["stock_actual"]) == int(row["stock_predicted"])
                else "X",
                axis=1,
            ),
            "buy_candidate": rows["buy_candidate"].map({True: "O", False: "X"}),
        }
    )
    return shared.reset_index(drop=True)


def _shared_readme(
    report: dict[str, object],
    payload: dict[str, object],
    rows: pd.DataFrame,
) -> str:
    window = report["window"]
    index = payload["index"]
    lines = [
        "# 최종 모델 결과",
        "",
        f"- 판단일: `{window['decision_date']}`",
        f"- 진입일: `{window['entry_date']}`",
        f"- 청산일: `{window['exit_date']}`",
        "- KOSPI200 예측/실제/적중: "
        f"`{index['prediction']}` / `{index['actual']}` / `{index['hit']}`",
        f"- 매수 후보: `{int((rows['buy_candidate'] == 'O').sum())}`건",
        "",
        "| 업종 순위 | 업종 | 종목명·코드 | 업종 내 시총 순위 | 종목 예측 | "
        "p_up | p_flat | p_down | 실제 결과 | 적중 | 매수 후보 |",
        "|---:|---|---|---:|---|---:|---:|---:|---|:---:|:---:|",
    ]
    for row in rows.itertuples(index=False):
        lines.append(
            f"| {row.sector_rank} | {row.industry} | {row.name}·{row.code} | "
            f"{row.industry_market_cap_rank} | {row.stock_prediction} | "
            f"{row.p_up:.4f} | {row.p_flat:.4f} | {row.p_down:.4f} | "
            f"{row.actual} | {row.hit} | {row.buy_candidate} |"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index-path", type=Path, default=DEFAULT_INDEX_PATH)
    parser.add_argument("--daily-path", type=Path, default=DEFAULT_DAILY_PATH)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--unseal-log", type=Path, default=DEFAULT_UNSEAL_LOG)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--shared-output-dir", type=Path, default=DEFAULT_SHARED_OUTPUT_DIR)
    args = parser.parse_args()

    require_single_unseal(args.unseal_log)
    for path in (args.index_path, args.daily_path):
        if not path.exists():
            raise FileNotFoundError(f"HF 최신 원본이 없습니다: {path}")

    config = config_from_dict(json.loads(args.config.read_text(encoding="utf-8")))
    index_prices = pd.read_parquet(args.index_path)
    daily = pd.read_parquet(
        args.daily_path,
        columns=list(DAILY_COLUMNS),
        filters=[("market", "==", "KOSPI")],
    )
    index_dates = set(
        _normalize_dates(
            index_prices.loc[index_prices["index_name"].eq(KOSPI200_NAME), "bas_dd"]
        )
    )
    daily_dates = set(_normalize_dates(daily["bas_dd"]))
    common_dates = sorted(index_dates & daily_dates)
    window = derive_final_window(common_dates)
    if window.decision_date < HOLDOUT_START:
        raise RuntimeError("마지막 평가 구간이 개봉된 홀드아웃 안에 있지 않습니다.")

    index_dataset = build_model_dataset(
        index_prices,
        config.index_combination,
        holdout_start=HOLDOUT_START,
        allow_unsealed=True,
    )
    candidates = build_sector_candidate_frame(
        daily,
        index_prices,
        holdout_start=HOLDOUT_START,
        allow_unsealed=True,
    )
    candidates = _exclude_corporate_action_samples(candidates)
    candidates = attach_adjustment_quality(candidates, daily)
    candidates = candidates.loc[~candidates["is_adj_suspect"]].copy()
    stock_dataset = build_sector_stock_model_dataset(
        daily,
        candidates,
        index_prices=index_prices,
        feature_columns=config.stock_features,
        drop_incomplete_features=False,
        holdout_start=HOLDOUT_START,
        allow_unsealed=True,
    )
    index_train, index_test = split_index_window(index_dataset.frame, window)
    stock_train, stock_test = split_stock_window(stock_dataset.frame, window)
    result = run_final_holdout(index_train, index_test, stock_train, stock_test, config)
    _write_outputs(
        args.output_dir,
        shared_output_dir=args.shared_output_dir,
        result=result,
        stock_test=stock_test,
        window=window,
        index_path=args.index_path,
        daily_path=args.daily_path,
        unseal_log=args.unseal_log,
    )
    print(
        "최종 한 구간 실행 완료: "
        f"판단 {window.decision_date} → 진입 {window.entry_date} → 청산 {window.exit_date} · "
        f"매수 후보 {int(result.combined_predictions['buy_candidate'].sum())}건"
    )


if __name__ == "__main__":
    main()
