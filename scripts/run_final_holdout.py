"""확정된 C/K 모델의 드라이런 또는 승인된 공식 홀드아웃 평가를 실행한다."""

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
from models.final_holdout import (  # noqa: E402
    FinalHoldoutResult,
    config_from_dict,
    run_final_holdout,
)

DEFAULT_CONFIG_PATH = ROOT / "config" / "final_holdout_model.json"
SOURCE_NAMES = ("index_dev", "index_holdout", "stock_dev", "stock_holdout")


def sha256_file(path: Path) -> str:
    """원본을 메모리에 전부 올리지 않고 입력 파일 지문을 계산한다."""

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_run_mode(
    *,
    mode: str,
    evaluation_dates: list[str],
    config_sha256: str,
    source_sha256: dict[str, str],
    authorization: dict[str, object] | None,
) -> str:
    """드라이런과 공식 개봉을 날짜 및 승인 파일로 서로 섞이지 않게 가른다."""

    if not evaluation_dates:
        raise ValueError("평가구간 날짜가 없습니다.")
    first = min(evaluation_dates)
    last = max(evaluation_dates)
    if mode == "dry-run":
        if last >= HOLDOUT_START:
            raise RuntimeError("드라이런은 실제 홀드아웃 날짜를 읽을 수 없습니다.")
        if authorization is not None:
            raise ValueError("드라이런에는 공식 개봉 승인 파일을 사용하지 않습니다.")
        return "dry-run"
    if mode != "official":
        raise ValueError(f"지원하지 않는 실행 모드입니다: {mode}")
    if first < HOLDOUT_START:
        raise RuntimeError("공식 평가는 홀드아웃 시작일 이전 행을 포함할 수 없습니다.")
    if authorization is None:
        raise RuntimeError("공식 평가는 데이터 담당자가 만든 개봉 승인 JSON이 필요합니다.")
    required = {
        "schema_version",
        "authorized",
        "run_id",
        "holdout_start",
        "config_sha256",
        "source_sha256",
    }
    missing = required - set(authorization)
    if missing:
        raise ValueError(f"개봉 승인 JSON 항목이 없습니다: {sorted(missing)}")
    if authorization["schema_version"] != 1 or authorization["authorized"] is not True:
        raise RuntimeError("개봉 승인 JSON이 활성 상태가 아닙니다.")
    if authorization["holdout_start"] != HOLDOUT_START:
        raise RuntimeError("개봉 승인 JSON의 홀드아웃 경계가 코드 정본과 다릅니다.")
    if authorization["config_sha256"] != config_sha256:
        raise RuntimeError("개봉 승인 뒤 최종 모델 설정이 변경됐습니다.")
    if authorization["source_sha256"] != source_sha256:
        raise RuntimeError("개봉 승인 JSON과 실제 입력 파일 SHA-256이 다릅니다.")
    run_id = str(authorization["run_id"]).strip()
    if not run_id:
        raise ValueError("개봉 승인 JSON의 run_id가 비어 있습니다.")
    return run_id


def _read_frame(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"입력 Parquet이 없습니다: {path}")
    return pd.read_parquet(path)


def _write_result(
    output_dir: Path,
    *,
    mode: str,
    run_id: str,
    result: FinalHoldoutResult,
    source_paths: dict[str, Path],
    source_sha256: dict[str, str],
) -> None:
    if output_dir.exists():
        raise FileExistsError(f"기존 결과를 덮어쓰지 않습니다: {output_dir}")
    output_dir.mkdir(parents=True)
    result.index_predictions.to_parquet(output_dir / "index_predictions.parquet", index=False)
    result.stock_predictions.to_parquet(output_dir / "stock_predictions.parquet", index=False)
    result.combined_predictions.to_parquet(
        output_dir / "combined_predictions.parquet",
        index=False,
    )
    report = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "mode": mode,
        "run_id": run_id,
        "holdout_start": HOLDOUT_START,
        "config_sha256": result.config_sha256,
        "source": {
            name: {"path": str(source_paths[name]), "sha256": source_sha256[name]}
            for name in SOURCE_NAMES
        },
        "models": {
            "index": "C LogisticRegression",
            "stock": "K LogisticRegression",
        },
        "rows": {
            "index": len(result.index_predictions),
            "stock": len(result.stock_predictions),
            "combined": len(result.combined_predictions),
            "buy_candidates": int(result.combined_predictions["buy_candidate"].sum()),
        },
        "metrics": result.metrics,
        "note": "결과를 본 뒤 모델·피처·임계값을 다시 선택하지 않는다.",
    }
    (output_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("dry-run", "official"), required=True)
    parser.add_argument("--index-dev", type=Path, required=True)
    parser.add_argument("--index-holdout", type=Path, required=True)
    parser.add_argument("--stock-dev", type=Path, required=True)
    parser.add_argument("--stock-holdout", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--authorization", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    source_paths = {
        "index_dev": args.index_dev,
        "index_holdout": args.index_holdout,
        "stock_dev": args.stock_dev,
        "stock_holdout": args.stock_holdout,
    }
    frames = {name: _read_frame(path) for name, path in source_paths.items()}
    source_sha256 = {name: sha256_file(path) for name, path in source_paths.items()}
    config = config_from_dict(json.loads(args.config.read_text(encoding="utf-8")))
    authorization = (
        json.loads(args.authorization.read_text(encoding="utf-8"))
        if args.authorization is not None
        else None
    )
    evaluation_dates = sorted(
        set(frames["index_holdout"]["bas_dd"].astype(str))
        | set(frames["stock_holdout"]["bas_dd"].astype(str))
    )
    run_id = validate_run_mode(
        mode=args.mode,
        evaluation_dates=evaluation_dates,
        config_sha256=config.sha256,
        source_sha256=source_sha256,
        authorization=authorization,
    )
    result = run_final_holdout(
        frames["index_dev"],
        frames["index_holdout"],
        frames["stock_dev"],
        frames["stock_holdout"],
        config,
    )
    _write_result(
        args.output_dir,
        mode=args.mode,
        run_id=run_id,
        result=result,
        source_paths=source_paths,
        source_sha256=source_sha256,
    )
    print(f"최종모델 {args.mode} 결과 저장: {args.output_dir}")


if __name__ == "__main__":
    main()
