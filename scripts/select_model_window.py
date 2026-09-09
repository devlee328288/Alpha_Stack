"""대표 조합으로 전역 학습창을 선택하고 재현 가능한 JSON을 남긴다."""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from features.model_dataset import build_model_dataset  # noqa: E402
from models.experiment import (  # noqa: E402
    LABEL_HORIZON,
    MIN_TRAIN_SIZE,
    N_FOLDS,
    ROLLING_WINDOWS,
    ROUND_TRIP_COST,
    VALID_SIZE,
    evaluate_window_candidates,
    summarize_window_results,
)
from supply.hf_model_data import REPO_ID, load_hf_index_prices  # noqa: E402

OUTPUT = ROOT / "reports" / "window_selection.json"


def main() -> int:
    snapshot = load_hf_index_prices()
    dataset = build_model_dataset(
        snapshot.frame,
        "F",
        return_features=("daily_return", "five_day_return"),
    )
    print(
        f"HF {REPO_ID}@{snapshot.repo_sha}\n"
        f"index SHA-256 {snapshot.file_sha256}\n"
        f"대표 데이터 {len(dataset.frame):,}행 "
        f"({dataset.frame['bas_dd'].min()}~{dataset.frame['bas_dd'].max()})"
    )
    results = evaluate_window_candidates(dataset, class_weight=None)
    summary, selected = summarize_window_results(results)
    print("\n학습창 요약")
    print(summary.to_string(index=False, float_format=lambda value: f"{value:.6f}"))
    print(f"\n선택: {selected}")

    payload = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source": {
            "repo": REPO_ID,
            "repo_sha": snapshot.repo_sha,
            "index_sha256": snapshot.file_sha256,
            "generated_at": snapshot.generated_at,
            "dev_end": snapshot.dev_end,
        },
        "representative": {
            "combination": "F",
            "return_features": ["daily_return", "five_day_return"],
            "rows": len(dataset.frame),
            "first_date": str(dataset.frame["bas_dd"].min()),
            "last_date": str(dataset.frame["bas_dd"].max()),
        },
        "rules": {
            "folds": N_FOLDS,
            "valid_size": VALID_SIZE,
            "gap": LABEL_HORIZON,
            "minimum_train": MIN_TRAIN_SIZE,
            "rolling_windows": list(ROLLING_WINDOWS),
            "common_validation_start_after": max(ROLLING_WINDOWS),
            "class_weight": None,
            "round_trip_cost": ROUND_TRIP_COST,
            "all_cash_sharpe": 0.0,
            "position": "상승=다음날 시가 진입, 중립·하락=현금, 5개 슬리브로 5일 보유",
            "selection": "4모델×12폴드 delta_sharpe_net 중앙값 최대, 동률이면 expanding",
        },
        "selected_window": selected,
        "summary": summary.to_dict(orient="records"),
        "model_medians": (
            results.groupby(["window", "model"])["delta_sharpe_net"]
            .median()
            .unstack("model")
            .reset_index()
            .to_dict(orient="records")
        ),
        "fold_results": results.to_dict(orient="records"),
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"결과 저장: {OUTPUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
