import os
import warnings
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from step1_core_features import load_data
from step5_optimize_6params import compute_bands_flexible
from focal_classifier import build_features, make_labels, FocalMLP, set_seed

warnings.filterwarnings("ignore")

# ============================================================
# 0. 설정
# ============================================================
GAP_DAYS = 5
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# build_features()가 생성하는 34개 Feature를 그룹별로 분류
FEATURE_GROUPS = {
    "momentum": [
        "ret_1d",
        "ret_3d",
        "ret_5d",
        "ret_10d",
        "ret_20d",
    ],
    "volatility": [
        "logret_vol_5",
        "logret_vol_10",
        "logret_vol_20",
        "natr14",
        "range_pct",
        "body_pct",
        "close_location",
    ],
    "trend": [
        "close_sma5",
        "close_sma20",
        "close_sma60",
        "sma5_sma20",
        "sma20_sma60",
        "rsi14",
        "macd_pct",
        "macd_hist_pct",
    ],
    "volume": [
        "volume_ratio20",
        "volume_z20",
        "log_volume_ratio20",
    ],
    "band": [
        "band_base_gap",
        "band_position",
        "upper_gap_atr",
        "lower_gap_atr",
        "band_width_pct",
        "above_upper",
        "below_lower",
    ],
    "parameter": [
        "alpha_up",
        "alpha_down",
        "beta_up",
        "beta_down",
    ],
}

ALL_FEATURES = []
for group in FEATURE_GROUPS.values():
    ALL_FEATURES.extend(group)

# Ablation 설정: "ALL"과 각 그룹을 제거한 경우
ABLATION_CONFIGS = ["ALL"] + [f"ALL - {name}" for name in FEATURE_GROUPS.keys()]


# ============================================================
# 1. Cross Entropy MLP 학습 함수 (Focal과 동일한 구조, Loss만 CE)
# ============================================================
def train_ce_model(
    features: pd.DataFrame,
    y: pd.Series,
    train_end: int,
    feature_cols: List[str],
    seed: int = 42,
    hidden1: int = 64,
    hidden2: int = 32,
    dropout: float = 0.15,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    batch_size: int = 64,
    max_epochs: int = 180,
    patience: int = 20,
    inner_val_ratio: float = 0.20,
) -> Tuple[object, StandardScaler, Dict[str, float]]:
    """
    Cross Entropy Loss를 사용한 MLP 학습.
    내부 검증(inner validation)은 Macro-F1 기준 Early Stopping.
    """
    set_seed(seed)

    # 1) 레이블 안전 경계 (5일 미래 정보 제외)
    label_safe_end = max(0, train_end - 5)
    train_idx = np.arange(0, label_safe_end)

    # 2) Feature / Target 정리
    x_raw = features.iloc[train_idx][feature_cols].copy()
    y_raw = y.iloc[train_idx].to_numpy()

    valid_y = np.isfinite(y_raw)
    x_raw = x_raw.loc[valid_y]
    y_raw = y_raw[valid_y].astype(int)

    # NaN / Inf 제거
    x_raw = x_raw.replace([np.inf, -np.inf], np.nan)
    valid_x = np.isfinite(x_raw.to_numpy(dtype=float)).all(axis=1)
    x_raw = x_raw.loc[valid_x]
    y_raw = y_raw[valid_x]

    if len(x_raw) < 150 or len(np.unique(y_raw)) < 3:
        raise ValueError("CE 학습 데이터 부족 또는 클래스 누락")

    # 3) 내부 Train/Val 분할 (시간 순서 유지)
    split = int(len(x_raw) * (1.0 - inner_val_ratio))
    split = min(max(split, 100), len(x_raw) - 30)
    x_tr, x_va = x_raw.iloc[:split], x_raw.iloc[split:]
    y_tr, y_va = y_raw[:split], y_raw[split:]

    # 4) Scaling (Train만 fit)
    scaler = StandardScaler()
    scaler.fit(x_tr.to_numpy(dtype=np.float32))
    xtr = scaler.transform(x_tr.to_numpy(dtype=np.float32)).astype(np.float32)
    xva = scaler.transform(x_va.to_numpy(dtype=np.float32)).astype(np.float32)

    # 5) 모델 / Loss / Optimizer
    model = FocalMLP(len(feature_cols), hidden1, hidden2, dropout).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    loader = DataLoader(
        TensorDataset(
            torch.tensor(xtr, dtype=torch.float32), torch.tensor(y_tr, dtype=torch.long)
        ),
        batch_size=min(batch_size, len(xtr)),
        shuffle=True,
    )

    xva_t = torch.tensor(xva, dtype=torch.float32, device=DEVICE)
    yva_t = torch.tensor(y_va, dtype=torch.long, device=DEVICE)

    best_score = -np.inf
    best_state = None
    best_epoch = 1
    wait = 0

    # 6) 학습 루프
    for epoch in range(1, max_epochs + 1):
        model.train()
        for xb, yb in loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(xb), yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        # 내부 검증
        model.eval()
        with torch.no_grad():
            logits = model(xva_t)
            prob = torch.softmax(logits, dim=1).cpu().numpy()
        pred = prob.argmax(axis=1)
        score = f1_score(y_va, pred, average="macro", zero_division=0)

        if score > best_score + 1e-5:
            best_score = float(score)
            best_epoch = epoch
            best_state = {
                k: v.detach().cpu().clone() for k, v in model.state_dict().items()
            }
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    # 7) 전체 Train 데이터로 재학습 (동일한 epoch 수만큼)
    scaler_full = StandardScaler()
    scaler_full.fit(x_raw.to_numpy(dtype=np.float32))
    xfull = scaler_full.transform(x_raw.to_numpy(dtype=np.float32)).astype(np.float32)

    model_full = FocalMLP(len(feature_cols), hidden1, hidden2, dropout).to(DEVICE)
    optimizer_full = torch.optim.AdamW(
        model_full.parameters(), lr=lr, weight_decay=weight_decay
    )
    full_loader = DataLoader(
        TensorDataset(
            torch.tensor(xfull, dtype=torch.float32),
            torch.tensor(y_raw, dtype=torch.long),
        ),
        batch_size=min(batch_size, len(xfull)),
        shuffle=True,
    )

    for _ in range(best_epoch):
        model_full.train()
        for xb, yb in full_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer_full.zero_grad(set_to_none=True)
            loss = criterion(model_full(xb), yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model_full.parameters(), 1.0)
            optimizer_full.step()

    model_full.eval()
    stats = {
        "inner_macro_f1": best_score,
        "best_epoch": float(best_epoch),
        "train_rows": float(len(x_raw)),
    }
    return model_full, scaler_full, stats


# ============================================================
# 2. Ablation 실험 실행
# ============================================================
def run_ablation_experiment(
    df: pd.DataFrame,
    fold_details: pd.DataFrame,
    threshold: float = 0.02,
) -> Dict[str, Dict]:
    """
    각 Ablation 설정별로 Walk-Forward OOS Macro-F1을 계산.
    """
    date_to_idx = {d: i for i, d in enumerate(df.index)}
    results = {}

    # 각 설정에 대해 실행
    for config in tqdm(ABLATION_CONFIGS, desc="Ablation 설정"):
        # 사용할 Feature 목록 결정
        if config == "ALL":
            use_features = ALL_FEATURES.copy()
        else:
            removed_group = config.replace("ALL - ", "")
            removed_features = FEATURE_GROUPS[removed_group]
            use_features = [f for f in ALL_FEATURES if f not in removed_features]

        fold_scores = []

        for fold_no, row in fold_details.iterrows():
            val_start = pd.Timestamp(row["val_start"])
            val_end = pd.Timestamp(row["val_end"])

            if val_start not in date_to_idx or val_end not in date_to_idx:
                continue

            start_idx = date_to_idx[val_start]
            end_idx = date_to_idx[val_end]
            train_end = start_idx - GAP_DAYS

            if train_end <= 150:
                continue

            # 1) 폴드 파라미터로 기준선 및 Feature 생성
            params = {
                "alpha_up": float(row["alpha_up"]),
                "alpha_down": float(row["alpha_down"]),
                "beta_up": float(row["beta_up"]),
                "beta_down": float(row["beta_down"]),
                "vol_period": int(round(row["vol_period"])),
                "volume_period": int(round(row["volume_period"])),
            }

            df_calc = df.iloc[: end_idx + 1].copy()
            bands = compute_bands_flexible(df_calc, **params)

            # 🔥 수정: 파라미터 컬럼을 bands에 명시적으로 추가
            for c in ["alpha_up", "alpha_down", "beta_up", "beta_down"]:
                bands[c] = params[c]

            features = build_features(df_calc, bands)
            y = make_labels(df_calc, threshold=threshold)

            # 2) CE 모델 학습 (지정된 Feature만 사용)
            try:
                model, scaler, stats = train_ce_model(
                    features=features,
                    y=y,
                    train_end=train_end,
                    feature_cols=use_features,
                )
            except Exception as e:
                # 디버깅을 위해 첫 번째 폴드에서만 에러 출력
                if fold_no == 0:
                    print(f"  ⚠️ Fold {fold_no+1} 학습 실패: {e}")
                continue

            # 3) OOS 예측
            oos_features = features.iloc[start_idx : end_idx + 1][use_features].copy()
            valid = np.isfinite(oos_features.to_numpy(dtype=float)).all(axis=1)
            if valid.sum() == 0:
                continue

            x_oos = oos_features.loc[valid].to_numpy(dtype=np.float32)
            x_oos_scaled = scaler.transform(x_oos).astype(np.float32)

            with torch.no_grad():
                logits = model(
                    torch.tensor(x_oos_scaled, dtype=torch.float32, device=DEVICE)
                )
                prob = torch.softmax(logits, dim=1).cpu().numpy()
            pred = prob.argmax(axis=1)

            # 실제 레이블
            y_oos = y.iloc[start_idx : end_idx + 1].to_numpy()
            y_true = y_oos[valid].astype(int)

            if len(np.unique(y_true)) < 2:
                continue

            fold_score = f1_score(y_true, pred, average="macro", zero_division=0)
            fold_scores.append(fold_score)

        if fold_scores:
            results[config] = {
                "scores": fold_scores,
                "mean": np.mean(fold_scores),
                "std": np.std(fold_scores),
                "median": np.median(fold_scores),
                "min": np.min(fold_scores),
                "max": np.max(fold_scores),
                "n_folds": len(fold_scores),
                "n_features": len(use_features),
            }
        else:
            results[config] = {
                "scores": [],
                "mean": np.nan,
                "std": np.nan,
                "median": np.nan,
                "min": np.nan,
                "max": np.nan,
                "n_folds": 0,
                "n_features": len(use_features),
            }

    return results


# ============================================================
# 3. 결과 출력 및 저장
# ============================================================
def print_ablation_results(results: Dict[str, Dict]):
    print("\n" + "=" * 70)
    print("📊 [2단계] Feature Ablation Test 결과 (CE Baseline)")
    print("=" * 70)
    print(
        f"{'Feature Set':<20} | {'Feat':>4} | {'Macro-F1':>8} | {'Std':>6} | {'Min':>6} | {'Max':>6} | {'Folds':>5}"
    )
    print("-" * 70)

    # ALL 먼저 출력, 나머지는 내림차순 정렬
    sorted_keys = sorted(
        results.keys(),
        key=lambda x: (
            x != "ALL",
            -results[x]["mean"] if not np.isnan(results[x]["mean"]) else -np.inf,
        ),
    )

    for key in sorted_keys:
        r = results[key]
        if r["n_folds"] == 0:
            print(
                f"{key:<20} | {r['n_features']:>4} | {'N/A':>8} | {'N/A':>6} | {'N/A':>6} | {'N/A':>6} | {0:>5}"
            )
        else:
            print(
                f"{key:<20} | {r['n_features']:>4} | {r['mean']:>8.4f} | {r['std']:>6.4f} | {r['min']:>6.4f} | {r['max']:>6.4f} | {r['n_folds']:>5}"
            )

    print("=" * 70)


# ============================================================
# 4. 메인 실행
# ============================================================
if __name__ == "__main__":
    print("🚀 2단계: Feature Ablation Test 시작")
    print(f"   - Device: {DEVICE}")
    print(f"   - Feature 그룹: {list(FEATURE_GROUPS.keys())}")
    print(f"   - 총 Feature 수: {len(ALL_FEATURES)}")

    # 데이터 로드
    df = load_data()
    print(f"   - 데이터: {len(df)}일")

    # 1단계에서 저장한 fold_details 로드 (우선 2% 기준)
    fold_path = "feature_validation_results/fold_details_2pct.csv"
    if not os.path.exists(fold_path):
        print(f"❌ {fold_path} 파일을 찾을 수 없습니다.")
        print("   1단계를 먼저 실행하여 fold_details를 저장해주세요.")
        exit(1)

    fold_details = pd.read_csv(fold_path)
    print(f"   - Fold 로드: {len(fold_details)}개")

    # Ablation 실행 (threshold=0.02)
    results = run_ablation_experiment(df, fold_details, threshold=0.02)

    # 결과 출력
    print_ablation_results(results)

    # CSV 저장
    output_dir = "ablation_results"
    os.makedirs(output_dir, exist_ok=True)
    save_df = pd.DataFrame(
        [
            {
                "feature_set": k,
                "n_features": v["n_features"],
                "mean_macro_f1": v["mean"],
                "std_macro_f1": v["std"],
                "median_macro_f1": v["median"],
                "min_macro_f1": v["min"],
                "max_macro_f1": v["max"],
                "n_folds": v["n_folds"],
            }
            for k, v in results.items()
        ]
    )
    save_path = os.path.join(output_dir, "ablation_results_2pct.csv")
    save_df.to_csv(save_path, index=False)
    print(f"\n💾 결과 저장: {os.path.abspath(save_path)}")

    print("\n✅ 2단계 Feature Ablation 완료!")
    print("   - 'ALL'이 기준선입니다.")
    print(
        "   - 'ALL - band'의 성능이 크게 떨어진다면 Band Feature가 중요함을 의미합니다."
    )
    print("   - 성능이 거의 같거나 오히려 올라간 그룹은 제거 대상이 될 수 있습니다.")
    print(
        "   - 이 결과를 바탕으로 최종 Feature set을 결정하고 3단계(CE vs Focal)로 넘어갑니다."
    )
