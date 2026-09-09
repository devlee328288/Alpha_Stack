import os
import warnings
from typing import Dict, List, Tuple

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
from focal_classifier import build_features, make_labels, FocalMLP, set_seed, FocalLoss

warnings.filterwarnings("ignore")

# ============================================================
# 0. 설정
# ============================================================
GAP_DAYS = 5
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
THRESHOLD = 0.02

# 4단계에서 선택된 최적 γ
BEST_GAMMA = 0.5

# 25개 Feature (3단계/4단계와 동일)
dummy_df = load_data().iloc[:100].copy()
dummy_bands = compute_bands_flexible(dummy_df)
dummy_features = build_features(dummy_df, dummy_bands)
ALL_FEATURES_FROM_BUILD = list(dummy_features.columns)

EXCLUDED_FEATURES = [
    "ret_1d",
    "ret_3d",
    "ret_5d",
    "ret_10d",
    "ret_20d",
    "alpha_up",
    "alpha_down",
    "beta_up",
    "beta_down",
]
FEATURES_TO_USE = [f for f in ALL_FEATURES_FROM_BUILD if f not in EXCLUDED_FEATURES]

# α 후보 리스트 (순서: [하락, 중립, 상승])
ALPHA_CANDIDATES = [
    [1.0, 1.0, 1.0],  # 기준
    [1.0, 1.0, 0.9],  # 상승 약화
    [1.0, 1.0, 0.8],  # 상승 더 약화
    [1.0, 1.0, 1.1],  # 상승 강화
    [0.9, 1.0, 1.0],  # 하락 약화
    [1.1, 1.0, 1.0],  # 하락 강화
    [1.0, 0.9, 1.0],  # 중립 약화
    [1.0, 1.1, 1.0],  # 중립 강화
]

print(f"사용 Feature 수: {len(FEATURES_TO_USE)}개")
print(f"고정 γ: {BEST_GAMMA}")
print(f"α 후보: {ALPHA_CANDIDATES}")


# ============================================================
# 1. Focal Loss 학습 함수 (γ/α 지정 가능)
# ============================================================
def train_focal_model(
    features: pd.DataFrame,
    y: pd.Series,
    train_end: int,
    feature_cols: List[str],
    gamma: float,
    alpha: List[float],
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
    set_seed(seed)

    label_safe_end = max(0, train_end - 5)
    train_idx = np.arange(0, label_safe_end)

    x_raw = features.iloc[train_idx][feature_cols].copy()
    y_raw = y.iloc[train_idx].to_numpy()

    valid_y = np.isfinite(y_raw)
    x_raw = x_raw.loc[valid_y]
    y_raw = y_raw[valid_y].astype(int)

    x_raw = x_raw.replace([np.inf, -np.inf], np.nan)
    valid_x = np.isfinite(x_raw.to_numpy(dtype=float)).all(axis=1)
    x_raw = x_raw.loc[valid_x]
    y_raw = y_raw[valid_x]

    if len(x_raw) < 150 or len(np.unique(y_raw)) < 3:
        raise ValueError("학습 데이터 부족 또는 클래스 누락")

    split = int(len(x_raw) * (1.0 - inner_val_ratio))
    split = min(max(split, 100), len(x_raw) - 30)
    x_tr, x_va = x_raw.iloc[:split], x_raw.iloc[split:]
    y_tr, y_va = y_raw[:split], y_raw[split:]

    scaler = StandardScaler()
    scaler.fit(x_tr.to_numpy(dtype=np.float32))
    xtr = scaler.transform(x_tr.to_numpy(dtype=np.float32)).astype(np.float32)
    xva = scaler.transform(x_va.to_numpy(dtype=np.float32)).astype(np.float32)

    model = FocalMLP(len(feature_cols), hidden1, hidden2, dropout).to(DEVICE)
    criterion = FocalLoss(gamma=gamma, alpha=alpha).to(DEVICE)
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

    for epoch in range(1, max_epochs + 1):
        model.train()
        for xb, yb in loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(xb), yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

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

    # 전체 재학습
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
# 2. α 튜닝 실행
# ============================================================
def run_alpha_tuning(
    df: pd.DataFrame,
    fold_details: pd.DataFrame,
    feature_cols: List[str],
    gamma: float,
    alpha_list: List[List[float]],
    threshold: float = 0.02,
) -> Dict[str, Dict]:
    date_to_idx = {d: i for i, d in enumerate(df.index)}
    results = {}

    for alpha in tqdm(alpha_list, desc="α 탐색"):
        alpha_key = f"[{alpha[0]:.1f}, {alpha[1]:.1f}, {alpha[2]:.1f}]"
        fold_scores = []
        fold_details_list = []

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
            features = build_features(df_calc, bands)
            y = make_labels(df_calc, threshold=threshold)

            # Focal 학습 (현재 alpha)
            try:
                model, scaler, stats = train_focal_model(
                    features, y, train_end, feature_cols, gamma=gamma, alpha=alpha
                )
            except Exception as e:
                print(f"  ⚠️ Fold {fold_no+1} α={alpha_key} 실패: {e}")
                continue

            # OOS 평가
            oos_features = features.iloc[start_idx : end_idx + 1][feature_cols].copy()
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

            y_oos = y.iloc[start_idx : end_idx + 1].to_numpy()
            y_true = y_oos[valid].astype(int)

            if len(np.unique(y_true)) < 2:
                continue

            f1 = f1_score(y_true, pred, average="macro", zero_division=0)
            fold_scores.append(f1)
            fold_details_list.append(
                {
                    "fold": fold_no + 1,
                    "f1_macro": f1,
                    "inner_f1": stats["inner_macro_f1"],
                    "epochs": stats["best_epoch"],
                }
            )

        if fold_scores:
            results[alpha_key] = {
                "alpha": alpha,
                "scores": fold_scores,
                "mean": np.mean(fold_scores),
                "std": np.std(fold_scores),
                "median": np.median(fold_scores),
                "min": np.min(fold_scores),
                "max": np.max(fold_scores),
                "n_folds": len(fold_scores),
                "details": pd.DataFrame(fold_details_list),
            }
        else:
            results[alpha_key] = {
                "alpha": alpha,
                "scores": [],
                "mean": np.nan,
                "std": np.nan,
                "median": np.nan,
                "min": np.nan,
                "max": np.nan,
                "n_folds": 0,
                "details": pd.DataFrame(),
            }

    return results


# ============================================================
# 3. 결과 출력 및 선택
# ============================================================
def print_alpha_results(results: Dict[str, Dict]):
    print("\n" + "=" * 70)
    print("📊 [5단계] Focal α 튜닝 결과")
    print(f"   Feature set: {len(FEATURES_TO_USE)}개 (ALL - momentum - parameter)")
    print(f"   고정 γ = {BEST_GAMMA}")
    print("=" * 70)

    # 테이블 헤더
    print(
        f"{'α (Down, Neut, Up)':<22} | {'Mean':>8} | {'Std':>8} | {'Median':>8} | {'Min':>8} | {'Max':>8} | {'Folds':>5}"
    )
    print("-" * 80)

    best_alpha_key = None
    best_mean = -np.inf

    for alpha_key in sorted(results.keys()):
        r = results[alpha_key]
        if r["n_folds"] == 0:
            print(
                f"{alpha_key:<22} | {'N/A':>8} | {'N/A':>8} | {'N/A':>8} | {'N/A':>8} | {'N/A':>8} | {0:>5}"
            )
        else:
            print(
                f"{alpha_key:<22} | {r['mean']:>8.4f} | {r['std']:>8.4f} | {r['median']:>8.4f} | {r['min']:>8.4f} | {r['max']:>8.4f} | {r['n_folds']:>5}"
            )
            if r["mean"] > best_mean:
                best_mean = r["mean"]
                best_alpha_key = alpha_key

    # 최적 α 상세
    if best_alpha_key is not None:
        print("\n" + "=" * 70)
        print(f"🏆 최적 α = {best_alpha_key} (평균 Macro-F1 = {best_mean:.4f})")
        print("=" * 70)

        best_details = results[best_alpha_key]["details"]
        if not best_details.empty:
            print("\n[최적 α 폴드별 Macro-F1]")
            print(best_details.to_string(index=False))

        # 통계 요약
        print("\n[α별 통계 요약]")
        summary = []
        for alpha_key, r in results.items():
            if r["n_folds"] > 0:
                summary.append(
                    {
                        "alpha": alpha_key,
                        "mean": r["mean"],
                        "std": r["std"],
                        "median": r["median"],
                        "min": r["min"],
                        "max": r["max"],
                        "n_folds": r["n_folds"],
                    }
                )
        summary_df = pd.DataFrame(summary)
        print(summary_df.to_string(index=False))

    return best_alpha_key


# ============================================================
# 4. 메인
# ============================================================
if __name__ == "__main__":
    print("🚀 5단계: Focal α 튜닝 시작")
    print(f"   - Device: {DEVICE}")
    print(f"   - Feature: {len(FEATURES_TO_USE)}개")
    print(f"   - γ 고정: {BEST_GAMMA}")
    print(f"   - α 후보: {len(ALPHA_CANDIDATES)}개")

    df = load_data()
    print(f"   - 데이터: {len(df)}일")

    fold_path = "feature_validation_results/fold_details_2pct.csv"
    if not os.path.exists(fold_path):
        print(f"❌ {fold_path} 파일을 찾을 수 없습니다.")
        exit(1)

    fold_details = pd.read_csv(fold_path)
    print(f"   - Fold 로드: {len(fold_details)}개")

    results = run_alpha_tuning(
        df,
        fold_details,
        FEATURES_TO_USE,
        gamma=BEST_GAMMA,
        alpha_list=ALPHA_CANDIDATES,
        threshold=THRESHOLD,
    )

    best_alpha_key = print_alpha_results(results)

    # 결과 저장
    os.makedirs("alpha_tuning_results", exist_ok=True)
    for alpha_key, r in results.items():
        if not r["details"].empty:
            safe_key = alpha_key.replace("[", "").replace("]", "").replace(", ", "_")
            r["details"].to_csv(
                f"alpha_tuning_results/alpha_{safe_key}_fold_details.csv", index=False
            )

    # 요약 저장
    summary_rows = []
    for alpha_key, r in results.items():
        summary_rows.append(
            {
                "alpha": alpha_key,
                "mean": r["mean"],
                "std": r["std"],
                "median": r["median"],
                "min": r["min"],
                "max": r["max"],
                "n_folds": r["n_folds"],
            }
        )
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv("alpha_tuning_results/alpha_summary.csv", index=False)

    print(f"\n💾 결과 저장: alpha_tuning_results/")
    print(f"   - 최적 α: {best_alpha_key}")
    print("\n✅ 5단계 완료! 다음 단계(Threshold 튜닝)로 진행하세요.")
