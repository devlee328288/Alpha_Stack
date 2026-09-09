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
# 0. 설정 - FEATURES_TO_USE 동적 생성 (모멘텀 + 파라미터 제거)
# ============================================================
GAP_DAYS = 5
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
THRESHOLD = 0.02

# 더미 데이터로 build_features()가 생성하는 모든 Feature 목록 확인
dummy_df = load_data().iloc[:100].copy()
dummy_bands = compute_bands_flexible(dummy_df)
dummy_features = build_features(dummy_df, dummy_bands)
ALL_FEATURES_FROM_BUILD = list(dummy_features.columns)

# 제거할 Feature: 모멘텀(5개) + 파라미터(4개)
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

print(f"build_features() 총 Feature 수: {len(ALL_FEATURES_FROM_BUILD)}개")
print(f"제거할 Feature: {EXCLUDED_FEATURES}")
print(f"최종 사용 Feature 수: {len(FEATURES_TO_USE)}개")


# ============================================================
# 1. 공통 학습 함수 (Loss만 다름)
# ============================================================
def train_model(
    features: pd.DataFrame,
    y: pd.Series,
    train_end: int,
    feature_cols: List[str],
    loss_fn: nn.Module,
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
            loss = loss_fn(model(xb), yb)
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
            loss = loss_fn(model_full(xb), yb)
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
# 2. CE vs Focal 비교 실행
# ============================================================
def run_ce_vs_focal(
    df: pd.DataFrame,
    fold_details: pd.DataFrame,
    feature_cols: List[str],
    threshold: float = 0.02,
) -> Dict[str, Dict]:
    date_to_idx = {d: i for i, d in enumerate(df.index)}
    results = {
        "CE": {"scores": [], "details": []},
        "Focal": {"scores": [], "details": []},
    }

    ce_loss = nn.CrossEntropyLoss()
    focal_loss = FocalLoss(gamma=2.0, alpha=[1.0, 1.0, 1.0])

    for fold_no, row in tqdm(
        fold_details.iterrows(), total=len(fold_details), desc="Fold 진행"
    ):
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

        # CE 학습
        try:
            model_ce, scaler_ce, stats_ce = train_model(
                features, y, train_end, feature_cols, ce_loss
            )
        except Exception as e:
            print(f"  ⚠️ Fold {fold_no+1} CE 실패: {e}")
            continue

        # Focal 학습
        try:
            model_focal, scaler_focal, stats_focal = train_model(
                features, y, train_end, feature_cols, focal_loss
            )
        except Exception as e:
            print(f"  ⚠️ Fold {fold_no+1} Focal 실패: {e}")
            continue

        # OOS 평가
        oos_features = features.iloc[start_idx : end_idx + 1][feature_cols].copy()
        valid = np.isfinite(oos_features.to_numpy(dtype=float)).all(axis=1)
        if valid.sum() == 0:
            continue

        x_oos = oos_features.loc[valid].to_numpy(dtype=np.float32)
        y_oos = y.iloc[start_idx : end_idx + 1].to_numpy()
        y_true = y_oos[valid].astype(int)

        # CE
        x_ce = scaler_ce.transform(x_oos).astype(np.float32)
        with torch.no_grad():
            logits_ce = model_ce(torch.tensor(x_ce, dtype=torch.float32, device=DEVICE))
            prob_ce = torch.softmax(logits_ce, dim=1).cpu().numpy()
        pred_ce = prob_ce.argmax(axis=1)
        f1_ce = f1_score(y_true, pred_ce, average="macro", zero_division=0)

        # Focal
        x_focal = scaler_focal.transform(x_oos).astype(np.float32)
        with torch.no_grad():
            logits_focal = model_focal(
                torch.tensor(x_focal, dtype=torch.float32, device=DEVICE)
            )
            prob_focal = torch.softmax(logits_focal, dim=1).cpu().numpy()
        pred_focal = prob_focal.argmax(axis=1)
        f1_focal = f1_score(y_true, pred_focal, average="macro", zero_division=0)

        results["CE"]["scores"].append(f1_ce)
        results["CE"]["details"].append(
            {
                "fold": fold_no + 1,
                "f1_macro": f1_ce,
                "inner_f1": stats_ce["inner_macro_f1"],
                "epochs": stats_ce["best_epoch"],
            }
        )
        results["Focal"]["scores"].append(f1_focal)
        results["Focal"]["details"].append(
            {
                "fold": fold_no + 1,
                "f1_macro": f1_focal,
                "inner_f1": stats_focal["inner_macro_f1"],
                "epochs": stats_focal["best_epoch"],
            }
        )

    return results


# ============================================================
# 3. 결과 출력
# ============================================================
def print_comparison(results: Dict[str, Dict]):
    ce_scores = results["CE"]["scores"]
    focal_scores = results["Focal"]["scores"]

    print("\n" + "=" * 70)
    print("📊 [3단계] CE vs Focal Loss 비교 결과")
    print(f"   Feature set: {len(FEATURES_TO_USE)}개 (ALL - momentum - parameter)")
    print("=" * 70)

    if not ce_scores or not focal_scores:
        print("❌ 유효한 폴드가 없습니다.")
        return

    ce_arr = np.array(ce_scores)
    focal_arr = np.array(focal_scores)

    print(f"{'Metric':<15} | {'CE':>10} | {'Focal':>10} | {'차이 (F-C)':>12}")
    print("-" * 70)
    print(
        f"{'Mean':<15} | {ce_arr.mean():>10.4f} | {focal_arr.mean():>10.4f} | {focal_arr.mean() - ce_arr.mean():>+12.4f}"
    )
    print(
        f"{'Std':<15} | {ce_arr.std():>10.4f} | {focal_arr.std():>10.4f} | {focal_arr.std() - ce_arr.std():>+12.4f}"
    )
    print(
        f"{'Median':<15} | {np.median(ce_arr):>10.4f} | {np.median(focal_arr):>10.4f} | {np.median(focal_arr) - np.median(ce_arr):>+12.4f}"
    )
    print(
        f"{'Min':<15} | {ce_arr.min():>10.4f} | {focal_arr.min():>10.4f} | {focal_arr.min() - ce_arr.min():>+12.4f}"
    )
    print(
        f"{'Max':<15} | {ce_arr.max():>10.4f} | {focal_arr.max():>10.4f} | {focal_arr.max() - ce_arr.max():>+12.4f}"
    )
    print(f"{'Folds':<15} | {len(ce_scores):>10} | {len(focal_scores):>10} | {'':>12}")

    print("\n[폴드별 Macro-F1]")
    print(f"{'Fold':>6} | {'CE':>10} | {'Focal':>10} | {'차이':>10}")
    print("-" * 40)
    for i in range(len(ce_scores)):
        print(
            f"{i+1:>6} | {ce_scores[i]:>10.4f} | {focal_scores[i]:>10.4f} | {focal_scores[i] - ce_scores[i]:>+10.4f}"
        )

    if focal_arr.mean() > ce_arr.mean():
        print("\n✅ Focal Loss가 CE보다 평균 Macro-F1이 높습니다.")
        print("   → Focal 튜닝(γ, α)으로 진행할 가치가 있습니다.")
    else:
        print("\n❌ CE가 Focal보다 우세합니다.")
        print("   → Focal Loss가 답이 아닐 수 있습니다. CE 개선에 집중하세요.")


# ============================================================
# 4. 메인
# ============================================================
if __name__ == "__main__":
    print("🚀 3단계: CE vs Focal Loss 비교 시작")
    print(f"   - Device: {DEVICE}")
    print(f"   - Feature: {len(FEATURES_TO_USE)}개 (ALL - momentum - parameter)")

    df = load_data()
    print(f"   - 데이터: {len(df)}일")

    fold_path = "feature_validation_results/fold_details_2pct.csv"
    if not os.path.exists(fold_path):
        print(f"❌ {fold_path} 파일을 찾을 수 없습니다.")
        exit(1)

    fold_details = pd.read_csv(fold_path)
    print(f"   - Fold 로드: {len(fold_details)}개")

    results = run_ce_vs_focal(df, fold_details, FEATURES_TO_USE, threshold=THRESHOLD)

    print_comparison(results)

    os.makedirs("ce_vs_focal_results", exist_ok=True)
    ce_df = pd.DataFrame(results["CE"]["details"])
    focal_df = pd.DataFrame(results["Focal"]["details"])
    ce_df.to_csv("ce_vs_focal_results/ce_fold_results.csv", index=False)
    focal_df.to_csv("ce_vs_focal_results/focal_fold_results.csv", index=False)
    print(f"\n💾 결과 저장: ce_vs_focal_results/")

    print("\n✅ 3단계 완료!")
