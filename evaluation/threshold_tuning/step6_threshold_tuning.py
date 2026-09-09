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
# 0. 설정 (5단계에서 확정된 값)
# ============================================================
GAP_DAYS = 5
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
THRESHOLD = 0.02
BEST_GAMMA = 0.5
BEST_ALPHA = [1.0, 1.0, 0.9]  # [down, neutral, up]

# 25개 Feature
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

# Threshold 후보 (하락/상승 각각)
THRESHOLD_CANDIDATES = [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60]

print(f"사용 Feature 수: {len(FEATURES_TO_USE)}개")
print(f"γ = {BEST_GAMMA}, α = {BEST_ALPHA}")
print(f"Threshold 후보: {THRESHOLD_CANDIDATES}")


# ============================================================
# 1. Focal 모델 학습 함수 (γ/α 고정)
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
# 2. Threshold 적용 함수
# ============================================================
def apply_threshold(proba: np.ndarray, t_down: float, t_up: float) -> np.ndarray:
    """
    p_down >= t_down and p_down > p_up → 0 (하락)
    p_up >= t_up and p_up > p_down → 2 (상승)
    그 외 → 1 (중립)
    """
    pred = np.full(len(proba), 1, dtype=int)  # 기본 중립
    p_down = proba[:, 0]
    p_neutral = proba[:, 1]
    p_up = proba[:, 2]

    # 하락 조건
    down_mask = (p_down >= t_down) & (p_down > p_up)
    pred[down_mask] = 0

    # 상승 조건 (하락이 아닌 경우만)
    up_mask = (p_up >= t_up) & (p_up > p_down) & (~down_mask)
    pred[up_mask] = 2

    return pred


def find_best_threshold(
    proba: np.ndarray,
    y_true: np.ndarray,
    t_candidates: List[float],
) -> Tuple[float, float, float]:
    """내부 검증 데이터에서 최적 threshold 찾기 (Macro-F1 기준)"""
    best_score = -np.inf
    best_t_down = 0.30
    best_t_up = 0.30

    for t_down in t_candidates:
        for t_up in t_candidates:
            pred = apply_threshold(proba, t_down, t_up)
            score = f1_score(y_true, pred, average="macro", zero_division=0)
            if score > best_score:
                best_score = score
                best_t_down = t_down
                best_t_up = t_up

    return best_t_down, best_t_up, best_score


# ============================================================
# 3. Threshold 튜닝 실행 (Inner Validation 방식)
# ============================================================
def run_threshold_tuning(
    df: pd.DataFrame,
    fold_details: pd.DataFrame,
    feature_cols: List[str],
    gamma: float,
    alpha: List[float],
    t_candidates: List[float],
    threshold: float = 0.02,
) -> Dict:
    date_to_idx = {d: i for i, d in enumerate(df.index)}

    fold_results = []
    all_t_down = []
    all_t_up = []
    all_inner_scores = []

    for fold_no, row in tqdm(
        fold_details.iterrows(), total=len(fold_details), desc="Threshold 튜닝"
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

        # 1) 모델 학습
        try:
            model, scaler, stats = train_focal_model(
                features, y, train_end, feature_cols, gamma=gamma, alpha=alpha
            )
        except Exception as e:
            print(f"  ⚠️ Fold {fold_no+1} 학습 실패: {e}")
            continue

        # 2) 내부 검증용 데이터 준비 (Train의 마지막 20%)
        label_safe_end = max(0, train_end - 5)
        train_idx = np.arange(0, label_safe_end)
        x_all = features.iloc[train_idx][feature_cols].copy()
        y_all = y.iloc[train_idx].to_numpy()

        valid_y = np.isfinite(y_all)
        x_all = x_all.loc[valid_y]
        y_all = y_all[valid_y].astype(int)
        x_all = x_all.replace([np.inf, -np.inf], np.nan)
        valid_x = np.isfinite(x_all.to_numpy(dtype=float)).all(axis=1)
        x_all = x_all.loc[valid_x]
        y_all = y_all[valid_x]

        if len(x_all) < 150:
            continue

        # Train의 마지막 20%를 내부 검증으로 사용
        split = int(len(x_all) * 0.8)
        x_tr, x_val = x_all.iloc[:split], x_all.iloc[split:]
        y_tr, y_val = y_all[:split], y_all[split:]

        # Scaling (Train으로 fit)
        scaler_inner = StandardScaler()
        scaler_inner.fit(x_tr.to_numpy(dtype=np.float32))
        x_val_scaled = scaler_inner.transform(x_val.to_numpy(dtype=np.float32)).astype(
            np.float32
        )

        # 내부 검증 예측
        with torch.no_grad():
            logits = model(
                torch.tensor(x_val_scaled, dtype=torch.float32, device=DEVICE)
            )
            proba = torch.softmax(logits, dim=1).cpu().numpy()

        # 3) 내부 검증에서 최적 threshold 찾기
        t_down, t_up, inner_score = find_best_threshold(proba, y_val, t_candidates)
        all_t_down.append(t_down)
        all_t_up.append(t_up)
        all_inner_scores.append(inner_score)

        # 4) OOS 평가 (찾은 threshold 적용)
        oos_features = features.iloc[start_idx : end_idx + 1][feature_cols].copy()
        valid = np.isfinite(oos_features.to_numpy(dtype=float)).all(axis=1)
        if valid.sum() == 0:
            continue

        x_oos = oos_features.loc[valid].to_numpy(dtype=np.float32)
        x_oos_scaled = scaler.transform(x_oos).astype(np.float32)

        with torch.no_grad():
            logits_oos = model(
                torch.tensor(x_oos_scaled, dtype=torch.float32, device=DEVICE)
            )
            proba_oos = torch.softmax(logits_oos, dim=1).cpu().numpy()

        pred_oos = apply_threshold(proba_oos, t_down, t_up)

        y_oos = y.iloc[start_idx : end_idx + 1].to_numpy()
        y_true = y_oos[valid].astype(int)

        if len(np.unique(y_true)) < 2:
            continue

        f1 = f1_score(y_true, pred_oos, average="macro", zero_division=0)

        fold_results.append(
            {
                "fold": fold_no + 1,
                "t_down": t_down,
                "t_up": t_up,
                "inner_score": inner_score,
                "oos_f1": f1,
                "inner_f1": stats["inner_macro_f1"],
                "epochs": stats["best_epoch"],
            }
        )

    return {
        "fold_results": pd.DataFrame(fold_results),
        "all_t_down": all_t_down,
        "all_t_up": all_t_up,
        "all_inner_scores": all_inner_scores,
    }


# ============================================================
# 4. 결과 출력
# ============================================================
def print_threshold_results(results: Dict):
    fold_df = results["fold_results"]

    print("\n" + "=" * 70)
    print("📊 [6단계] Threshold 튜닝 결과")
    print(f"   Feature set: {len(FEATURES_TO_USE)}개")
    print(f"   γ = {BEST_GAMMA}, α = {BEST_ALPHA}")
    print(f"   Threshold 후보: {THRESHOLD_CANDIDATES}")
    print("=" * 70)

    if fold_df.empty:
        print("❌ 유효한 폴드가 없습니다.")
        return

    # 통계
    mean_f1 = fold_df["oos_f1"].mean()
    std_f1 = fold_df["oos_f1"].std()
    median_f1 = fold_df["oos_f1"].median()
    min_f1 = fold_df["oos_f1"].min()
    max_f1 = fold_df["oos_f1"].max()

    print(f"\n[OOS Macro-F1 통계]")
    print(f"  Mean  : {mean_f1:.4f}")
    print(f"  Std   : {std_f1:.4f}")
    print(f"  Median: {median_f1:.4f}")
    print(f"  Min   : {min_f1:.4f}")
    print(f"  Max   : {max_f1:.4f}")

    # Threshold 분포
    print(f"\n[Threshold 분포]")
    print(
        f"  T_down 평균: {fold_df['t_down'].mean():.3f} (중앙: {fold_df['t_down'].median():.3f})"
    )
    print(
        f"  T_up 평균  : {fold_df['t_up'].mean():.3f} (중앙: {fold_df['t_up'].median():.3f})"
    )

    # 폴드별 상세
    print("\n[폴드별 상세]")
    print(fold_df.to_string(index=False))

    # 최종 선택 (중앙값 사용)
    final_t_down = fold_df["t_down"].median()
    final_t_up = fold_df["t_up"].median()
    print("\n" + "=" * 70)
    print(
        f"🏆 최종 Threshold (중앙값) : T_down = {final_t_down:.2f}, T_up = {final_t_up:.2f}"
    )
    print(f"   OOS 평균 Macro-F1 = {mean_f1:.4f}")
    print("=" * 70)

    return final_t_down, final_t_up, mean_f1


# ============================================================
# 5. 메인
# ============================================================
if __name__ == "__main__":
    print("🚀 6단계: Threshold 튜닝 시작")
    print(f"   - Device: {DEVICE}")
    print(f"   - Feature: {len(FEATURES_TO_USE)}개")
    print(f"   - γ = {BEST_GAMMA}, α = {BEST_ALPHA}")

    df = load_data()
    print(f"   - 데이터: {len(df)}일")

    fold_path = "feature_validation_results/fold_details_2pct.csv"
    if not os.path.exists(fold_path):
        print(f"❌ {fold_path} 파일을 찾을 수 없습니다.")
        exit(1)

    fold_details = pd.read_csv(fold_path)
    print(f"   - Fold 로드: {len(fold_details)}개")

    results = run_threshold_tuning(
        df,
        fold_details,
        FEATURES_TO_USE,
        gamma=BEST_GAMMA,
        alpha=BEST_ALPHA,
        t_candidates=THRESHOLD_CANDIDATES,
        threshold=0.02,
    )

    final_t_down, final_t_up, mean_f1 = print_threshold_results(results)

    # 결과 저장
    os.makedirs("threshold_tuning_results", exist_ok=True)
    results["fold_results"].to_csv(
        "threshold_tuning_results/threshold_fold_results.csv", index=False
    )

    # 최종 설정 저장
    final_config = {
        "label_threshold": 0.02,
        "feature_count": len(FEATURES_TO_USE),
        "features": FEATURES_TO_USE,
        "gamma": BEST_GAMMA,
        "alpha_down": BEST_ALPHA[0],
        "alpha_neutral": BEST_ALPHA[1],
        "alpha_up": BEST_ALPHA[2],
        "t_down": final_t_down,
        "t_up": final_t_up,
        "oos_mean_macro_f1": mean_f1,
        "oos_std_macro_f1": results["fold_results"]["oos_f1"].std(),
    }
    pd.DataFrame([final_config]).to_csv(
        "threshold_tuning_results/final_config.csv", index=False
    )

    print(f"\n💾 결과 저장: threshold_tuning_results/")
    print("\n✅ 6단계 완료! 최종 설정이 확정되었습니다.")
    print("   → 7단계: 최종 OOS 평가 (Fold 11 Holdout)을 실행하세요.")
