import os
import warnings
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from focal_classifier import FocalLoss, FocalMLP, build_features, make_labels, set_seed
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
)
from sklearn.preprocessing import StandardScaler
from step1_core_features import load_data
from step5_optimize_6params import compute_bands_flexible
from torch.utils.data import DataLoader, TensorDataset

warnings.filterwarnings("ignore")

# ============================================================
# 0. 최종 설정 (6단계에서 확정)
# ============================================================
GAP_DAYS = 5
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LABEL_THRESHOLD = 0.02

BEST_GAMMA = 0.5
BEST_ALPHA = [1.0, 1.0, 0.9]  # [down, neutral, up]
T_DOWN = 0.2
T_UP = 0.2

# 25개 Feature (ALL - momentum - parameter)
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

print("=" * 70)
print("🚀 7단계: 최종 Holdout 평가 (Fold 11)")
print("=" * 70)
print(f"γ = {BEST_GAMMA}, α = {BEST_ALPHA}")
print(f"Threshold: T_down = {T_DOWN}, T_up = {T_UP}")
print(f"Feature: {len(FEATURES_TO_USE)}개")


# ============================================================
# 1. Focal 모델 학습 함수 (최종 설정 고정)
# ============================================================
def train_final_model(
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

    # Label leakage 방지 (train_end - 5)
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

    # 내부 Train/Validation 분할 (Early Stopping용)
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
    _yva_t = torch.tensor(y_va, dtype=torch.long, device=DEVICE)

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

    # 전체 Train 데이터로 재학습 (동일 epoch)
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
    pred = np.full(len(proba), 1, dtype=int)
    p_down = proba[:, 0]
    p_up = proba[:, 2]

    down_mask = (p_down >= t_down) & (p_down > p_up)
    pred[down_mask] = 0

    up_mask = (p_up >= t_up) & (p_up > p_down) & (~down_mask)
    pred[up_mask] = 2

    return pred


# ============================================================
# 3. 평가 지표 계산
# ============================================================
def calculate_all_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, proba: np.ndarray
) -> Dict:
    metrics = {}

    # 기본 분류 지표
    metrics["accuracy"] = accuracy_score(y_true, y_pred)
    metrics["balanced_accuracy"] = balanced_accuracy_score(y_true, y_pred)
    metrics["f1_macro"] = f1_score(y_true, y_pred, average="macro", zero_division=0)
    metrics["f1_weighted"] = f1_score(
        y_true, y_pred, average="weighted", zero_division=0
    )
    metrics["mcc"] = matthews_corrcoef(y_true, y_pred)

    # 클래스별 F1
    f1_per_class = f1_score(y_true, y_pred, average=None, zero_division=0)
    metrics["f1_down"] = f1_per_class[0]
    metrics["f1_neutral"] = f1_per_class[1]
    metrics["f1_up"] = f1_per_class[2]

    # 혼동 행렬
    metrics["confusion_matrix"] = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])

    # 분류 리포트
    metrics["report"] = classification_report(
        y_true,
        y_pred,
        labels=[0, 1, 2],
        target_names=["하락", "중립", "상승"],
        digits=4,
        zero_division=0,
    )

    # PR-AUC (확률이 있을 경우)
    if proba is not None and np.isfinite(proba).all():
        y_onehot = np.eye(3)[y_true]
        metrics["pr_auc_macro"] = average_precision_score(
            y_onehot, proba, average="macro"
        )
        metrics["pr_auc_down"] = average_precision_score(y_onehot[:, 0], proba[:, 0])
        metrics["pr_auc_neutral"] = average_precision_score(y_onehot[:, 1], proba[:, 1])
        metrics["pr_auc_up"] = average_precision_score(y_onehot[:, 2], proba[:, 2])

    # 예측 클래스 비율
    unique, counts = np.unique(y_pred, return_counts=True)
    ratio_dict = dict(zip(unique, counts / len(y_pred), strict=True))
    metrics["pred_ratio_down"] = ratio_dict.get(0, 0.0)
    metrics["pred_ratio_neutral"] = ratio_dict.get(1, 0.0)
    metrics["pred_ratio_up"] = ratio_dict.get(2, 0.0)

    return metrics


# ============================================================
# 4. 메인 실행
# ============================================================
if __name__ == "__main__":
    # 데이터 로드
    df = load_data()
    print(f"📊 데이터: {len(df)}일")

    # Fold 11 정보 로드
    fold_path = "feature_validation_results/fold_details_2pct.csv"
    if not os.path.exists(fold_path):
        print(f"❌ {fold_path} 파일을 찾을 수 없습니다.")
        exit(1)

    fold_details = pd.read_csv(fold_path)

    # Fold 11만 사용 (마지막 Fold)
    fold_11 = fold_details.iloc[-1]
    print("\n📌 Fold 11 (최종 Holdout)")
    print(f"   Train End : {fold_11['train_end']}")
    print(f"   OOS Start : {fold_11['val_start']}")
    print(f"   OOS End   : {fold_11['val_end']}")

    # 날짜 → 인덱스 매핑
    date_to_idx = {d: i for i, d in enumerate(df.index)}
    val_start = pd.Timestamp(fold_11["val_start"])
    val_end = pd.Timestamp(fold_11["val_end"])

    if val_start not in date_to_idx or val_end not in date_to_idx:
        print("❌ OOS 날짜를 데이터에서 찾을 수 없습니다.")
        exit(1)

    start_idx = date_to_idx[val_start]
    end_idx = date_to_idx[val_end]
    train_end = start_idx - GAP_DAYS

    print(f"   Train End Index: {train_end}")
    print(f"   OOS Index      : {start_idx} ~ {end_idx} ({end_idx - start_idx + 1}일)")

    # Fold 11 파라미터
    params = {
        "alpha_up": float(fold_11["alpha_up"]),
        "alpha_down": float(fold_11["alpha_down"]),
        "beta_up": float(fold_11["beta_up"]),
        "beta_down": float(fold_11["beta_down"]),
        "vol_period": int(round(fold_11["vol_period"])),
        "volume_period": int(round(fold_11["volume_period"])),
    }

    # Feature 생성
    print("\n🔧 Feature 생성 중...")
    df_calc = df.iloc[: end_idx + 1].copy()
    bands = compute_bands_flexible(df_calc, **params)
    features = build_features(df_calc, bands)
    y = make_labels(df_calc, threshold=LABEL_THRESHOLD)

    # 모델 학습
    print("🧠 모델 학습 중...")
    model, scaler, stats = train_final_model(
        features, y, train_end, FEATURES_TO_USE, gamma=BEST_GAMMA, alpha=BEST_ALPHA
    )
    print(f"   내부 검증 Macro-F1: {stats['inner_macro_f1']:.4f}")
    print(f"   최적 Epoch: {int(stats['best_epoch'])}")

    # OOS 예측
    print("🔮 OOS 예측 중...")
    oos_features = features.iloc[start_idx : end_idx + 1][FEATURES_TO_USE].copy()
    y_oos = y.iloc[start_idx : end_idx + 1].to_numpy()

    # Feature와 Label 모두 유효한 행만 선택 (🔥 수정)
    valid_feat = np.isfinite(oos_features.to_numpy(dtype=float)).all(axis=1)
    valid_label = np.isfinite(y_oos)
    valid = valid_feat & valid_label

    if valid.sum() == 0:
        print("❌ OOS에 유효한 Feature/Label이 없습니다.")
        exit(1)

    x_oos = oos_features.loc[valid].to_numpy(dtype=np.float32)
    x_oos_scaled = scaler.transform(x_oos).astype(np.float32)

    with torch.no_grad():
        logits = model(torch.tensor(x_oos_scaled, dtype=torch.float32, device=DEVICE))
        proba = torch.softmax(logits, dim=1).cpu().numpy()

    # Threshold 적용
    pred = apply_threshold(proba, T_DOWN, T_UP)

    # 실제 레이블 (유효한 행만)
    y_true = y_oos[valid].astype(int)

    # 평가
    print("\n" + "=" * 70)
    print("📊 [7단계] 최종 Holdout 평가 결과 (Fold 11)")
    print("=" * 70)

    metrics = calculate_all_metrics(y_true, pred, proba)

    # 주요 지표 출력
    print("\n[분류 지표]")
    print(f"  Accuracy         : {metrics['accuracy']:.4f}")
    print(f"  Balanced Accuracy: {metrics['balanced_accuracy']:.4f}")
    print(f"  Macro-F1         : {metrics['f1_macro']:.4f}")
    print(f"  Weighted-F1      : {metrics['f1_weighted']:.4f}")
    print(f"  MCC              : {metrics['mcc']:.4f}")

    print("\n[클래스별 F1]")
    print(f"  하락  : {metrics['f1_down']:.4f}")
    print(f"  중립  : {metrics['f1_neutral']:.4f}")
    print(f"  상승  : {metrics['f1_up']:.4f}")

    print("\n[PR-AUC]")
    print(f"  Macro : {metrics.get('pr_auc_macro', np.nan):.4f}")
    print(f"  하락  : {metrics.get('pr_auc_down', np.nan):.4f}")
    print(f"  중립  : {metrics.get('pr_auc_neutral', np.nan):.4f}")
    print(f"  상승  : {metrics.get('pr_auc_up', np.nan):.4f}")

    print("\n[예측 클래스 비율]")
    print(f"  하락  : {metrics['pred_ratio_down']:.2%}")
    print(f"  중립  : {metrics['pred_ratio_neutral']:.2%}")
    print(f"  상승  : {metrics['pred_ratio_up']:.2%}")

    print("\n[혼동 행렬]")
    print("         Pred Down  Pred Neut  Pred Up")
    print(
        f"True Down  {metrics['confusion_matrix'][0][0]:>6}  "
        f"{metrics['confusion_matrix'][0][1]:>9}  {metrics['confusion_matrix'][0][2]:>7}"
    )
    print(
        f"True Neut  {metrics['confusion_matrix'][1][0]:>6}  "
        f"{metrics['confusion_matrix'][1][1]:>9}  {metrics['confusion_matrix'][1][2]:>7}"
    )
    print(
        f"True Up    {metrics['confusion_matrix'][2][0]:>6}  "
        f"{metrics['confusion_matrix'][2][1]:>9}  {metrics['confusion_matrix'][2][2]:>7}"
    )

    print("\n[분류 리포트]")
    print(metrics["report"])

    # 결과 저장
    os.makedirs("final_holdout_results", exist_ok=True)

    # 메트릭 저장
    metrics_flat = {
        k: v for k, v in metrics.items() if not isinstance(v, (np.ndarray, str))
    }
    metrics_flat["inner_f1"] = stats["inner_macro_f1"]
    metrics_flat["best_epoch"] = stats["best_epoch"]
    metrics_flat["train_rows"] = stats["train_rows"]
    metrics_flat["oos_rows"] = len(y_true)
    metrics_flat["gamma"] = BEST_GAMMA
    metrics_flat["alpha_down"] = BEST_ALPHA[0]
    metrics_flat["alpha_neutral"] = BEST_ALPHA[1]
    metrics_flat["alpha_up"] = BEST_ALPHA[2]
    metrics_flat["t_down"] = T_DOWN
    metrics_flat["t_up"] = T_UP

    pd.DataFrame([metrics_flat]).to_csv(
        "final_holdout_results/final_metrics.csv", index=False
    )

    # 혼동 행렬 저장
    cm_df = pd.DataFrame(
        metrics["confusion_matrix"],
        index=["True Down", "True Neut", "True Up"],
        columns=["Pred Down", "Pred Neut", "Pred Up"],
    )
    cm_df.to_csv("final_holdout_results/confusion_matrix.csv", index=True)

    # 분류 리포트 저장
    with open("final_holdout_results/classification_report.txt", "w") as f:
        f.write(metrics["report"])

    print("\n💾 결과 저장: final_holdout_results/")
    print("\n✅ 7단계 완료! 전체 파이프라인 종료.")
