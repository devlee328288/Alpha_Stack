import random
import warnings
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

warnings.filterwarnings("ignore")

LABEL_DOWN = 0
LABEL_NEUTRAL = 1
LABEL_UP = 2

# ADR-AS-0002 반영: 라벨 축이 T+1→T+6 로 바뀌었으므로 feature 버전 상향
FEATURE_VERSION = "FocalML-v2-ADR-AS-0002"

# ADR-AS-0002: T 종가 신호 → T+1 시가 체결 → T+6 시가 평가 (horizon = 6 거래일)
HORIZON = 6


@dataclass
class FocalConfig:
    gamma: float = 2.0
    alpha_down: float = 1.0
    alpha_neutral: float = 1.0
    alpha_up: float = 1.20
    hidden1: int = 64
    hidden2: int = 32
    dropout: float = 0.15
    lr: float = 1e-3
    weight_decay: float = 1e-4
    batch_size: int = 64
    max_epochs: int = 180
    patience: int = 20
    inner_val_ratio: float = 0.20
    seed: int = 42
    # ADR-AS-0002: 학습/검증/OOS 라벨 임계값. Step 5 와 동일한 값을 써야 한다.
    # 0.01 / 0.02 두 실험을 각각 FocalConfig(threshold=...) 로 실행.
    threshold: float = 0.01


class FocalLoss(nn.Module):
    """Multi-class focal loss using logits.

    FL = alpha_y * (1-p_y)^gamma * CE
    """

    def __init__(self, gamma: float = 2.0, alpha: Optional[List[float]] = None):
        super().__init__()
        self.gamma = float(gamma)
        if alpha is None:
            alpha = [1.0, 1.0, 1.0]
        self.register_buffer("alpha", torch.tensor(alpha, dtype=torch.float32))

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce = nn.functional.cross_entropy(logits, targets, reduction="none")
        pt = torch.exp(-ce).clamp(min=1e-8, max=1.0)
        alpha_t = self.alpha[targets]
        return (alpha_t * (1.0 - pt).pow(self.gamma) * ce).mean()


class FocalMLP(nn.Module):
    def __init__(self, n_features: int, hidden1: int, hidden2: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, hidden1),
            nn.LayerNorm(hidden1),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden1, hidden2),
            nn.LayerNorm(hidden2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden2, 3),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================
# ADR-AS-0002 공용 라벨 함수
# ------------------------------------------------------------
# - T일 종가까지의 정보로 신호 생성
# - T+1일 시가 체결
# - T+6일 시가 평가
# - fwd_ret = open.shift(-6) / open.shift(-1) - 1
# - 마지막 6행은 NaN 유지 (중립으로 변환 금지)
# - 학습/검증/OOS 모두 동일 threshold 사용
#
# Step 5, Step 7, Rolling Feature 검증 파이프라인 모두 이 함수만 사용해야
# 한다. 어떤 경로도 직접 라벨을 계산하지 않는다.
# ============================================================
def make_labels(df: pd.DataFrame, threshold: float = 0.01) -> pd.Series:
    """ADR-AS-0002 라벨: Open(T+1) → Open(T+6) 수익률 기반 3-class.

    Parameters
    ----------
    df : DataFrame
        'open' 컬럼 필요. 'code' 컬럼이 있으면 종목별로 계산.
    threshold : float
        상승/하락 판정 임계값. |fwd_ret| <= threshold 는 중립.

    Returns
    -------
    pd.Series (float) : {0.0, 1.0, 2.0} 또는 NaN.
        미래값이 없는 마지막 6행은 NaN 으로 유지된다(중립으로 변환 금지).
    """
    if "code" in df.columns:
        future_ret = df.groupby("code")["open"].transform(
            lambda x: x.shift(-HORIZON) / x.shift(-1) - 1.0
        )
    else:
        future_ret = df["open"].shift(-HORIZON) / df["open"].shift(-1) - 1.0

    y = np.full(len(df), np.nan)
    y[future_ret > threshold] = LABEL_UP
    y[future_ret < -threshold] = LABEL_DOWN
    y[(future_ret >= -threshold) & (future_ret <= threshold)] = LABEL_NEUTRAL

    # 미래값이 없는 마지막 HORIZON 행은 NaN 유지
    y[future_ret.isna().to_numpy()] = np.nan

    return pd.Series(y, index=df.index, name="target")


def make_labels_for_config(df: pd.DataFrame, cfg: FocalConfig) -> pd.Series:
    """FocalConfig.threshold 를 사용하는 편의 래퍼."""
    return make_labels(df, threshold=cfg.threshold)


def _regression_test_make_labels() -> None:
    """make_labels() 축·끝행·분포 회귀 테스트.

    ADR-AS-0002 완료 조건:
      - 시간축: fwd_ret = open.shift(-6)/open.shift(-1) - 1
      - 마지막 6행 NaN 유지 (중립으로 변환 금지)
      - code 그룹 경로도 동일 규칙
    """
    n = 30
    prices = np.arange(1, n + 1, dtype=float)
    df_t = pd.DataFrame({"open": prices})

    y = make_labels(df_t, threshold=0.02)

    # 1) 마지막 6행 NaN
    assert y.iloc[-6:].isna().all(), (
        f"마지막 6행은 NaN이어야 함. 실제: {y.iloc[-6:].tolist()}"
    )

    # 2) 시간축 정확성
    for i in [0, 1, 5, 10, n - 7]:
        fwd = prices[i + 6] / prices[i + 1] - 1.0
        if fwd > 0.02:
            expected = 2
        elif fwd < -0.02:
            expected = 0
        else:
            expected = 1
        assert y.iloc[i] == expected, (
            f"y[{i}]: expected {expected}, got {y.iloc[i]} (fwd={fwd:.6f})"
        )

    # 3) code 그룹 경로도 동일 규칙
    df_g = pd.DataFrame(
        {
            "code": ["A"] * n + ["B"] * n,
            "open": list(prices) + list(prices * 2.0),
        }
    )
    y_g = make_labels(df_g, threshold=0.02)
    assert y_g.iloc[-6:].isna().all(), "code 경로 끝행 NaN 규칙 위반"
    assert y_g.iloc[0:n - 6].tolist() == y.iloc[0:n - 6].tolist(), (
        "code 경로/단일 경로 라벨 불일치"
    )

    # 4) 중립 라벨
    df_flat = pd.DataFrame({"open": np.full(n, 100.0)})
    y_flat = make_labels(df_flat, threshold=0.02)
    assert y_flat.iloc[0] == 1, f"평탄 구간 중립 라벨 오류: {y_flat.iloc[0]}"

    print("✅ make_labels 회귀 테스트 통과 (T+1→T+6 / 끝행 NaN / code 경로)")


def _safe_div(a: pd.Series, b: pd.Series) -> pd.Series:
    return a / b.replace(0, np.nan)


def build_features(
    df: pd.DataFrame, bands: Optional[pd.DataFrame] = None
) -> pd.DataFrame:
    """Causal technical features. No future values are used."""
    x = df.copy()
    close = x["close"].astype(float)
    open_ = x["open"].astype(float)
    high = x["high"].astype(float)
    low = x["low"].astype(float)
    volume = x["volume"].astype(float).replace(0, np.nan)

    ret1 = close.pct_change()
    logret = np.log(close).diff()

    f = pd.DataFrame(index=x.index)
    f["ret_1d"] = ret1
    f["ret_3d"] = close.pct_change(3)
    f["ret_5d"] = close.pct_change(5)
    f["ret_10d"] = close.pct_change(10)
    f["ret_20d"] = close.pct_change(20)
    f["logret_vol_5"] = logret.rolling(5).std()
    f["logret_vol_10"] = logret.rolling(10).std()
    f["logret_vol_20"] = logret.rolling(20).std()

    tr = pd.concat(
        [(high - low), (high - close.shift(1)).abs(), (low - close.shift(1)).abs()],
        axis=1,
    ).max(axis=1)
    atr14 = tr.rolling(14).mean()
    f["natr14"] = _safe_div(atr14, close)

    f["range_pct"] = _safe_div(high - low, close)
    f["body_pct"] = _safe_div(close - open_, open_)
    f["close_location"] = _safe_div(close - low, high - low)

    sma5 = close.rolling(5).mean()
    sma20 = close.rolling(20).mean()
    sma60 = close.rolling(60).mean()
    f["close_sma5"] = _safe_div(close, sma5) - 1
    f["close_sma20"] = _safe_div(close, sma20) - 1
    f["close_sma60"] = _safe_div(close, sma60) - 1
    f["sma5_sma20"] = _safe_div(sma5, sma20) - 1
    f["sma20_sma60"] = _safe_div(sma20, sma60) - 1

    delta = close.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = _safe_div(gain, loss)
    f["rsi14"] = 100 - (100 / (1 + rs))

    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    macd_signal = macd.ewm(span=9, adjust=False).mean()
    f["macd_pct"] = _safe_div(macd, close)
    f["macd_hist_pct"] = _safe_div(macd - macd_signal, close)

    vol_ma20 = volume.rolling(20).mean()
    vol_std20 = volume.rolling(20).std()
    f["volume_ratio20"] = _safe_div(volume, vol_ma20)
    f["volume_z20"] = _safe_div(volume - vol_ma20, vol_std20)
    f["log_volume_ratio20"] = np.log(_safe_div(volume, vol_ma20))

    if bands is not None:
        b = bands.reindex(x.index)
        base = b["base"].astype(float)
        upper = b["upper"].astype(float)
        lower = b["lower"].astype(float)
        width = (upper - lower).replace(0, np.nan)
        f["band_base_gap"] = _safe_div(close - base, close)
        f["band_position"] = _safe_div(close - lower, width)
        f["upper_gap_atr"] = _safe_div(close - upper, atr14)
        f["lower_gap_atr"] = _safe_div(close - lower, atr14)
        f["band_width_pct"] = _safe_div(width, close)
        f["above_upper"] = (close > upper).astype(float)
        f["below_lower"] = (close < lower).astype(float)

        for c in ["alpha_up", "alpha_down", "beta_up", "beta_down"]:
            if c in b.columns:
                f[c] = b[c]

    f = f.replace([np.inf, -np.inf], np.nan)
    return f


def _clean_training_data(
    features: pd.DataFrame, y: pd.Series, train_idx: np.ndarray
) -> Tuple[pd.DataFrame, np.ndarray, List[str]]:
    x = features.iloc[train_idx].copy()
    target = y.iloc[train_idx].to_numpy()
    valid_y = np.isfinite(target)
    x = x.loc[valid_y]
    target = target[valid_y].astype(int)
    feature_cols = [c for c in x.columns if pd.api.types.is_numeric_dtype(x[c])]
    x = x[feature_cols].replace([np.inf, -np.inf], np.nan)
    valid_x = np.isfinite(x.to_numpy(dtype=float)).all(axis=1)
    x = x.loc[valid_x]
    target = target[valid_x]
    return x, target, feature_cols


def _fit_scaler(x: pd.DataFrame) -> StandardScaler:
    scaler = StandardScaler()
    scaler.fit(x.to_numpy(dtype=np.float32))
    return scaler


def _make_loader(
    x: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool
) -> DataLoader:
    ds = TensorDataset(
        torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.long)
    )
    return DataLoader(ds, batch_size=min(batch_size, len(ds)), shuffle=shuffle)


def _predict_model(model, scaler, x: pd.DataFrame, device: str) -> np.ndarray:
    arr = x.to_numpy(dtype=np.float32)
    arr = scaler.transform(arr).astype(np.float32)
    with torch.no_grad():
        logits = model(torch.tensor(arr, dtype=torch.float32, device=device))
        return torch.softmax(logits, dim=1).cpu().numpy()


def train_focal_model(
    features: pd.DataFrame,
    y: pd.Series,
    train_end: int,
    config: Optional[FocalConfig] = None,
    device: Optional[str] = None,
) -> Tuple[object, StandardScaler, List[str], Dict[str, float]]:
    """Train with an inner chronological validation split.

    ADR-AS-0002:
      - 라벨 y 는 make_labels(df, threshold=cfg.threshold) 로 생성되어야 한다.
      - 학습 라벨이 OOS 첫날 open 을 참조하지 않도록 train_end 에서
        HORIZON(=6) 만큼 잘라낸다.
      - outer OOS 구간은 모델 선택에 사용되지 않는다.
    """
    cfg = config or FocalConfig()
    set_seed(cfg.seed)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    # ADR-AS-0002: 학습 라벨의 미래 참조 구간 = HORIZON 거래일
    label_safe_end = max(0, train_end - HORIZON)
    train_idx = np.arange(0, label_safe_end)
    x_all, y_all, feature_cols = _clean_training_data(features, y, train_idx)
    if len(x_all) < 150 or len(np.unique(y_all)) < 3:
        raise ValueError(
            "Focal classifier training data is insufficient or lacks a class."
        )

    split = int(len(x_all) * (1.0 - cfg.inner_val_ratio))
    split = min(max(split, 100), len(x_all) - 30)
    x_tr, x_va = x_all.iloc[:split], x_all.iloc[split:]
    y_tr, y_va = y_all[:split], y_all[split:]

    scaler = _fit_scaler(x_tr)
    xtr = scaler.transform(x_tr.to_numpy(dtype=np.float32)).astype(np.float32)
    xva = scaler.transform(x_va.to_numpy(dtype=np.float32)).astype(np.float32)

    model = FocalMLP(len(feature_cols), cfg.hidden1, cfg.hidden2, cfg.dropout).to(
        device
    )
    criterion = FocalLoss(
        gamma=cfg.gamma,
        alpha=[cfg.alpha_down, cfg.alpha_neutral, cfg.alpha_up],
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
    )

    loader = _make_loader(xtr, y_tr, cfg.batch_size, shuffle=True)
    best_state = None
    best_score = -np.inf
    best_epoch = 1
    wait = 0

    xva_t = torch.tensor(xva, dtype=torch.float32, device=device)
    _yva_t = torch.tensor(y_va, dtype=torch.long, device=device)

    for epoch in range(1, cfg.max_epochs + 1):
        model.train()
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(xb), yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_prob = torch.softmax(model(xva_t), dim=1).cpu().numpy()
        val_pred = val_prob.argmax(axis=1)
        score = f1_score(y_va, val_pred, average="macro", zero_division=0)

        if score > best_score + 1e-5:
            best_score = float(score)
            best_epoch = epoch
            best_state = {
                k: v.detach().cpu().clone() for k, v in model.state_dict().items()
            }
            wait = 0
        else:
            wait += 1
            if wait >= cfg.patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    # Refit on the complete outer training set for the selected epoch count.
    scaler_full = _fit_scaler(x_all)
    xfull = scaler_full.transform(x_all.to_numpy(dtype=np.float32)).astype(np.float32)
    full_loader = _make_loader(xfull, y_all, cfg.batch_size, shuffle=True)
    model_full = FocalMLP(len(feature_cols), cfg.hidden1, cfg.hidden2, cfg.dropout).to(
        device
    )
    criterion_full = FocalLoss(
        gamma=cfg.gamma,
        alpha=[cfg.alpha_down, cfg.alpha_neutral, cfg.alpha_up],
    ).to(device)
    optimizer_full = torch.optim.AdamW(
        model_full.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
    )
    for _ in range(best_epoch):
        model_full.train()
        for xb, yb in full_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer_full.zero_grad(set_to_none=True)
            loss = criterion_full(model_full(xb), yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model_full.parameters(), 1.0)
            optimizer_full.step()

    model_full.eval()
    stats = {
        "inner_macro_f1": best_score,
        "best_epoch": float(best_epoch),
        "train_rows": float(len(x_all)),
        "label_safe_end": float(label_safe_end),
        "n_features": float(len(feature_cols)),
        "threshold": float(cfg.threshold),
        "horizon": float(HORIZON),
    }
    return model_full, scaler_full, feature_cols, stats


def predict_focal(
    model,
    scaler: StandardScaler,
    feature_cols: List[str],
    features: pd.DataFrame,
    device: Optional[str] = None,
) -> np.ndarray:
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    x = features.reindex(columns=feature_cols).copy()
    valid = np.isfinite(x.to_numpy(dtype=float)).all(axis=1)
    out = np.full((len(x), 3), np.nan, dtype=float)
    if valid.any():
        out[valid] = _predict_model(model, scaler, x.loc[valid], device)
    return out


# ============================================================
# 회귀 테스트 진입점
# ============================================================
if __name__ == "__main__":
    _regression_test_make_labels()
