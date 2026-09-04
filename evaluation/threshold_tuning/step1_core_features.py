import numpy as np
import pandas as pd
from huggingface_hub import hf_hub_download


# ============================================================
# 0. 데이터 로드
# ============================================================
def load_data() -> pd.DataFrame:
    path = hf_hub_download(
        repo_id="qurious-quant/alphastack-krx-dev",
        filename="small/features_labels_kospi200_dev.csv",
        repo_type="dataset",
    )
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"])
    df.sort_values("date", inplace=True)
    df.set_index("date", inplace=True)

    required = ["open", "high", "low", "close", "volume"]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"데이터에 필요한 컬럼이 없습니다: {missing}")
    return df


# ============================================================
# 1. Base (중심선) 계산
# ============================================================
def compute_base(df: pd.DataFrame, base_type: str = "SMA20") -> pd.Series:
    close = df["close"]
    if base_type == "SMA20":
        return close.rolling(20).mean()
    elif base_type == "EMA20":
        return close.ewm(span=20, adjust=False).mean()
    elif base_type == "EMA30":
        return close.ewm(span=30, adjust=False).mean()
    elif base_type == "SMA60":
        return close.rolling(60).mean()
    else:
        raise ValueError(f"지원하지 않는 Base 타입: {base_type}")


# ============================================================
# 2. Volatility (변동성) 계산
# ============================================================
def compute_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high = df["high"]
    low = df["low"]
    close = df["close"].shift(1)
    tr1 = high - low
    tr2 = (high - close).abs()
    tr3 = (low - close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.rolling(period).mean()


def compute_natr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    atr = compute_atr(df, period)
    return atr / df["close"]


def compute_std_ret(df: pd.DataFrame, period: int = 20) -> pd.Series:
    ret = df["close"].pct_change()
    return ret.rolling(period).std()


# ============================================================
# 3. Volume (거래량) 계산
# ============================================================
def compute_log_rv(df: pd.DataFrame, period: int = 20) -> pd.Series:
    volume = df["volume"]
    sma_vol = volume.rolling(period).mean()
    return np.log(volume / sma_vol)


def compute_volume_zscore(df: pd.DataFrame, period: int = 20) -> pd.Series:
    volume = df["volume"]
    sma_vol = volume.rolling(period).mean()
    std_vol = volume.rolling(period).std()
    return (volume - sma_vol) / std_vol


def compute_volume_shock(df: pd.DataFrame, period: int = 20) -> pd.Series:
    volume = df["volume"]
    sma_vol = volume.rolling(period).mean()
    return volume / sma_vol - 1


# ============================================================
# 4. 기준선 (Upper / Lower) 생성 (핵심, 업데이트됨)
# ============================================================
def compute_bands(
    df: pd.DataFrame,
    base_type: str = "SMA20",  # 문서에 따라 SMA20으로 기본값 변경
    vol_type: str = "ATR14",
    volume_type: str = "LogRV20",
    alpha: float = 1.0,
    beta: float = 0.0,
    asym: bool = False,
    alpha_up: float = None,
    alpha_down: float = None,
    beta_up: float = None,  # 추가: 상승 측 거래량 승수
    beta_down: float = None,  # 추가: 하락 측 거래량 승수
) -> pd.DataFrame:
    # 1) Base
    base = compute_base(df, base_type)

    # 2) Volatility
    if vol_type == "ATR14":
        vol = compute_atr(df, 14)
    elif vol_type == "NATR14":
        vol = compute_natr(df, 14)
    elif vol_type == "STD20":
        vol = compute_std_ret(df, 20)
    else:
        raise ValueError(f"지원하지 않는 Volatility 타입: {vol_type}")

    # 3) Volume effect
    if volume_type is None:
        vol_effect = 1.0
    elif volume_type == "LogRV20":
        vol_effect = compute_log_rv(df, 20)
    elif volume_type == "Zscore20":
        vol_effect = compute_volume_zscore(df, 20)
    elif volume_type == "Shock20":
        vol_effect = compute_volume_shock(df, 20)
    else:
        raise ValueError(f"지원하지 않는 Volume 타입: {volume_type}")

    # 4) Width 계산 (exp 적용)
    if asym:
        # 비대칭 모드: 각각의 alpha, beta 사용
        if alpha_up is None or alpha_down is None:
            raise ValueError(
                "비대칭 모드에서는 alpha_up, alpha_down을 반드시 지정해야 합니다."
            )
        # beta_up/down이 없으면 기존 beta 값으로 통일 (하위 호환성)
        _beta_up = beta_up if beta_up is not None else beta
        _beta_down = beta_down if beta_down is not None else beta

        upper_width = vol * np.exp(alpha_up + _beta_up * vol_effect)
        lower_width = vol * np.exp(alpha_down + _beta_down * vol_effect)
    else:
        # 대칭 모드
        width = vol * np.exp(alpha + beta * vol_effect)
        upper_width = width
        lower_width = width

    result = df.copy()
    result["base"] = base
    result["upper"] = base + upper_width
    result["lower"] = base - lower_width
    return result


# ============================================================
# 5. 모든 피처 한 번에 계산 (분석/디버깅용)
# ============================================================
def prepare_all_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["base_SMA20"] = compute_base(df, "SMA20")
    df["base_EMA20"] = compute_base(df, "EMA20")
    df["base_EMA30"] = compute_base(df, "EMA30")
    df["base_SMA60"] = compute_base(df, "SMA60")

    df["ATR14"] = compute_atr(df, 14)
    df["NATR14"] = compute_natr(df, 14)
    df["STD20"] = compute_std_ret(df, 20)

    df["LogRV20"] = compute_log_rv(df, 20)
    df["VolZ20"] = compute_volume_zscore(df, 20)
    df["VolShock20"] = compute_volume_shock(df, 20)
    return df


if __name__ == "__main__":
    # 기존 검증용 출력 코드 (생략 가능, 필요시 주석 해제)
    pass
