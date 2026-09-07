# config/features.py

FEATURE_COLUMNS = [
    "sma_5",
    "sma_20",
    "sma_60",
    "ema_12",
    "ema_26",
    "rsi_14",
    "macd",
    "macd_signal",
    "macd_hist",
    "bb_mid",
    "bb_upper",
    "bb_lower",
    "bb_bandwidth",
    "true_range",
    "atr_14",
    "hv_20",
    "parkinson_20",
    "vol_sma_20",
    "vol_ratio_20",
    "obv",
    "vwap_20",
    "vol_roc_5",
]


def get_features(df):
    """
    DataFrame에서 FEATURE_COLUMNS에 해당하는 칼럼만 반환합니다.

    Parameters
    ----------
    df : pd.DataFrame
        입력 데이터

    Returns
    -------
    list
        FEATURE_COLUMNS에 포함된 칼럼 이름들의 리스트
    """
    return [c for c in df.columns if c in FEATURE_COLUMNS]
