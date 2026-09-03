# 성과 지표

from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from huggingface_hub import hf_hub_download

# 실제 데이터 불러오기

path = hf_hub_download(
    repo_id="qurious-quant/alphastack-krx-dev",
    filename="small/features_labels_kospi200_dev.csv",
    repo_type="dataset",
)
df = pd.read_csv(path)

FEATURES = [
    c
    for c in df.columns
    if c
    not in (
        "bas_dd",
        "date",
        "index_name",
        "index_class",
        "open",
        "high",
        "low",
        "close",
        "change",
        "change_rate",
        "volume",
        "value",
        "market_cap",
        "fwd_return_5d",
        "label",
    )
]
X, y = df[FEATURES], df["label"]


# ============================================================
# 1. Maximum Drawdown (MDD) - 최대 낙폭
# ============================================================


def maximum_drawdown(returns: np.ndarray) -> float:
    """
    Maximum Drawdown (MDD) - 최대 낙폭

    📐 HTML 수식 (동치 표현)
    MDD = (V_peak - V_trough) / V_peak
       = max( (MaxCR_t - CR_t) / (1 + MaxCR_t) )

    Parameters
    ----------
    returns : array-like
        기간별 단순 수익률 (Simple Returns)

    Returns
    -------
    float
        최대 낙폭 (양수). 예: 0.25 = -25% MDD

    Notes
    -----
    - V_peak : 누적 자산의 최고점 (Running Max)
    - V_trough : 누적 자산의 최저점 (Trough)
    - CR_t : 시점 t의 누적 수익률
    - MaxCR_t : 시점 t까지의 누적 수익률 최고점
    - 내부 구현: cumulative / running_max - 1 (음수) → abs(min) 으로 양수 변환
    """
    returns = np.asarray(returns, dtype=float)

    # 1. 누적 자산 곡선 (Cumulative Wealth): 1원을 기준으로 한 누적 곱
    cumulative = np.cumprod(1 + returns)

    # 2. 누적 최고점 (Running Maximum): V_peak
    running_max = np.maximum.accumulate(cumulative)

    # 3. 고점 대비 하락률 (Drawdown): (V_t / V_peak) - 1 (항상 0 이하)
    drawdown = cumulative / running_max - 1

    # 4. 최소값(가장 깊은 하락, V_trough)의 절댓값 반환
    return abs(np.min(drawdown))


# ============================================================
# 2. Sharpe Ratio (샤프 비율)
# ============================================================


def sharpe_ratio(
    returns: np.ndarray, risk_free_rate: float = 0.0, periods_per_year: int = 252
) -> float:
    """
    Annualized Sharpe Ratio - 연환산 샤프 비율

    📐 HTML 수식
    Sharpe_annual = (R_p - R_f) / σ_p * √N

    여기서:
    - R_p : 포트폴리오 평균 수익률 (Mean of Returns)
    - R_f : 무위험 수익률 (Risk-Free Rate)
    - σ_p : 포트폴리오 수익률의 표준편차 (Standard Deviation)
    - N : 연간 관측 횟수 (일간 252, 주간 52, 월간 12)

    Parameters
    ----------
    returns : array-like
        기간별 수익률
    risk_free_rate : float
        연간 무위험수익률 (예: 0.02 = 2%)
    periods_per_year : int
        연간 관측 횟수

    Returns
    -------
    float
        연환산 Sharpe Ratio
    """
    returns = np.asarray(returns, dtype=float)

    # 기간별 무위험수익률로 변환 (연간 -> 일간/주간/월간)
    rf_period = (1 + risk_free_rate) ** (1 / periods_per_year) - 1

    # 초과 수익률 (Excess Returns) = R_p - R_f
    excess_returns = returns - rf_period

    mean_excess = np.mean(excess_returns)  # R_p - R_f (기간별 평균)
    std_excess = np.std(excess_returns, ddof=1)  # σ_p (표본 표준편차)

    if std_excess == 0:
        return np.nan

    # 연율화: (기간별 SR) * √N
    return (mean_excess / std_excess) * np.sqrt(periods_per_year)


# ============================================================
# 3. Sortino Ratio (소티노 비율) - 하방 변동성만 사용
# ============================================================


def sortino_ratio(
    returns: np.ndarray,
    target_return: float = 0.0,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252,
) -> float:
    """
    Annualized Sortino Ratio - 연환산 소티노 비율

    📐 HTML 수식
    Sortino_annual = (R_p - MAR) / σ_d * √N

    여기서 하방 변동성 (σ_d) :
    σ_d = √( 1/N * Σ_t=1^N min(0, R_t - MAR)^2 )

    - MAR : Minimum Acceptable Return (최소 요구 수익률)
    - R_t : 시점 t의 수익률
    - σ_d : 하방 변동성 (Downside Deviation, 음수 수익률만 고려)

    Parameters
    ----------
    returns : array-like
        기간별 수익률
    target_return : float
        기간별 목표수익률 (MAR). 기본값 0.
    risk_free_rate : float
        연간 무위험수익률
    periods_per_year : int
        연간 관측 횟수

    Returns
    -------
    float
        연환산 Sortino Ratio
    """
    returns = np.asarray(returns, dtype=float)

    # 기간별 무위험수익률
    rf_period = (1 + risk_free_rate) ** (1 / periods_per_year) - 1

    # 초과 수익률 (R_t - R_f)
    excess_returns = returns - rf_period

    # MAR 대비 하방 부분만 추출 (음수만 남기고 양수는 0으로)
    # = min(0, R_t - MAR)
    downside = np.minimum(excess_returns - target_return, 0)

    # 하방 변동성 (σ_d)
    downside_deviation = np.sqrt(np.mean(downside**2))

    if downside_deviation == 0:
        return np.nan

    # 연환산
    annualized_excess_return = np.mean(excess_returns) * periods_per_year
    annualized_downside_deviation = downside_deviation * np.sqrt(periods_per_year)

    return annualized_excess_return / annualized_downside_deviation


# ============================================================
# 4. Drawdown Periods 추출 (Sterling Ratio용)
# ============================================================


def _get_drawdown_periods(returns: np.ndarray) -> List[float]:
    """
    전체 기간에서 발생한 각각의 하락 구간(Drawdown Episode)의 최대 깊이를 반환.

    Returns
    -------
    List[float]
        각 하락 구간별 MDD 값 (양수)의 리스트.
        예: [0.15, 0.12, 0.10, ...]
    """
    returns = np.asarray(returns, dtype=float)
    cumulative = np.cumprod(1 + returns)
    running_max = np.maximum.accumulate(cumulative)

    drawdown_series = cumulative / running_max - 1  # 항상 0 이하

    periods = []
    i = 0
    while i < len(drawdown_series):
        if drawdown_series[i] < 0:
            # 하락 시작 (Peak)
            start = i  # noqa: F841
            max_dd = drawdown_series[i]
            # 하락이 끝날 때까지(0으로 복귀) 또는 데이터 끝까지 이동
            while i < len(drawdown_series) and drawdown_series[i] < 0:
                if drawdown_series[i] < max_dd:
                    max_dd = drawdown_series[i]  # 최저점 (Trough)
                i += 1
            periods.append(abs(max_dd))  # 양수로 변환하여 저장
        else:
            i += 1

    return periods


def average_drawdown(returns: np.ndarray) -> float:
    """
    Average Drawdown - 평균 낙폭 (전체 기간 평균)

    📐 HTML 수식
    Avg_DD = mean( (Cumulative(t) / RunningMax(t)) - 1 )

    Parameters
    ----------
    returns : array-like
        기간별 수익률

    Returns
    -------
    float
        평균 낙폭 (양수)
    """
    returns = np.asarray(returns, dtype=float)
    cumulative = np.cumprod(1 + returns)
    running_max = np.maximum.accumulate(cumulative)
    drawdown = cumulative / running_max - 1
    return abs(np.mean(drawdown))


# ============================================================
# 5. Sterling Ratio (스털링 비율) - 평균 MDD 대비 (상위 K개)
# ============================================================


def sterling_ratio(
    returns: np.ndarray,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252,
    top_k: Optional[int] = 3,
) -> float:
    """
    Sterling Ratio - 스털링 비율

    📐 HTML 수식 (실무 표준)
    Sterling = (Annualized Return - R_f) / (Average of Top K MDDs)

    - 일반적으로 Top 3~5개의 MDD 평균을 분모로 사용 (이상치에 강건).
    - top_k=None 으로 설정하면 전체 기간의 평균 낙폭(Average Drawdown)을 사용.

    Parameters
    ----------
    returns : array-like
        기간별 수익률
    risk_free_rate : float
        연간 무위험수익률
    periods_per_year : int
        연간 관측 횟수
    top_k : Optional[int]
        평균낼 MDD의 개수 (기본값 3). None이면 전체 평균 사용.

    Returns
    -------
    float
        Sterling Ratio
    """
    returns = np.asarray(returns, dtype=float)

    # CAGR (연평균 복리 수익률)
    annualized_return = np.prod(1 + returns) ** (periods_per_year / len(returns)) - 1
    excess_return = annualized_return - risk_free_rate

    if top_k is not None and top_k > 0:
        # 상위 K개 MDD 추출
        periods = _get_drawdown_periods(returns)
        if not periods:
            return np.nan
        # 내림차순 정렬 후 상위 K개 평균
        sorted_periods = sorted(periods, reverse=True)
        avg_mdd = np.mean(sorted_periods[:top_k])
    else:
        # 전체 평균 낙폭 사용
        avg_mdd = average_drawdown(returns)

    if avg_mdd == 0:
        return np.nan

    return excess_return / avg_mdd


# ============================================================
# 6. Calmar Ratio (칼마 비율) - 최대 MDD 대비
# ============================================================


def calmar_ratio(
    returns: np.ndarray, risk_free_rate: float = 0.0, periods_per_year: int = 252
) -> float:
    """
    Calmar Ratio - 칼마 비율

    📐 HTML 수식
    Calmar = (Annualized Return - R_f) / Maximum Drawdown (MDD)

    - Sterling과 동일한 분자(CAGR 기준 초과수익)를 사용.
    - 분모로 '단일 최대 MDD'를 사용하여 가장 보수적인 하방 리스크 지표.

    Parameters
    ----------
    returns : array-like
        기간별 수익률
    risk_free_rate : float
        연간 무위험수익률
    periods_per_year : int
        연간 관측 횟수

    Returns
    -------
    float
        Calmar Ratio
    """
    returns = np.asarray(returns, dtype=float)

    # CAGR (연평균 복리 수익률)
    annualized_return = np.prod(1 + returns) ** (periods_per_year / len(returns)) - 1
    excess_return = annualized_return - risk_free_rate

    mdd = maximum_drawdown(returns)

    if mdd == 0:
        return np.nan

    return excess_return / mdd


# ============================================================
# 7. Deflated Sharpe Ratio (디플레이티드 샤프 비율)
# ============================================================


def deflated_sharpe_ratio(
    sharpe: float,
    n_observations: int,
    skewness: float,
    kurtosis: float,
    n_trials: int = 1,
    expected_max_sharpe: float = 0.0,
) -> float:
    """
    Deflated Sharpe Ratio (DSR) - 디플레이티드 샤프 비율

    📐 HTML 수식 (Bailey & Lopez de Prado, 2014)
    DSR = Φ( (SR* - SR0) / σ(SR*) )

    - σ(SR*) = √( (1 - γ₃·SR* + ((γ₄ - 1)/4)·SR*²) / (T - 1) )
    - SR0 = norm.ppf(1 - 1/N) · σ(SR*)  (극값 통계로 추정된 기대 최대 샤프)

    Parameters
    ----------
    sharpe : float
        관측된 Sharpe Ratio (SR*)
    n_observations : int
        수익률 관측치 수 (T)
    skewness : float
        수익률 분포의 왜도 (γ₃)
    kurtosis : float
        수익률 분포의 첨도 (γ₄) - Fisher=False 기준 (정규분포=3)
    n_trials : int
        백테스트한 전략/모델의 총 개수 (N). 기본값 1.
    expected_max_sharpe : float
        (선택) SR0를 외부에서 직접 지정할 경우 사용.

    Returns
    -------
    float
        Deflated Sharpe Ratio (0 ~ 1 확률값)
    """
    from scipy.stats import norm

    if n_observations <= 1:
        return np.nan

    # 1. Sharpe Ratio의 표준오차 (σ(SR*)) 계산
    sharpe_std = np.sqrt(
        (1 - skewness * sharpe + ((kurtosis - 1) / 4) * sharpe**2)
        / (n_observations - 1)
    )

    if sharpe_std == 0:
        return np.nan

    # 2. 귀무가설 하 기대 최대 Sharpe (SR0) 계산
    if n_trials > 1:
        expected_max_sharpe = norm.ppf(1 - 1 / n_trials) * sharpe_std

    # 3. DSR = Φ( (SR* - SR0) / σ(SR*) )
    dsr = norm.cdf((sharpe - expected_max_sharpe) / sharpe_std)

    return dsr


# ============================================================
# 8. 전체 지표 계산 (Wrapper)
# ============================================================


def calculate_all_metrics(
    returns: np.ndarray,
    risk_free_rate: float = 0.0,
    target_return: float = 0.0,
    periods_per_year: int = 252,
    n_trials: int = 1,
    sterling_top_k: int = 3,
) -> Dict[str, float]:
    """
    모든 성과지표를 한 번에 계산하는 통합 함수

    Parameters
    ----------
    returns : array-like
        기간별 수익률
    risk_free_rate : float
        연간 무위험수익률 (R_f)
    target_return : float
        Sortino Ratio 계산 시 사용할 MAR (Minimum Acceptable Return)
    periods_per_year : int
        연간 관측 횟수 (일간 252, 주간 52, 월간 12)
    n_trials : int
        Deflated Sharpe Ratio 계산 시 백테스트한 총 전략의 수 (N)
    sterling_top_k : int
        Sterling Ratio 계산 시 사용할 상위 MDD 개수 (기본 3)

    Returns
    -------
    dict
        MDD, Sharpe, Sortino, Sterling, Calmar, Deflated Sharpe Ratio
    """
    returns = np.asarray(returns, dtype=float)

    # Sharpe Ratio 계산
    sharpe = sharpe_ratio(returns, risk_free_rate, periods_per_year)

    # Deflated Sharpe Ratio 계산용 왜도/첨도
    from scipy.stats import kurtosis, skew

    skewness = skew(returns)
    kurt = kurtosis(returns, fisher=False)  # 정규분포 기준 3

    dsr = deflated_sharpe_ratio(
        sharpe=sharpe,
        n_observations=len(returns),
        skewness=skewness,
        kurtosis=kurt,
        n_trials=n_trials,
    )

    return {
        "MDD": maximum_drawdown(returns),
        "Sharpe Ratio": sharpe,
        "Sortino Ratio": sortino_ratio(
            returns, target_return, risk_free_rate, periods_per_year
        ),
        "Sterling Ratio": sterling_ratio(
            returns, risk_free_rate, periods_per_year, top_k=sterling_top_k
        ),
        "Calmar Ratio": calmar_ratio(returns, risk_free_rate, periods_per_year),
        "Deflated Sharpe Ratio": dsr,
    }


# ============================================================
# 🚀 실행 예시 (Example Usage) - 실제 데이터 적용
# ============================================================
if __name__ == "__main__":

    print("📊 성과 지표 계산을 시작합니다...\n")

    # ============================================================
    # 🔥 [필수] 1. 데이터를 날짜 기준으로 오름차순 정렬
    # ============================================================
    # date 컬럼이 문자열 'YYYY-MM-DD' 형식이므로, 그대로 정렬해도 시간순이 됨
    df = df.sort_values("date").reset_index(drop=True)

    # 코스피 일때는 close, 종목일 때는 adj_close 사용
    daily_returns = df["change_rate"].dropna().values / 100.0

    # ============================================================
    # 📌 [디버깅] 계산된 수익률의 기본 통계를 출력해 정상인지 확인
    # ============================================================
    print("📈 수익률 통계 (일간):")
    print(f"   개수: {len(daily_returns):,}개")
    print(f"   평균: {daily_returns.mean():.6f} ({daily_returns.mean()*100:.4f}%)")
    print(f"   표준편차: {daily_returns.std():.6f} ({daily_returns.std()*100:.4f}%)")
    print(f"   최소: {daily_returns.min():.6f} ({daily_returns.min()*100:.4f}%)")
    print(f"   최대: {daily_returns.max():.6f} ({daily_returns.max()*100:.4f}%)\n")

    # ------------------------------------------------------------
    # 📌 성과 지표 계산 실행
    # ------------------------------------------------------------
    results = calculate_all_metrics(
        returns=daily_returns,
        risk_free_rate=0.02,  # 연 2% 무위험수익률 (예: 국고채 3년물)
        target_return=0.0,  # MAR (Minimum Acceptable Return) = 0%
        periods_per_year=252,  # 일간 데이터 기준
        n_trials=50,  # 백테스트해본 전략이 50개라고 가정 (DSR용)
        sterling_top_k=3,  # 상위 3개 MDD 평균 사용
    )

    # ------------------------------------------------------------
    # 📌 결과 출력 (Print Results)
    # ------------------------------------------------------------
    print("✅ 계산 완료! 결과는 다음과 같습니다.\n")
    print("=" * 50)
    print("📈 성과 지표 (Performance Metrics)")
    print("=" * 50)

    for key, value in results.items():
        if key == "Deflated Sharpe Ratio":
            print(f"{key:20s} : {value:.4f}  ({value*100:.2f}%)")
        elif key == "MDD":
            print(f"{key:20s} : {value:.4f}  ({value*100:.2f}%)")
        else:
            print(f"{key:20s} : {value:.4f}")

    print("=" * 50)

    # 간단한 해석 가이드 출력
    print("\n💡 해석 가이드")
    print("-" * 50)
    # 1. 실제 연평균 복리 수익률(CAGR)과 초과 수익률 계산
    n = len(daily_returns)  # 전체 관측치 개수
    rf = 0.02  # calculate_all_metrics에 넣은 값과 동일
    cagr = (np.prod(1 + daily_returns) ** (252 / n)) - 1
    excess_cagr = cagr - rf  # ← 여기서 변수가 정의됩니다!

    # 2. CAGR 관련 정보 출력
    print(f"📈 연평균 수익률(CAGR)   : {cagr*100:.2f}%")
    print(f"📈 연 초과수익률 (CAGR - {rf*100:.0f}%) : {excess_cagr*100:.2f}%")
    print("-" * 50)
    print(f"📉 MDD         : 최대 손실률 {results['MDD']*100:.2f}%")
    print(
        f"📊 Sharpe      : 위험 1단위당 초과수익 {results['Sharpe Ratio']:.2f} (연율)"
    )
    print(
        f"🔽 Sortino     : 하방위험 1단위당 초과수익 {results['Sortino Ratio']:.2f} (연율)"
    )
    print(
        f"⚖️ Sterling    : 연 초과수익률({excess_cagr*100:.2f}%) / "
        f"평균MDD = {results['Sterling Ratio']:.4f}"
    )
    print(
        f"📉 Calmar      : 연 초과수익률({excess_cagr*100:.2f}%) / "
        f"최대MDD = {results['Calmar Ratio']:.4f}"
    )
    print(
        f"🎯 DSR         : 이 전략이 우연이 아닐 확률 {results['Deflated Sharpe Ratio']*100:.2f}%"
    )
