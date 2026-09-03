# weekly_predictor.py
# 영업일 기준 5일 후 예측을 위한 배치 실행기 (KRX 캘린더 + ML 모델 통합)
# - 한국 공휴일을 반영한 KRX 영업일 캘린더 사용
# - 입력일 기준으로 5영업일 후를 예측일로 설정
# - 기준일과 예측일 각각 직전 5평일(월~금, 공휴일 포함)을 추출하여 페어링
# - 양쪽 모두 영업일이고 market_data에 존재하는 페어에 대해서만 모델 예측 수행 (아니면 "측정 불가")

from datetime import datetime
from typing import Callable, Dict, List, Union

import exchange_calendars as ec  # 한국 공휴일 포함 영업일 계산
import pandas as pd

# 실제 데이터 로드 및 ML 모델을 위한 임포트
from huggingface_hub import hf_hub_download
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

# ============================================================
# 1. 캘린더 관련 헬퍼 함수
# ============================================================


def get_krx_calendar() -> ec.ExchangeCalendar:
    """KRX(한국거래소) 영업일 캘린더를 반환합니다."""
    return ec.get_calendar("XKRX")


def is_business_day(date: pd.Timestamp) -> bool:
    """해당 날짜가 KRX 영업일인지 확인합니다."""
    cal = get_krx_calendar()
    return date in cal.schedule.index


def add_business_days(date: pd.Timestamp, days: int) -> pd.Timestamp:
    """
    주어진 날짜에서 days 영업일 후의 날짜를 반환합니다.
    days는 양수여야 합니다.
    """
    cal = get_krx_calendar()
    trading_days = cal.schedule.index

    if date not in trading_days:
        raise ValueError(f"{date.strftime('%Y-%m-%d')}은(는) 영업일이 아닙니다.")

    idx = trading_days.get_loc(date)
    target_idx = idx + days

    if target_idx >= len(trading_days):
        raise ValueError(f"해당 날짜({date})로부터 {days} 영업일을 찾을 수 없습니다.")

    return trading_days[target_idx]


def get_last_n_weekdays_until(date: pd.Timestamp, n: int) -> List[pd.Timestamp]:
    """
    주어진 날짜를 포함하여 직전 n개의 평일(월~금, 공휴일 포함)을 리스트로 반환합니다.
    (오름차순, 즉 과거→현재 순서로 반환)
    """
    weekdays = []
    current = date
    while len(weekdays) < n:
        # weekday: 0=월, 4=금, 5=토, 6=일
        if current.weekday() < 5:  # 월~금만 포함
            weekdays.append(current)
        current = current - pd.Timedelta(days=1)

    # 역순으로 들어갔으므로 오름차순(과거->현재)으로 정렬
    return weekdays[::-1]


# ============================================================
# 2. 내부 예측 헬퍼
# ============================================================


def _map_predictions(
    model_func: Callable[[pd.DataFrame], str],
    market_data: pd.DataFrame,
    ref_dates: List[pd.Timestamp],
    target_dates: List[pd.Timestamp],
) -> Dict[pd.Timestamp, str]:
    """
    각 기준일(ref_date)에 대해 모델을 실행하여 해당 예측일(target_date)의 예측값을 매핑합니다.
    - ref_date와 target_date가 모두 영업일이고 market_data에 존재하는 경우만 예측합니다.
    - 그렇지 않으면 "측정 불가"로 표시합니다.
    """
    predictions = {}
    for ref_date, target_date in zip(ref_dates, target_dates):
        # 기준일과 예측일이 모두 영업일인지 확인
        if not is_business_day(ref_date) or not is_business_day(target_date):
            predictions[target_date] = "측정 불가"
            continue

        # 기준일과 예측일이 모두 market_data 인덱스에 있는지 확인
        if ref_date not in market_data.index or target_date not in market_data.index:
            predictions[target_date] = "측정 불가"
            continue

        # 기준일까지의 데이터 슬라이스 (미래 정보 차단)
        data_slice = market_data[market_data.index <= ref_date]
        if data_slice.empty:
            predictions[target_date] = "측정 불가"
            continue

        # 모델 실행
        label = model_func(data_slice)
        predictions[target_date] = label

    return predictions


# ============================================================
# 3. 배치 예측 실행기 (Main Wrapper)
# ============================================================


def run_weekly_forecast(
    base_date: Union[str, pd.Timestamp, datetime],
    market_data: pd.DataFrame,
    model_func: Callable[[pd.DataFrame], str],
    num_days: int = 5,
) -> Dict[str, Union[List[pd.Timestamp], Dict[pd.Timestamp, str], List[str]]]:
    """
    입력된 base_date를 기준으로 영업일 기준 num_days일 후를 예측일로 설정하고,
    기준일과 예측일 각각의 직전 num_days개 평일(월~금, 공휴일 포함)을 페어링하여 예측을 수행합니다.

    📐 작동 원리 (Logic)
    --------------------
    1. base_date는 반드시 영업일이어야 합니다 (공휴일/주말 불가).
    2. target_date = base_date + num_days 영업일 (한국 공휴일 고려).
    3. ref_dates   = base_date를 포함한 직전 num_days개 평일 (공휴일/주말 제외, 평일만)
    4. target_dates = target_date를 포함한 직전 num_days개 평일 (공휴일/주말 제외, 평일만)
    5. 각 인덱스별로 (ref_dates[i], target_dates[i]) 쌍을 생성합니다.
    6. 각 쌍에 대해 양쪽 모두 영업일이고 market_data에 존재하면 모델 예측 실행,
       그렇지 않으면 "측정 불가"로 표시합니다.

    Parameters
    ----------
    base_date : str, pd.Timestamp, or datetime
        기준이 되는 날짜 (반드시 KRX 영업일이어야 함)
    market_data : pd.DataFrame
        날짜(DatetimeIndex)를 인덱스로 가지는 특징 데이터
    model_func : Callable[[pd.DataFrame], str]
        과거 데이터 슬라이스를 받아 '상승'/'중립'/'하락'을 반환하는 함수
    num_days : int
        예측할 영업일 간격 (기본 5)

    Returns
    -------
    Dict[str, Any]
        - 'target_dates'     : 예측 대상 날짜 리스트 (길이 num_days)
        - 'ref_dates'        : 각 예측에 사용된 기준일 리스트 (길이 num_days)
        - 'predictions_by_date' : {날짜: 예측값} 딕셔너리
        - 'prediction_list'  : 예측값 리스트 (target_dates 순서)
    """
    # 1. 입력 날짜 정규화 및 영업일 검증
    if not isinstance(base_date, pd.Timestamp):
        base_date = pd.Timestamp(base_date)

    if not is_business_day(base_date):
        raise ValueError(
            f"입력된 날짜({base_date.strftime('%Y-%m-%d')})는 KRX 영업일이 아닙니다. "
            "주말이나 공휴일을 제외한 영업일을 입력해주세요."
        )

    # 2. 예측 대상일(target_date) 계산: base_date + num_days 영업일
    target_date = add_business_days(base_date, num_days)

    # 3. 기준일 및 예측일 각각 직전 num_days개 평일 추출 (공휴일 포함)
    ref_dates = get_last_n_weekdays_until(base_date, num_days)
    target_dates = get_last_n_weekdays_until(target_date, num_days)

    # 4. 모델 실행하여 예측 매핑 (유효 쌍만 예측)
    predictions_by_date = _map_predictions(
        model_func=model_func,
        market_data=market_data,
        ref_dates=ref_dates,
        target_dates=target_dates,
    )

    # 5. 결과 포맷 정리 (순서 보장)
    prediction_list = [predictions_by_date[date] for date in target_dates]

    return {
        "target_dates": target_dates,
        "ref_dates": ref_dates,
        "predictions_by_date": predictions_by_date,
        "prediction_list": prediction_list,
    }


def _map_predictions(
    model_func: Callable[[pd.DataFrame], str],
    market_data: pd.DataFrame,
    ref_dates: List[pd.Timestamp],
    target_dates: List[pd.Timestamp],
) -> Dict[pd.Timestamp, str]:
    """
    각 기준일(ref_date)에 대해 모델을 실행하여 해당 예측일(target_date)의 예측값을 매핑합니다.
    """
    predictions = {}
    for ref_date, target_date in zip(ref_dates, target_dates):
        print(
            f"\n🔍 [디버깅] 기준일={ref_date.strftime('%Y-%m-%d')}, 예측일={target_date.strftime('%Y-%m-%d')}"
        )

        # 1. 영업일 체크
        is_ref_biz = is_business_day(ref_date)
        is_target_biz = is_business_day(target_date)
        print(f"   기준일 영업일 여부: {is_ref_biz}")
        print(f"   예측일 영업일 여부: {is_target_biz}")

        if not is_ref_biz or not is_target_biz:
            print("   ❌ 영업일이 아니므로 '측정 불가'")
            predictions[target_date] = "측정 불가"
            continue

        # 2. market_data 존재 여부 체크
        ref_exists = ref_date in market_data.index
        target_exists = target_date in market_data.index
        print(f"   기준일 데이터 존재: {ref_exists}")
        print(f"   예측일 데이터 존재: {target_exists}")

        if not ref_exists or not target_exists:
            print("   ❌ 데이터에 없으므로 '측정 불가'")
            predictions[target_date] = "측정 불가"
            continue

        # 3. 데이터 슬라이스
        data_slice = market_data[market_data.index <= ref_date]
        if data_slice.empty:
            print("   ❌ 데이터 슬라이스가 비어있음")
            predictions[target_date] = "측정 불가"
            continue

        # 4. 모델 실행
        print("   ✅ 모든 조건 통과! 모델 실행")
        label = model_func(data_slice)
        predictions[target_date] = label

    return predictions


# ============================================================
# 🚀 실행 예시 (Example Usage) - 실제 데이터 + ML 모델 적용
# ============================================================
if __name__ == "__main__":

    print("📆 KRX 실제 데이터 기반 영업일 5일 후 예측 배치 실행을 시작합니다...\n")

    # ------------------------------------------------------------
    # 📌 [0] Hugging Face에서 실제 데이터 먼저 로드 (범위 확인용)
    # ------------------------------------------------------------
    print("🔽 실제 데이터 로드 중...")
    path = hf_hub_download(
        repo_id="qurious-quant/alphastack-krx-dev",
        filename="small/features_labels_kospi200_dev.csv",
        repo_type="dataset",
    )
    df = pd.read_csv(path)

    df["date"] = pd.to_datetime(df["date"])
    df.sort_values("date", inplace=True)
    market_df = df.set_index("date")

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

    print(f"✅ 데이터 로드 완료: {len(market_df)}개 행, {len(FEATURES)}개 피처")

    # ------------------------------------------------------------
    # 📌 [1] 데이터 범위 확인 및 사용자에게 표시
    # ------------------------------------------------------------
    data_start = market_df.index.min().strftime("%Y-%m-%d")
    data_end = market_df.index.max().strftime("%Y-%m-%d")

    print(f"\n📊 사용 가능한 데이터 범위: {data_start} ~ {data_end}")
    print("   (이 범위 내의 영업일만 입력 가능합니다)\n")

    # ------------------------------------------------------------
    # 📌 [2] 터미널에서 날짜 입력 받기 (YYYY-MM-DD 형식)
    # ------------------------------------------------------------
    while True:
        input_date_str = input(
            "📅 기준 날짜를 입력하세요 (YYYY-MM-DD 형식, 예: 2024-03-07): "
        ).strip()

        if not input_date_str:
            print("❌ 날짜를 입력해주세요.\n")
            continue

        try:
            input_date = pd.Timestamp(input_date_str)

            # 데이터 범위 내에 있는지 확인
            if input_date < market_df.index.min() or input_date > market_df.index.max():
                print(
                    f"⚠️ 입력한 날짜({input_date.strftime('%Y-%m-%d')})가 데이터 범위({data_start} ~ {data_end})를 벗어났습니다."
                )
                print("   다시 입력해주세요.\n")
                continue

            break
        except Exception:
            print("❌ 잘못된 형식입니다. YYYY-MM-DD 형식으로 다시 입력해주세요.\n")
            continue

    print(f"✅ 입력된 날짜: {input_date.strftime('%Y-%m-%d')}\n")

    # ------------------------------------------------------------
    # 📌 [3] base_date 설정 (입력받은 날짜가 영업일인지 검증)
    # ------------------------------------------------------------
    base_date = input_date

    if not is_business_day(base_date):
        print(f"⚠️ 입력된 날짜({base_date.strftime('%Y-%m-%d')})는 영업일이 아닙니다.")
        print("주말이나 공휴일을 제외한 영업일을 입력해주세요.")
        exit(1)

    print(f"📅 기준일(base_date) 설정: {base_date.strftime('%Y-%m-%d')}")

    # ------------------------------------------------------------
    # 📌 [4] 모델 학습 (base_date 이전 데이터만 사용)
    # ------------------------------------------------------------
    train_df = market_df[market_df.index <= base_date]
    X_train = train_df[FEATURES]
    y_train = train_df["label"]

    le = LabelEncoder()
    y_train_encoded = le.fit_transform(y_train)

    print("🧠 모델 학습 중 (base_date 이전 데이터만 사용)...")
    model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train_encoded)
    print("✅ 모델 학습 완료")

    # ------------------------------------------------------------
    # 📌 [5] 모델 함수 정의
    # ------------------------------------------------------------
    def my_trading_model(data_slice: pd.DataFrame) -> str:
        if data_slice.empty:
            return "중립"

        latest_row = data_slice.iloc[-1]
        features_df = latest_row[FEATURES].to_frame().T
        pred_encoded = model.predict(features_df)[0]
        pred_label = le.inverse_transform([pred_encoded])[0]
        return pred_label

    # ------------------------------------------------------------
    # 📌 [6] 실행: 영업일 기준 5일 후 예측 (공휴일 반영)
    # ------------------------------------------------------------
    results = run_weekly_forecast(
        base_date=base_date,
        market_data=market_df,
        model_func=my_trading_model,
        num_days=5,
    )

    # ------------------------------------------------------------
    # 📌 [7] 결과 출력
    # ------------------------------------------------------------
    print("\n" + "=" * 60)
    print(f"✅ 기준일 (base_date): {base_date.strftime('%Y-%m-%d')}")
    print("=" * 60)
    print(
        "📅 예측 대상 날짜 (Target Dates) 및 기준일 (Ref Dates) - 평일 기준 (공휴일 포함)"
    )
    print("-" * 60)
    for ref, target in zip(results["ref_dates"], results["target_dates"]):
        print(
            f"  기준일(Ref): {ref.strftime('%Y-%m-%d')} → 예측일(Target): {target.strftime('%Y-%m-%d')}"
        )

    print("\n" + "=" * 60)
    print("🔮 예측 결과 (Predictions)")
    print("-" * 60)
    for date, pred in results["predictions_by_date"].items():
        print(f"  {date.strftime('%Y-%m-%d')} : {pred}")

    print("\n" + "=" * 60)
    print(f"📈 최종 예측 리스트 (순서대로): {results['prediction_list']}")
    print("=" * 60)

    print("\n💡 운영 가이드")
    print("-" * 60)
    print(f"📊 데이터 범위: {data_start} ~ {data_end}")
    print("1. base_date는 반드시 KRX 영업일(주말/공휴일 제외)을 입력해야 합니다.")
    print("2. 예측일(target_date) = base_date + 5 영업일로 계산됩니다.")
    print("3. 기준일과 예측일 각각의 직전 5평일(월~금, 공휴일 포함)을 페어링합니다.")
    print("4. 기준일 또는 예측일이 비영업일(주말/공휴일)이면 '측정 불가'로 표시됩니다.")
    print("5. 모든 날짜는 한국 공휴일(개천절, 삼일절, 추석 등)을 고려합니다.")
