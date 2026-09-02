"""모델에 홀드아웃이 들어가기 전에 개발구간만 남긴다."""

from __future__ import annotations

import pandas as pd

from evaluation.horizon import HOLDOUT_START


def holdout_safe_frame(
    frame: pd.DataFrame,
    *,
    holdout_start: str = HOLDOUT_START,
    label_horizon: int = 5,
    date_column: str = "bas_dd",
) -> pd.DataFrame:
    """홀드아웃 이전 행에서 경계 직전 레이블 지평까지 제거한다.

    모델 노트북마다 날짜 필터를 따로 쓰면 한 곳은 언젠가 빠진다. 모든 모델은 이
    함수가 반환한 표만 사용한다. `label_horizon=5`이면 개발구간의 마지막 5개
    거래일을 추가로 제거한다. 그 행들의 미래 5거래일 라벨이 봉인 구간을 볼 수 있기
    때문이다.

    원본 ``frame``은 바꾸지 않는다. 반환값의 ``attrs["holdout_filter"]``에는 몇 행과
    어떤 날짜를 제거했는지 남겨 실행 결과에서 확인할 수 있게 한다.
    """
    if date_column not in frame.columns:
        raise ValueError(f"기준일 열이 없습니다: {date_column}")
    if label_horizon < 0:
        raise ValueError(f"label_horizon은 0 이상이어야 합니다: {label_horizon}")
    if not isinstance(holdout_start, str) or len(holdout_start) != 8:
        raise ValueError(f"holdout_start는 YYYYMMDD 문자열이어야 합니다: {holdout_start!r}")

    dates = (
        frame[date_column]
        .astype("string")
        .str.strip()
        .str.replace(r"\.0$", "", regex=True)
        .str.zfill(8)
    )
    invalid = dates.isna() | ~dates.str.fullmatch(r"\d{8}")
    if invalid.any():
        examples = frame.loc[invalid, date_column].head(3).tolist()
        raise ValueError(f"YYYYMMDD로 해석할 수 없는 기준일이 있습니다: {examples}")

    dev_mask = dates < holdout_start
    dev = frame.loc[dev_mask].copy()
    dev_dates = dates.loc[dev_mask]
    if dev.empty:
        raise ValueError(f"{holdout_start} 이전 개발구간 행이 없습니다.")

    holdout_rows_removed = int((~dev_mask).sum())
    purged_dates: list[str] = []
    # 이미 공급 단계에서 라벨 꼬리까지 제거한 dev 전용 파일은 다시 자르지 않는다.
    # 원본에 봉인 구간이 있을 때만 모델 계층이 직접 경계를 만들고 마지막 지평을 비운다.
    if label_horizon and holdout_rows_removed:
        trading_dates = sorted(dev_dates.unique().tolist())
        if len(trading_dates) <= label_horizon:
            raise ValueError(
                "개발구간 거래일이 레이블 지평보다 짧습니다: "
                f"거래일 {len(trading_dates)}개, 지평 {label_horizon}개"
            )
        purged_dates = trading_dates[-label_horizon:]
        keep_mask = ~dev_dates.isin(purged_dates)
        dev = dev.loc[keep_mask].copy()
        dev_dates = dev_dates.loc[keep_mask]

    if dev.empty or (dev_dates >= holdout_start).any():
        raise RuntimeError("홀드아웃 차단 뒤에도 안전한 개발구간을 만들지 못했습니다.")

    dev[date_column] = dev_dates.astype("string").to_numpy()
    dev = dev.sort_values(date_column, kind="stable").reset_index(drop=True)
    dev.attrs["holdout_filter"] = {
        "holdout_start": holdout_start,
        "label_horizon": label_horizon,
        "source_rows": int(len(frame)),
        "holdout_rows_removed": holdout_rows_removed,
        "boundary_rows_removed": int(dates.loc[dev_mask].isin(purged_dates).sum()),
        "purged_dates": purged_dates,
        "max_dev_date": str(dev[date_column].max()),
    }
    return dev
