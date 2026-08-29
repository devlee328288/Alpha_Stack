"""시세를 **그 시점에 알 수 있었던 만큼만** 내보낸다.

이 모듈이 하는 일은 하나다 — 저장소가 주는 시계열에서 **`as_of` 시점에 아직 몰랐던
구간을 잘라내는 것**이다. 그게 전부인데, 그게 전부이기 때문에 중요하다.

왜 저장소를 직접 부르면 안 되나
------------------------------
저장소는 표에 있는 것을 전부 준다. 표에는 **오늘까지** 들어 있다. 2020년 폴드를
학습하면서 그 함수를 그대로 부르면 2026년까지가 딸려 오고, 그 상태로 피처를 만들면
이동평균 한 줄에 미래가 섞인다. **예외는 안 난다.** 성능만 좋아진다.

그래서 저장소를 직접 부르는 길을 막고 이 문 하나만 열어 둔다. 문을 지나려면
`as_of` 를 내야 하고, **기본값이 없어서 빠뜨릴 수가 없다.**

    from supply import index_series, price_series

    idx = index_series(as_of="2020-06-30")          # 2020-06-29 까지만
    px  = price_series("005930", as_of="2020-06-30")
    index_series()                                  # TypeError — 빠뜨릴 수 없다

## 🔴 이 문은 **미래를 보는 판정을 하지 않는다**

정리매매·신규상장 같은 플래그는 *"이 뒤로 체결이 끊긴다"* 를 보고 정해진다. 즉 그
시점에는 알 수 없는 사실이다. 여기서 그걸 계산해 행을 덜어내면, 문이 지키려던 바로
그 경계를 문이 스스로 넘는다.

그 판정이 필요한 곳은 **학습 자료를 만드는 자리**다. 그래서 경로를 아예 갈랐다.

    supply.price_series(as_of=...)      그 시점에 알 수 있었던 것만. 플래그 없음
    supply.training_frame(...)          전 구간을 보고 정리매매·신규상장을 덜어낸다

이름에 `training` 이 박혀 있어서 예측 경로에서 부르면 눈에 띈다. 손잡이 하나로
켜고 끄게 두면 언젠가 켜진 채로 예측에 들어가고, **그때도 예외는 안 난다.**

## 반환은 `pandas.DataFrame` 이다

피처·모델·평가가 전부 pandas 로 짜인다. 여기서 딕셔너리 목록을 주면 부르는 쪽마다
`pd.DataFrame(rows)` 를 한 줄씩 쓰게 되고, **빈 결과일 때 칸이 없는 표**가 만들어져
`df["close"]` 가 `KeyError` 로 터진다. 그래서 여기서 한 번에 만든다 —
**행이 0개여도 칸은 언제나 있다.**

⚠️ 정렬 방향이 계약의 일부다(저장소와 같다). **과거 → 현재**. 부르는 쪽에서 다시
   정렬하지 않는다. 최근순으로 뒤집히면 이동평균·차분이 전부 틀린 값을 내는데
   예외가 나지 않는다.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import pandas as pd

from ingest.clients import krx_data as api
from ingest.store import krx_index, krx_store
from supply.clock import AsOf, as_bas_dd, known_at, latest_known_day, to_kst

#: 1차 프로젝트의 예측 대상.
TARGET_INDEX = api.TARGET_INDEX

#: 지수 표의 칸. **행이 0개여도 이 칸들은 있어야 한다.**
#: 빈 표에 칸이 없으면 `df["close"]` 가 KeyError 로 터지고, 부르는 쪽은 그걸
#: "자료가 이상하다" 가 아니라 "코드가 깨졌다" 로 읽는다.
#: ⚠️ 저장소가 칸을 늘리면 여기도 늘려야 한다 — `tests/test_supply_boundary.py` 가
#:    실제 반환과 이 목록을 맞춰 보고 어긋나면 실패한다.
INDEX_COLUMNS: Sequence[str] = (
    "bas_dd", "date", "index_name", "index_class", "open", "high", "low",
    "close", "change", "change_rate", "volume", "value", "market_cap",
)

#: 종목 표의 칸. 위와 같은 이유로 둔다.
PRICE_COLUMNS: Sequence[str] = (
    "bas_dd", "date", "code", "name", "market", "sector", "open", "high", "low",
    "close", "change", "change_rate", "volume", "value", "market_cap",
    "listed_shares",
)


def _cutoff(as_of: AsOf, end: Optional[AsOf]) -> str:
    """이 조회의 실제 상한(`YYYYMMDD`). `as_of` 와 부르는 쪽 `end` 중 **이른 쪽**.

    🔴 **표기를 맞추지 않고 비교하면 답이 뒤집힌다.**
    `min('2026-08-21', '20260825')` 는 `'2026-08-21'` 이다 — `'-'` 가 `'0'` 보다 작아서
    하이픈이 든 쪽이 언제나 이긴다. 그 값이 `bas_dd <= ?` 로 들어가면 8자리 날짜와
    비교가 성립하지 않아 **결과가 0행**이 되고, 받은 쪽은 "그 구간에 자료가 없구나"
    로 읽는다. 예외는 나지 않는다. 그래서 양쪽을 `as_bas_dd` 로 8자리로 강제한다.
    """
    limit = latest_known_day(as_of)
    asked = as_bas_dd(end)
    return min(asked, limit) if asked else limit


def to_frame(rows: List[Dict], columns: Sequence[str],
             with_known_at: bool = False) -> pd.DataFrame:
    """저장소가 준 딕셔너리 목록을 표로 만든다. **빈 결과에도 칸을 남긴다.**

    저장소는 `bas_dd` 를 빼고 `date`(`YYYY-MM-DD`)로 바꿔 준다. 그런데 거래일 달력과
    `common.corporate_actions` 는 `YYYYMMDD` 로 말한다. 부르는 쪽마다
    `.str.replace('-','')` 를 쓰게 두면 언젠가 한 곳에서 빠뜨리므로 여기서 되살린다.
    """
    cols = list(columns) + (["known_at"] if with_known_at else [])
    if not rows:
        return pd.DataFrame({c: pd.Series(dtype="object") for c in cols})

    frame = pd.DataFrame(rows)
    if "date" in frame.columns:
        frame.insert(0, "bas_dd", frame["date"].str.replace("-", "", regex=False))
    if with_known_at:
        frame["known_at"] = [
            known_at(b).isoformat(timespec="seconds") for b in frame["bas_dd"]
        ]
    # 저장소가 새 칸을 내보내도 버리지 않는다 — 아는 칸을 앞에, 나머지를 뒤에 붙인다.
    앞 = [c for c in cols if c in frame.columns]
    뒤 = [c for c in frame.columns if c not in 앞]
    return frame[앞 + 뒤].reset_index(drop=True)


def index_series(index_name: str = TARGET_INDEX, *, as_of: AsOf,
                 start: Optional[AsOf] = None, end: Optional[AsOf] = None,
                 days: Optional[int] = None,
                 with_known_at: bool = False) -> pd.DataFrame:
    """`as_of` 시점에 알 수 있었던 지수 시계열을 **과거 → 현재 순**으로.

    `as_of` 는 키워드 전용이고 **기본값이 없다.** 빠뜨리면 `TypeError` 로 즉시 터진다 —
    기본값이 "지금" 이면 빠뜨린 코드가 조용히 미래를 보게 된다.

    `with_known_at=True` 면 각 행에 **언제부터 알 수 있었는지**를 붙여 준다. 누수를
    의심할 때 눈으로 확인하는 용도다.
    """
    rows = krx_index.series(index_name, days=days, start=as_bas_dd(start),
                            end=_cutoff(as_of, end))
    return to_frame(rows, INDEX_COLUMNS, with_known_at)


def price_series(code: str, *, as_of: AsOf,
                 start: Optional[AsOf] = None, end: Optional[AsOf] = None,
                 days: Optional[int] = None,
                 with_known_at: bool = False) -> pd.DataFrame:
    """`as_of` 시점에 알 수 있었던 **종목** 시계열을 과거 → 현재 순으로.

    지수와 같은 계약이다. 다른 점은 종목에 **구멍이 있다**는 것뿐이다 — 거래정지,
    상장 전, 상장폐지 후. 그 구멍을 메우지 않는다. 행 번호로 "5거래일 뒤" 를 세면
    3개월 정지된 종목의 정지 전후 차이가 "5일 수익률" 로 둔갑하는데, 그건 부르는
    쪽이 **시장 거래일 달력**으로 거리를 재서 막아야 할 일이지 여기서 채울 일이 아니다.

    ⚠️ **플래그를 붙이지도, 행을 덜어내지도 않는다.** 정리매매·신규상장 판정은
       미래를 보고 정해진다. 그게 필요하면 `supply.training_frame` 을 쓴다.

    `days=None` 이 기본이다 — 라벨과 정리매매 판정이 종목 이력의 양 끝에 의존해서,
    최근 250일만 떼어 오면 잘린 자리가 곧 "체결 단절" 로 보인다.
    """
    rows = krx_store.series(code, days=days, start=as_bas_dd(start),
                            end=_cutoff(as_of, end))
    return to_frame(rows, PRICE_COLUMNS, with_known_at)


def as_of_bounds(as_of: AsOf) -> Dict[str, str]:
    """이 `as_of` 가 실제로 어디를 자르는지 알려 준다. 리포트와 사전등록에 적는 값이다.

    사전등록 문서에 *"홀드아웃은 언제부터"* 를 적을 때 사람이 손으로 계산하면 하루씩
    어긋난다. 코드가 답하게 한다.
    """
    moment = to_kst(as_of)
    return {
        "as_of": moment.isoformat(timespec="seconds"),
        "last_known_trading_day": latest_known_day(as_of),
    }
