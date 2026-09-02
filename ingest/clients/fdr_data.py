"""FinanceDataReader 수정주가 호출 (외부 연동 계층)

`daily_price` 의 `open`·`high`·`low`·`close` 는 KRX 원문 그대로라 **액면분할이 조정되지
않은 원가격**이다. 분할일에 가격이 그대로 뚝 떨어지므로 수익률로 계산하면 삼성전자
2018-05-04 가 **-98.04%** 로 읽힌다 (실제 그날 등락은 -2.08%). 그 조정을 붙이려고
FinanceDataReader(MIT · 0.9.202)에서 수정 OHLC 를 받는다.

실측으로 확인한 것 (2026-09-02)
-------------------------------
**① 최근 3,000거래일만 준다 — 이게 이 모듈에서 제일 중요한 사실이다.**

FDR 의 한국 주식 경로는 네이버 `fchart` 이고, 그 서버가 3,000건에서 자른다.
FDR 코드는 이미 `count=6000` 을 보내고 있는데도 그렇다 — 직접 확인했다:

    count=3000 → 3000건 · 첫날 20140613
    count=6000 → 3000건 · 첫날 20140613     ← 우리 요청이 아니라 서버가 자른다
    count=9000 → 3000건 · 첫날 20140613

`start` 를 2010-01-01 로 줘도 그 앞은 **0행**이다. pykrx 의 `adjusted=True` 도 같은
네이버 경로라 결과가 같다. 우리 달력은 4,102일(20100104~)이므로

    20100104 ~ 20140612 · 1,103거래일 · 2,146,042행 (전체의 23.3%)

이 비고, 홀드아웃이 20240901 이라 **그 구멍은 전부 학습구간 안**이다. 그래서 이 모듈은
받을 수 있는 데까지만 받고, 그 앞은 `ingest.store.adj_price` 가 조정계수로 이어 붙인다.

**② 거래정지일에는 시·고·저가 `0` 으로 온다.** 종가만 직전 값을 붙들고 있다.

    2018-04-27  53380  53639  52440  53000    606216   ← 평상일
    2018-04-30      0      0      0  53000         0   ← 정지 (분할 전 주권 교체)
    2018-05-04  53000  53900  51800  51900  39565391   ← 재개일 = 분할일

`0` 을 가격으로 실으면 그 날 수익률이 -100% 가 되고, 고·저가 검사도 통과해 버린다
(0 ≤ 0 ≤ 0 은 참이다). **`None` 으로 바꿔서 넘긴다** — 없는 값을 0 으로 적지 않는다.

**③ 이벤트가 없으면 원가격과 완전히 같다.** 무이벤트 종목(SK하이닉스·현대차) 각
2,863일에서 원주가/FDR 비율이 최소·최대 모두 1.0000 이었다. 즉 이 값은 "조정이
필요한 곳에만 다른" 계열이라, 원가격과의 비율이 곧 조정 배율이다.

**④ 상장폐지 종목도 준다.** 소멸 912종 중 30종을 표본으로 조회해 전부 정상
반환됐다(생존 30종도 30/30). 생존 편향 구멍은 표본상 없다.

⚠️ **재현성** — 수정주가는 후방조정이라 **새 분할이 하나 생기면 과거 값이 전부 바뀐다.**
   이건 FDR 의 문제가 아니라 후방조정의 성질이다. 그래서 원 가격 칸을 덮지 않고
   `adj_*` 를 따로 두며, 재계산 시점을 대장(`collect_log`)에 남긴다.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import FinanceDataReader as fdr
import pandas as pd

#: FDR(네이버 fchart)이 돌려주는 최대 거래일 수. **서버가 정한 값이라 늘릴 수 없다.**
#: 이 값을 키우고 싶어지면 위 실측표를 다시 보라 — `count` 를 올려도 3,000 에서 잘린다.
MAX_SESSIONS = 3000

#: 이 모듈이 `daily_price` 에 채우는 칸 ← FDR 이 주는 칸.
#: `Volume` 은 받지 않는다 — 거래량은 조정 대상이 아니고 원문이 이미 있다.
COLUMN_MAP = {
    "adj_open": "Open",
    "adj_high": "High",
    "adj_low": "Low",
    "adj_close": "Close",
}

#: 이 행의 수정값이 어디서 왔는지 표시하는 값. `daily_price.adj_source` 에 그대로 들어간다.
SOURCE_FDR = "fdr"


class FdrUnavailable(RuntimeError):
    """FDR 이 그 종목을 주지 못했다. 무엇을 해야 하는지까지 메시지에 담는다."""


#: 한 종목의 수정 OHLC — `{YYYYMMDD: {칸: 값}}`. 값이 `None` 이면 그 날 그 가격이 없다.
Adjusted = Dict[str, Dict[str, Optional[float]]]


def fetch_adjusted(code: str, *, start: str = "2010-01-01") -> Adjusted:
    """한 종목의 수정 OHLC 를 `{YYYYMMDD: {adj_open: ..., ...}}` 로 돌려준다.

    빈 결과는 **예외가 아니다** — 빈 딕셔너리다. 신규 상장이라 아직 자료가 없거나
    아주 오래전에 사라진 종목이 실제로 있고, 그때마다 예외를 던지면 전 종목 적재가
    한 종목 때문에 멈춘다. 부르는 쪽이 "몇 종이 비었나" 를 세어 판단한다.

    ⚠️ 돌려주는 날짜는 **FDR 이 준 날짜 그대로**다. 우리 달력과 맞는지는 여기서 보지
       않는다 — 대조는 적재하는 쪽(`ingest.store.adj_price`)의 일이고, 여기서 미리
       걸러 내면 "FDR 이 우리가 모르는 날을 줬다" 는 사실 자체가 사라진다.
    """
    try:
        frame = fdr.DataReader(code, start)
    except Exception as exc:                       # noqa: BLE001 — 되살려 던진다
        raise FdrUnavailable(
            f"FDR 이 {code} 를 주지 못했다: {exc}\n"
            "  왜 세우나: 네트워크 실패와 '자료 없음' 은 다른 사실이다. 앞엣것을 뒤엣것으로\n"
            "             적으면 그 종목은 영영 안 채워진 채 정상으로 보인다.\n"
            "  할 일: 잠시 뒤 다시 부르거나, 그 종목을 건너뛴 사실을 대장에 남긴다."
        ) from exc

    if frame is None or frame.empty:
        return {}

    return _normalize(frame)


def _normalize(frame: pd.DataFrame) -> Adjusted:
    """FDR 프레임을 `{YYYYMMDD: {칸: 값}}` 으로 바꾼다. 정지일 `0` 은 `None` 이 된다.

    분리해 둔 이유는 **네트워크 없이 검사할 수 있게** 하기 위해서다. 정지일 처리는
    조용히 틀리는 종류라 테스트가 반드시 있어야 하는데, 실제 호출로 재현하려면
    특정 종목·특정 날짜에 의존하게 된다.
    """
    out: Adjusted = {}
    for stamp, row in frame.iterrows():
        bas_dd = pd.Timestamp(stamp).strftime("%Y%m%d")
        values: Dict[str, Optional[float]] = {}
        for column, source in COLUMN_MAP.items():
            raw = row.get(source)
            # 정지일의 0 과 결측을 **둘 다 None 으로** 만든다. 0 을 그대로 실으면
            # 그 날 수익률이 -100% 가 되고, 고·저 검사(0 ≤ 0 ≤ 0)도 통과해 버린다.
            if raw is None or pd.isna(raw) or float(raw) <= 0:
                values[column] = None
            else:
                values[column] = float(raw)
        out[bas_dd] = values
    return out


def coverage_span(adjusted: Adjusted) -> Optional[tuple]:
    """받은 자료가 덮는 `(첫날, 마지막날)`. 비었으면 `None`.

    적재하는 쪽이 *"어디까지가 FDR 이고 어디부터 이어 붙여야 하나"* 를 정할 때 쓴다.
    **종가가 있는 날만** 센다 — 정지일은 시·고·저가 전부 `None` 이라 구간의 끝으로
    삼으면 앵커를 정지일에 놓게 된다.
    """
    days = sorted(d for d, v in adjusted.items() if v.get("adj_close") is not None)
    return (days[0], days[-1]) if days else None


def fetch_many(codes: List[str], *, start: str = "2010-01-01",
               on_progress=None) -> Dict[str, Adjusted]:
    """여러 종목을 차례로 받는다. 실패한 종목은 **비워 두고 계속한다.**

    실측에서 종목당 0.10초였다 — 3,677종이면 약 6분이다. 병렬로 밀지 않는 이유는
    남의 서버이고, 6분은 기다릴 만한 시간이기 때문이다.

    `on_progress(code, 행수, 오류)` 를 주면 종목마다 부른다. 오류는 문자열이거나 `None`.
    """
    out: Dict[str, Adjusted] = {}
    for code in codes:
        error = None
        try:
            out[code] = fetch_adjusted(code, start=start)
        except FdrUnavailable as exc:
            out[code] = {}
            error = str(exc).splitlines()[0]
        if on_progress is not None:
            on_progress(code, len(out[code]), error)
    return out
