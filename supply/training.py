"""학습 자료를 만드는 자리 — **여기서만 미래를 본다.**

## 왜 문을 둘로 갈랐나

`supply.price_series` 는 *"그 시점에 알 수 있었던 것"* 만 낸다. 그게 문의 전부다.

그런데 학습 자료를 만들 때는 그것만으로 부족하다. 정리매매 구간을 남겨 두면 모델은
*"5일에 -90%"* 라는 존재하지 않는 신호를 배우고, 백테스트는 그걸 피하거나 반등을
사서 **거대한 가짜 수익**을 만든다. 실제로는 유동성이 없어 그 가격에 체결되지 않는다.

문제는 **그 판정이 미래를 본다**는 것이다. *"이 뒤로 체결이 끊긴다"* 는 그 시점에는
알 수 없다. 그래서 손잡이 하나로 켜고 끄게 두면 안 된다 — 언젠가 켜진 채로 예측
경로에 들어가고, **그때도 예외는 나지 않는다.** 경로를 이름으로 가른다.

    supply.price_series(as_of=...)   그 시점에 알 수 있었던 것만. 플래그 없음
    supply.training_frame(...)       ← 여기. 전 구간을 보고 덜어낸다

`training` 이 이름에 박혀 있어서 예측 코드에서 부르면 리뷰에 걸린다.

## 이건 누수가 아니라 **표본 선택**이다

둘은 다르다.

    누수      그 시점에 몰랐을 값을 **피처나 라벨에 넣는** 것
    표본 선택 어떤 행을 **학습 자료에 넣을지 말지**를 고르는 것

정리매매 행을 빼는 것은 후자다. 모델은 그 행을 본 적이 없고, 예측할 때도 그 판정을
쓰지 않는다. 다만 표본 선택도 공짜는 아니다 — 아래 ⚠️ 를 읽고 쓴다.

⚠️ **곧 죽을 종목을 골라서 빼면 생존 편향이 되돌아온다.** 우리가 빼는 것은 종목이
   아니라 **구간**이다. 소멸 종목 910개는 전부 남기고 그 마지막 10체결일만 뺀다.
   백테스트에서 "상장폐지로 잃는 돈" 은 그 앞 구간에 이미 들어 있다.

⚠️ **덜어낸 양을 반드시 보고한다.** 조용히 빠지면 표본 수가 왜 줄었는지 아무도
   모른다. `training_frame` 은 `attrs["dropped"]` 에 무엇을 얼마나 뺐는지 남긴다.

실측 (2026-08-29 · `daily_price` 9,209,812행)

    정리매매        20,763행 (0.23%)
    신규상장 첫날    1,716행 (0.02%)
    거래정지 표시행 283,468행 (3.08%)   ← 체결이 없어 애초에 못 쓰는 행

사용법
------
    from evaluation.horizon import HOLDOUT_START
    from supply import training_frame, training_frames

    df = training_frame("005930", holdout_start=HOLDOUT_START)   # 개발구간만
    df = training_frame("005930", holdout_start=None)            # 전 구간
    df.attrs["dropped"]        # {'liquidation': 12, 'first_listing': 1, ...}

    # 여러 종목이면 이쪽. 시장 달력을 한 번만 만든다 (실측 10배)
    for code, frame in training_frames(codes, holdout_start=HOLDOUT_START):
        ...
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, FrozenSet, Iterable, Iterator, Mapping, Optional, Tuple

import pandas as pd

from common.corporate_actions import (
    LIQUIDATION_DAYS,
    SUSPENSION_GAP_DAYS,
    flag_series,
    is_traded,
    market_calendar_index,
)
from ingest.store import krx_store
from supply.clock import as_bas_dd
from supply.market import PRICE_COLUMNS, to_frame


@dataclass(frozen=True)
class MarketContext:
    """플래그를 매기는 데 필요한 **시장 전체의 사실들**. 한 번 만들어 돌려 쓴다.

    🔴 **이걸 종목마다 다시 만들면 학습 자료 준비가 한 시간을 넘는다.**
    거래일 달력은 `SELECT DISTINCT bas_dd FROM daily_price` 라 920만 행을 훑는다.
    실측 종목당 1.1초 × 3,677종목 = **67분**. 한 번만 만들면 3,677종목 전체가
    달력 한 번 + 종목별 조회로 끝난다.

    ⚠️ **몰래 캐시하지 않는다.** 모듈 전역에 담아 두면 수집 배치가 새 행을 넣은 뒤에도
       낡은 달력을 계속 쓰게 되고, **그래도 예외는 나지 않는다** — 새 거래일이 달력에
       없어서 그 날짜가 조용히 빠질 뿐이다. 그래서 부르는 쪽이 언제 만들지 정한다.
    """

    calendar_index: Mapping[str, int]
    market_last_index: int
    collect_start: str
    listed_codes: FrozenSet[str]


def market_context() -> MarketContext:
    """`MarketContext` 를 한 번 만든다. 실측 약 1초(920만 행 · 4,097거래일)."""
    with krx_store.connect() as conn:
        index, last_index = market_calendar_index(conn)
        last_day = conn.execute("SELECT MAX(bas_dd) FROM daily_price").fetchone()[0]
        listed = frozenset(
            r[0] for r in conn.execute(
                "SELECT code FROM daily_price WHERE bas_dd = ?", (last_day,))
        )
        collect_start = conn.execute(
            "SELECT MIN(bas_dd) FROM daily_price").fetchone()[0]
    return MarketContext(index, last_index, collect_start, listed)


def training_frame(code: str, *,
                   holdout_start: Optional[str],
                   context: Optional[MarketContext] = None,
                   drop_liquidation: bool = True,
                   drop_first_listing: bool = True,
                   drop_halted: bool = True,
                   liquidation_days: int = LIQUIDATION_DAYS,
                   gap_days: int = SUSPENSION_GAP_DAYS) -> pd.DataFrame:
    """한 종목의 **학습용** 시계열. 가격이 시장 수익률이 아닌 구간을 덜어낸다.

    `holdout_start` 는 키워드 전용이고 **기본값이 없다.** 그 날짜 **이전**만 남기고,
    `None` 을 명시하면 자르지 않는다(최종 평가에서 홀드아웃까지 쓸 때).

    🔴 **기본값을 두지 않는 이유는 `as_of` 와 같다.** 기본이 "자르지 않음" 이면
       빠뜨려도 돌아가고, 빠뜨린 그 한 번이 봉인 구간을 연다. 빠뜨리면 `TypeError`
       로 터지게 만들어야 막힌다. `None` 을 적는 것은 **의도를 적는 것**이다.

       값은 `evaluation.horizon.HOLDOUT_START` 를 가져다 넣는다. 여기서 그 상수를
       import 하지 않는 이유는 계층 방향이다 — `supply → evaluation` 은 화살표를
       거꾸로 뒤집고, 나중에 evaluation 이 supply 를 부르는 순간 순환이 된다.

    무엇을 왜 빼나
    --------------
    `drop_liquidation`   정리매매 구간. 가격제한폭이 적용되지 않아 하루 -90% 가 난다
    `drop_first_listing` 신규상장 첫날. 등락률이 전일종가가 아니라 **공모가** 기준이다
    `drop_halted`        거래정지 표시행. 시·고·저가가 0 이라 체결을 가정할 수 없다

    자본변동(액면분할·병합·감자)은 **빼지 않는다.** 그 날의 가격은 실제 체결가이고,
    문제는 그 앞뒤를 이어 붙일 때 생긴다. 이어 붙이는 쪽(라벨 계산)이
    `common.corporate_actions.is_basis_adjusted` 로 판단할 일이지 여기서 버릴 일이 아니다.

    ⚠️ **판정에는 전 구간이 필요하다.** 그래서 자르는 것은 판정이 **끝난 뒤**다.
       먼저 자르면 잘린 자리가 곧 "체결 단절" 로 보여 없는 정리매매가 생긴다.
    """
    rows = krx_store.series(code, days=None)          # 전 구간. 자르지 않는다
    if not rows:
        frame = to_frame([], PRICE_COLUMNS, False)
        frame.attrs["dropped"] = {}
        frame.attrs["input_rows"] = 0
        return frame

    ctx = context or market_context()

    # 저장소는 `bas_dd` 를 `date` 로 바꿔 주는데 판정 함수들은 `bas_dd` 로 말한다.
    ordered = [{**r, "bas_dd": r["date"].replace("-", "")} for r in rows]
    flags = flag_series(ordered, calendar_index=ctx.calendar_index,
                        market_last_index=ctx.market_last_index,
                        still_listed=code in ctx.listed_codes,
                        collect_start=ctx.collect_start,
                        liquidation_days=liquidation_days, gap_days=gap_days)

    dropped: Dict[str, int] = {}
    kept = []
    끝 = as_bas_dd(holdout_start)
    for row, flag in zip(ordered, flags, strict=True):
        if 끝 and row["bas_dd"] >= 끝:
            dropped["holdout"] = dropped.get("holdout", 0) + 1
            continue
        why = None
        if drop_liquidation and flag.liquidation:
            why = "liquidation"
        elif drop_first_listing and flag.first_listing:
            why = "first_listing"
        elif drop_halted and not is_traded(row):
            why = "halted"
        if why:
            dropped[why] = dropped.get(why, 0) + 1
            continue
        kept.append({k: v for k, v in row.items() if k != "bas_dd"})

    frame = to_frame(kept, PRICE_COLUMNS, False)
    # 조용히 빠지면 표본 수가 왜 줄었는지 아무도 모른다. 세서 표에 붙여 둔다.
    frame.attrs["dropped"] = dropped
    frame.attrs["input_rows"] = len(ordered)
    return frame


def training_frames(codes: Iterable[str], *, holdout_start: Optional[str],
                    **kwargs) -> Iterator[Tuple[str, pd.DataFrame]]:
    """여러 종목을 돌 때 **시장 사실을 한 번만** 만든다. `(종목코드, 표)` 를 흘려보낸다.

    종목마다 `training_frame` 을 그냥 부르면 거래일 달력을 매번 다시 만든다.
    실측 2026-08-29 (920만 행 · 3,677종목 · 30종목 표본으로 잰 값)

        context 없이     종목당 2.71초   → 전 종목 166분
        context 재사용   종목당 0.266초  → 전 종목 16.3분  (준비 0.77초 1회)

    남은 0.266초는 종목별 `SELECT *` 가 인덱스 밖 칸을 읽으러 가는 비용이다.
    더 줄이려면 전 종목을 한 번에 흘려보내야 하는데(`scripts/check_data.py` 가 그렇게
    해서 98.6초다), 그건 종목 단위 API 와 모양이 다르다. **필요해지면 그때 만든다** —
    지금은 학습 자료를 만드는 일이 하루 한 번이라 16분이 문제가 되지 않는다.

    ⚠️ 표를 한꺼번에 모아 두지 않는다(제너레이터다). 920만 행을 전부 DataFrame 으로
       들고 있으면 수 GB 다. 부르는 쪽이 한 종목씩 처리하고 버리게 한다.
    """
    ctx = kwargs.pop("context", None) or market_context()
    for code in codes:
        yield code, training_frame(code, holdout_start=holdout_start,
                                   context=ctx, **kwargs)


#: 반출본에 싣는 기업행위 판정 세 칸. 이 순서로 카드 표에 나간다.
CORPORATE_ACTION_COLUMNS = ("is_liquidation", "is_halted", "is_first_listing")


def attach_corporate_action_flags(frame: pd.DataFrame, *,
                                  context: Optional[MarketContext] = None,
                                  ) -> pd.DataFrame:
    """시세 표에 **가격이 시장 수익률이 아닌 이유** 세 칸을 붙인다.

    `training_frame` 은 이 판정으로 행을 *덜어내고*, 이 함수는 *표시만* 한다. 둘이
    같은 함수(`flag_series`)를 쓰므로 판정이 갈리지 않는다.

    ## 왜 이 함수가 필요한가

    지금까지 작은 벌(`stocks_sample30_train_dev.csv`)은 이 셋을 덜어낸 표본이고
    큰 벌(`full/daily_price_dev.parquet`)은 안 덜어낸 표본인데, **칸 구성만 보고는
    그것을 알 수 없었다.** 큰 벌로 학습하면 작은 벌과 결과가 갈리는데 아무 경고도
    없다. 판정을 파일에 실어 그 차이를 눈에 보이게 한다.

    실측 2026-09-09 · 개발구간 7,888,945행:

        is_liquidation      17,973  (0.228%)   정리매매 — 체결이 끊기기 직전 10체결일
        is_halted          231,808  (2.938%)   거래정지 — 체결이 없던 행
        is_first_listing     1,501  (0.019%)   신규상장 첫 거래일
        ── 셋 중 하나라도  251,282  (3.185%)   겹침 0

    ## 🔴 피처로 쓰면 안 된다

    `is_liquidation` 은 *"이 뒤로 체결이 끊긴다"* 를 보고 매긴다. 그 시점에는 알 수
    없는 사실이라 예측 시점의 피처로 넣으면 곧 미래참조다. **표본을 고르는 용도**로만
    쓴다. 그래서 이 함수는 `as_of` 를 받지 않는다 — `supply` 의 시점 규칙 안에서
    계산되면 안 되는 값이고, 반출 스크립트가 명시적으로 부른다.

    ## 왜 DB 를 다시 읽나

    `flag_series` 는 **그 종목의 전 구간**을 요구한다. 정리매매는 "체결이 끊기기 직전"
    이라 이력의 끝을 봐야 하고, 신규상장은 첫 행을 봐야 한다. 개발구간만 잘라 넘기면
    개발구간 끝에서 잘린 종목이 전부 정리매매로 오판된다. 그래서 전 구간을 읽어
    판정한 뒤 입력 행에만 붙인다.
    """
    for col in ("bas_dd", "code"):
        if col not in frame.columns:
            raise ValueError(f"기업행위 판정을 붙이려면 '{col}' 칸이 있어야 한다.")

    out = frame.copy()
    if out.empty:
        for col in CORPORATE_ACTION_COLUMNS:
            out[col] = pd.Series([], dtype="bool")
        return out

    ctx = context or market_context()
    with krx_store.connect() as conn:
        conn.execute("PRAGMA cache_size = -1000000")
        전구간 = pd.read_sql_query(
            "SELECT bas_dd, code, open, high, low, volume, listed_shares "
            "FROM daily_price ORDER BY code, bas_dd", conn)

    키, 판정 = [], []
    for code, g in 전구간.groupby("code", sort=False):
        rows = g.to_dict("records")
        flags = flag_series(rows, calendar_index=ctx.calendar_index,
                            market_last_index=ctx.market_last_index,
                            still_listed=code in ctx.listed_codes,
                            collect_start=ctx.collect_start)
        for r, f in zip(rows, flags, strict=True):
            키.append((str(r["bas_dd"]), str(code)))
            # 🔴 `is_halted` 는 `flag.halt_resume`(직전이 정지행) 이 아니라 **이 행에
            #    체결이 있었나** 다. `training_frame` 이 덜어내는 기준과 같아야 큰 벌과
            #    작은 벌이 같은 표본이 된다.
            판정.append((f.liquidation, not is_traded(r), f.first_listing))

    표 = pd.DataFrame(판정, columns=list(CORPORATE_ACTION_COLUMNS))
    표["bas_dd"] = [k[0] for k in 키]
    표["code"] = [k[1] for k in 키]

    left = out[["bas_dd", "code"]].copy()
    left["bas_dd"] = left["bas_dd"].astype(str)
    left["code"] = left["code"].astype(str)
    left["_order"] = range(len(left))
    merged = left.merge(표, on=["bas_dd", "code"], how="left")
    if len(merged) != len(left):
        raise ValueError(
            f"기업행위 판정 조인이 행을 늘렸다: {len(left):,} → {len(merged):,}."
        )
    빈것 = int(merged["is_liquidation"].isna().sum())
    if 빈것:
        raise ValueError(
            f"판정을 못 붙인 행이 {빈것:,} 있다 — 입력 표에 `daily_price` 밖의 행이 있다."
        )
    merged = merged.sort_values("_order")
    for col in CORPORATE_ACTION_COLUMNS:
        out[col] = merged[col].to_numpy().astype(bool)
    return out
