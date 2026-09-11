"""공급 계층 — 자료를 쓰는 쪽이 지나야 하는 **유일한 문**.

피처·모델·평가는 저장소(`ingest/`)를 직접 부르지 않고 여기를 지난다. 이유는 하나다.

    저장소는 **표에 있는 것을 전부** 준다. 표에는 오늘까지 들어 있다.
    그걸 2020년 폴드 학습에 그대로 쓰면 미래가 섞이고, **예외는 나지 않는다.**

이 문을 지나려면 `as_of` 를 내야 한다. 기본값이 없으므로 **빠뜨릴 수가 없다.**

    from supply import index_series

    rows = index_series(as_of="2020-06-30")

중기·장기 원천(재무·거시·공시 텍스트)도 같은 문을 지난다 — 2026-09-11 에 열었다.
행 T 에 보이는 조건은 셋 다 `known_at <= T` 하나다.

    from supply import attach_financial, attach_macro, attach_text

    panel = attach_financial(prices, as_of="2024-08-31")   # 접수일 다음 거래일 · 정정본은 정정일
    panel = attach_macro(panel, as_of="2024-08-31")        # 발표된 날부터 · 순환변동치 제외
    panel = attach_text(panel, as_of="2024-08-31")         # 그날 새로 보인 공시 건수·확률

계층 방향
--------
    ingest/(수집·저장)  →  supply/(이 문)  →  features/ · models/ · evaluation/

화살표가 한 방향이라 순환이 생기지 않고, 무엇보다 **미래가 역류하지 못한다.**
`ingest` 와 `supply` 는 `as_of` 를 다루는 책임이 다르다.

| | `ingest/` | `supply/` (여기) |
|---|---|---|
| 관심사 | 받은 것을 **빠짐없이** 쌓는다 | 그 시점에 **알 수 있었던 것만** 낸다 |
| `as_of` | 모른다 (알 필요가 없다) | **필수로 받는다** |
| 부르는 쪽 | 수집 스크립트·화면 | 피처·모델·평가 |

⚠️ `ingest/` 는 내부 계층이다. `features/`·`models/`·`evaluation/` 에서 `ingest` 를
   import 하면 **테스트가 실패한다**(`tests/test_supply_boundary.py`). 규칙을 문서에만
   적어 두면 급할 때 지나가고, 지나간 코드는 티가 안 난다.
"""

from supply.clock import (
    DART_KNOWN_RULE,
    AsOfRequired,
    as_bas_dd,
    dart_known_at,
    is_known,
    known_at,
    latest_known_day,
    row_day,
    to_kst,
)
from supply.financial import (
    FINANCIAL_ACCOUNTS,
    FINANCIAL_COLUMNS,
    attach_financial,
    financial_as_of,
    financial_lines_as_of,
)
from supply.macro import (
    DEFAULT_INDICATORS,
    REVISED_INDICATORS,
    attach_macro,
    macro_as_of,
    macro_history,
)
from supply.market import (
    INDEX_COLUMNS,
    PRICE_COLUMNS,
    TARGET_INDEX,
    as_of_bounds,
    index_series,
    price_series,
    to_frame,
)
from supply.text import TEXT_DAILY_COLUMNS, attach_text, text_as_of
from supply.training import MarketContext, market_context, training_frame, training_frames
from supply.universe import (
    UNIVERSE_COLUMNS,
    common_stocks,
    coverage,
    excluded,
    top_by_market_cap,
)

__all__ = [
    # 시각·경계
    "AsOfRequired",
    "DART_KNOWN_RULE",
    "as_bas_dd",
    "as_of_bounds",
    "dart_known_at",
    "is_known",
    "known_at",
    "latest_known_day",
    "row_day",
    "to_kst",
    # 예측 경로 — as_of 를 내고 그 시점에 알 수 있었던 것만 받는다
    "INDEX_COLUMNS",
    "PRICE_COLUMNS",
    "TARGET_INDEX",
    "index_series",
    "price_series",
    "to_frame",
    # 유니버스 — 그날 무엇을 후보로 삼을 수 있었나 (보통주 판별은 KRX 정본으로)
    "UNIVERSE_COLUMNS",
    "common_stocks",
    "coverage",
    "excluded",
    "top_by_market_cap",
    # 중기·장기 원천 — 재무(접수일) · 거시(발표일) · 공시 텍스트(접수일)
    "DEFAULT_INDICATORS",
    "FINANCIAL_ACCOUNTS",
    "FINANCIAL_COLUMNS",
    "REVISED_INDICATORS",
    "TEXT_DAILY_COLUMNS",
    "attach_financial",
    "attach_macro",
    "attach_text",
    "financial_as_of",
    "financial_lines_as_of",
    "macro_as_of",
    "macro_history",
    "text_as_of",
    # 학습 경로 — 여기서만 미래를 본다 (supply/training.py 를 읽고 쓴다)
    "MarketContext",
    "market_context",
    "training_frame",
    "training_frames",
]
