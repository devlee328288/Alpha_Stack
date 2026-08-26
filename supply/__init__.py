"""공급 계층 — 자료를 쓰는 쪽이 지나야 하는 **유일한 문**.

피처·모델·평가는 저장소(`ingest/`)를 직접 부르지 않고 여기를 지난다. 이유는 하나다.

    저장소는 **표에 있는 것을 전부** 준다. 표에는 오늘까지 들어 있다.
    그걸 2020년 폴드 학습에 그대로 쓰면 미래가 섞이고, **예외는 나지 않는다.**

이 문을 지나려면 `as_of` 를 내야 한다. 기본값이 없으므로 **빠뜨릴 수가 없다.**

    from supply import index_series

    rows = index_series(as_of="2020-06-30")

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
    AsOfRequired,
    is_known,
    known_at,
    latest_known_day,
    to_kst,
)
from supply.market import TARGET_INDEX, as_of_bounds, index_series

__all__ = [
    "AsOfRequired",
    "TARGET_INDEX",
    "as_of_bounds",
    "index_series",
    "is_known",
    "known_at",
    "latest_known_day",
    "to_kst",
]
