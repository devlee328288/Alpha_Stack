"""종목코드 형식 (공통 계층)

**한국 종목코드는 여섯 자리 숫자가 아니다.** 5·6번째 자리에 영문이 올 수 있다.
우리 `daily_price` 920만 행을 실측하니 영문이 섞인 종목이 84종 · 56,190행(0.61%)이었고,
그중 `0009K0`(에임드바이오)·`0126Z0`(삼성에피스홀딩스)는 350종 핵심 유니버스 안에 있다.
1~4번째 자리는 3,677종 전부 숫자였다.

`^\\d{6}$` 로 가정한 코드가 저장소 곳곳에 흩어져 있었고, 그중 조회 분기 하나는 예외도
로그도 없이 *"그런 종목 없음"* 으로 위장하고 있었다. 그래서 형식을 이 한 곳에 모은다.

**두 개인 이유 — 조이는 이유가 자리마다 다르다.**

    STOCK_CODE_PATTERN   "이것이 종목코드가 맞는가"   → 좁게
    CODE_LIKE_PATTERN    "코드처럼 생겼는가"          → 넓게

앞의 것은 *남의 자료를 받는 정문*과 *바깥에 보낼 값을 만드는 자리*가 쓴다. 틀린 값을
들이거나 내보내면 그 순간 고장이므로 아는 형식만 통과시킨다.

뒤의 것은 *우리 DB 를 찾는 자리*가 쓴다. 여기서 하는 일은 검증이 아니라 **갈래 고르기**
— "코드로 찾을까 이름으로 찾을까" 다. 한글 종목명과만 구분되면 되고, 못 찾으면 어차피
`None` 이 나온다. 좁게 잡으면 정상 종목을 이름 검색으로 흘려보내 조용히 놓친다.

⚠️ **규격 JSON 이 정본이다.** `ingest/inbox/schemas/*.json` 의 `code.pattern` 이 반입
정문의 계약이고 이 파일은 그 사본이다. 두 벌로 두면 한쪽만 고쳐진 채 오래 가므로,
`tests/test_common_codes.py` 가 둘이 같은지 매번 검사한다. 규격을 고치면 테스트가 먼저
깨진다.
"""

from __future__ import annotations

import re
from typing import Final

#: 종목코드가 맞는가 — 앞 4자리는 숫자, 5·6번째는 숫자 또는 대문자 영문.
#: 반입 규격(`ohlcv_stock`·`news`·`financial` 의 `code.pattern`)과 같은 값이어야 한다.
STOCK_CODE_PATTERN: Final = re.compile(r"^[0-9]{4}[0-9A-Z]{2}$")

#: 코드처럼 생겼는가 — 조회에서 코드 검색과 이름 검색을 가르는 데만 쓴다.
#: `A05930` 같은 HTS 표기까지 코드로 받아 주는 편이 안전하다. 없으면 `None` 이 나올 뿐이다.
CODE_LIKE_PATTERN: Final = re.compile(r"^[0-9A-Z]{6}$")


def is_stock_code(value: str) -> bool:
    """종목코드 형식인지 판정한다. 소문자로 와도 같은 것으로 본다.

    반입 정제기가 `upper` 를 먼저 걸지만, 이 함수를 직접 부르는 자리도 있으므로
    여기서도 올려 준다. 판정만 하고 값을 바꾸지는 않는다.
    """
    return bool(STOCK_CODE_PATTERN.fullmatch((value or "").strip().upper()))


def looks_like_code(value: str) -> bool:
    """코드 검색으로 갈지 이름 검색으로 갈지 가른다.

    한글 종목명('삼성전자')과 구분하는 것이 목적이므로 `is_stock_code` 보다 넓다.
    """
    return bool(CODE_LIKE_PATTERN.fullmatch((value or "").strip().upper()))
