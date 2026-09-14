# erd — 버전 목록

> 최신 버전 폴더가 정본입니다. 옛 버전은 **발전 과정의 기록**이라 지우지 않습니다.
> 규약 전문은 [docs/README.md](../README.md).

| 버전 | 날짜 | 무엇이 담겼나 |
|---|---|---|
| [**version2.7**](version2.7/) ⭐ | 2026-09-11 | **표 23개 · v16 그대로** — 스키마는 안 바뀌고 [§1.5 **정문 지도**](version2.7/ERD.md)를 더했다(표마다 여는 문 · 행 T 에 보이는 조건 · 함정). 🔴 **시점 칸이 있다고 시점이 지켜지는 것이 아니다** — `stock_base_info`·업종 스냅샷은 `known_at` 이 있었는데 붙이는 코드가 날짜로 조인해 하루 먼저 봤다(5,300 · 6,276 · 3,677행) · DB 밖의 기록 `reports/unseal.log`(ADR 0004 형식 · #240) ([변경사항](version2.7/변경사항.md)) |
| [version2.6](version2.6/) | 2026-09-10 | **표 23개 · v16** — [`corporate_action`](version2.6/ERD.md) 기업행위 사건 44,290건 · 결과(`adj_*`)가 아니라 사건 · chain 판정과 대조해 어긋남 8 ([변경사항](version2.6/변경사항.md)) |
| [version2.5](version2.5/) | 2026-09-09 · 09-10 | **표 22개 · v14·v15** — `dividend` 71,681행(🔴 조인 축은 `ex_date`) · 총수익 3칸 `adj_close_tr` ([변경사항](version2.5/변경사항.md)) |
| [version2.4](version2.4/) | 2026-09-07 | **표 21개 · v13** — 표는 안 늘고 **기본키 하나가 바뀌었다**: [`index_price` PK `(bas_dd, index_name, index_class)`](version2.4/ERD.md)(KOSPI·KOSDAQ 이 같은 이름의 업종지수를 각각 가져 시장 없이는 한쪽이 덮는다 · 09-07 아침 17종 40,324행을 잃고 복구 · 🔴 **행 수는 오히려 늘어** 축별로 세고서야 보였다 · 가드 → 기본키 · PR #157) · 행 수 09-07 갱신 · v14 후보(#159 `source` 칸 — 권고는 별도 파일이라 안 생길 수도) ([변경사항](version2.4/변경사항.md)) |
| [version2.3](version2.3/) | 2026-09-04 | **표 21개 · v12** — v11 [`stock_base_info`](version2.3/ERD.md)(우선주 판별의 정본 · 🔴 액면가로 감자를 판정하면 안 보인다 · **그날의 사실**이라 PK 에 날짜) · v12 [`text_signal`](version2.3/ERD.md)(고유 제목마다 · 리비전 · 🔴 **시점 칸이 없다** — 시점은 반출 때 · 확률 합 1 을 `CHECK` 가 막는다) · `dart_disclosure` 0 → **1,555,556행**(날짜로 받는다 · `page_count` 상한 100) ([변경사항](version2.3/변경사항.md)) |
| [version2.2](version2.2/) | 2026-09-03 | **표 19개** — v10 으로 신원 표 둘이 늘었다. [`stock_identity`(코드↔법인등록번호↔ISIN) · `corp_profile`(상장·폐지일·감사의견)](version2.2/ERD.md) · 🔴 **PK 가 `crno` 하나가 아니다**(법인 5곳 → 82행) · `basDt` 목록은 시점 목록이 아니다(4년 뒤 상장 33종) · `known_at` 이 한 표는 계산값 한 표는 관측값 ([변경사항](version2.2/변경사항.md)) |
| [version2.1](version2.1/) | 2026-09-02 | **표 17개** — v9 수정주가 4칸(`adj_*`)·`adj_source` 와 실측 거래일 달력(`trading_calendar`). FDR 이 최근 3,000거래일만 주고 그 경계가 **종목마다 다르다**는 것을 칸으로 남겼다 (이슈 #51) |
| [version2.0](version2.0/) | 2026-09-02 | **설계도가 아니라 실물** — `krx_cache.db` v8 · 표 16개를 `PRAGMA` 전수 실측으로 그렸다. 자료 5표 + 운영 11표 · 마이그레이션 v1~v8 이력 · 함정 칸(수정주가 아님 · `account_detail` · `known_at` 계산값) |
| [version1.0](version1.0/) | 2026-08-26 | PostgreSQL `data-service` 설계 이식 — **실행된 적 없는 경로**였다 (v2.0 변경사항 참조) |

## 다음 버전을 팔 때

```bash
cp -r docs/erd/version2.3 docs/erd/version2.4
# version2.3 을 고치고, 변경사항.md 에 무엇이 왜 바뀌었는지 적는다
python scripts/check_doc_links.py    # 상대 경로가 바뀌므로 반드시 돌린다
```
