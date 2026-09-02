# erd — 버전 목록

> 최신 버전 폴더가 정본입니다. 옛 버전은 **발전 과정의 기록**이라 지우지 않습니다.
> 규약 전문은 [docs/README.md](../README.md).

| 버전 | 날짜 | 무엇이 담겼나 |
|---|---|---|
| [**version2.0**](version2.0/) ⭐ | 2026-09-02 | **설계도가 아니라 실물** — `krx_cache.db` v8 · 표 16개를 `PRAGMA` 전수 실측으로 그렸다. 자료 5표 + 운영 11표 · 마이그레이션 v1~v8 이력 · 함정 칸(수정주가 아님 · `account_detail` · `known_at` 계산값) |
| [version1.0](version1.0/) | 2026-08-26 | PostgreSQL `data-service` 설계 이식 — **실행된 적 없는 경로**였다 (v2.0 변경사항 참조) |

## 다음 버전을 팔 때

```bash
cp -r docs/erd/version2.0 docs/erd/version2.1
# version2.1 을 고치고, 변경사항.md 에 무엇이 왜 바뀌었는지 적는다
python scripts/check_doc_links.py    # 상대 경로가 바뀌므로 반드시 돌린다
```
