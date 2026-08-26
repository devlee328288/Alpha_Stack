# 아키텍처 — 버전 목록

> 최신 버전 폴더가 정본입니다. 옛 버전은 **발전 과정의 기록**이라 지우지 않습니다.
> 규약 전문은 [docs/README.md](../README.md).

| 버전 | 날짜 | 무엇이 담겼나 |
|---|---|---|
| [version1.0](version1.0/) | 2026-08-26 | 계층 구조와 의존 방향 |

## 다음 버전을 팔 때

```bash
cp -r docs/아키텍처/version1.0 docs/아키텍처/version1.1
# version1.1 을 고치고, 변경사항.md 에 무엇이 왜 바뀌었는지 적는다
python scripts/check_doc_links.py    # 상대 경로가 바뀌므로 반드시 돌린다
```
