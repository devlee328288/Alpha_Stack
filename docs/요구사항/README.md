# 요구사항 — 버전 목록

> 최신 버전 폴더가 정본입니다. 옛 버전은 **발전 과정의 기록**이라 지우지 않습니다.
> 규약 전문은 [docs/README.md](../README.md).

| 버전 | 날짜 | 무엇이 담겼나 |
|---|---|---|
| [**version2.0**](version2.0/) ⭐ | 2026-08-27 | **1차 = Must 18건으로 범위를 그었다** — Should·Could·Won't 12건을 2차 로드맵 표 하나로 · F-30 조사 160줄을 [docs/조사/](../조사/)로 · 팀원 제안(F1·ARIMA 대조) 수용기준 반영 |
| [version1.0](version1.0/) | 2026-08-26 | 무엇을 만드나 — 기능 29건 · MoSCoW · 수용 기준 |

## 다음 버전을 팔 때

```bash
cp -r docs/요구사항/version2.0 docs/요구사항/version2.1
# version2.1 을 고치고, 변경사항.md 에 무엇이 왜 바뀌었는지 적는다
python scripts/check_doc_links.py    # 상대 경로가 바뀌므로 반드시 돌린다
```
