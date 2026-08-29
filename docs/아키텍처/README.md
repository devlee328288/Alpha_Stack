# 아키텍처 — 버전 목록

> 최신 버전 폴더가 정본입니다. 옛 버전은 **발전 과정의 기록**이라 지우지 않습니다.
> 규약 전문은 [docs/README.md](../README.md).

| 버전 | 날짜 | 무엇이 담겼나 |
|---|---|---|
| **[version1.1](version1.1/)** | 2026-08-29 | **`supply/` as_of 정문 · Postgres 제거 · Streamlit · 다이어그램 3장** |
| [version1.0](version1.0/) | 2026-08-26 | 계층 구조와 의존 방향 |

## 다음 버전을 팔 때

```bash
cp -r docs/아키텍처/version1.1 docs/아키텍처/version1.2
# version1.2 를 고치고, 변경사항.md 에 무엇이 왜 바뀌었는지 적는다
python scripts/check_doc_links.py    # 상대 경로가 바뀌므로 반드시 돌린다
```

## 다이어그램

`.mmd` 가 정본이고 `.png` 는 거기서 굽습니다.

```bash
cd docs/아키텍처/version1.1
mmdc -i 계층아키텍처.mmd -o 계층아키텍처.png -b white -s 2
```

⚠️ 내용 없는 `%%` 줄은 mermaid 파서를 깨뜨립니다. 주석에는 항상 내용을 붙입니다.
