# 아키텍처 — 버전 목록

> 최신 버전 폴더가 정본입니다. 옛 버전은 **발전 과정의 기록**이라 지우지 않습니다.
> 규약 전문은 [docs/README.md](../README.md).

| 버전 | 날짜 | 무엇이 담겼나 |
|---|---|---|
| **[version1.2](version1.2/)** | 2026-08-29 | **발표용 구성도 2장(HTML) · `supply/` 문이 둘로 · 서 있는 것과 계획을 가름** |
| [version1.1](version1.1/) | 2026-08-29 | `supply/` as_of 정문 · Postgres 제거 · Streamlit · 다이어그램 3장 |
| [version1.0](version1.0/) | 2026-08-26 | 계층 구조와 의존 방향 |

## 다음 버전을 팔 때

```bash
cp -r docs/아키텍처/version1.2 docs/아키텍처/version1.3
# version1.3 을 고치고, 변경사항.md 에 무엇이 왜 바뀌었는지 적는다
python scripts/check_doc_links.py    # 상대 경로가 바뀌므로 반드시 돌린다
```

## 다이어그램 — 두 종류를 쓴다

**원문이 정본이고 그림은 거기서 굽습니다.** 그림만 커밋하면 이력에 바이너리 덩어리만
쌓이고 무엇이 달라졌는지 아무도 못 읽습니다.

### ① 흐름 그림 — mermaid (`.mmd`)

"무엇이 무엇으로 흐르나" 가 전부인 그림. 짧고 고치기 쉽습니다.

```bash
cd docs/아키텍처/version1.2
mmdc -i 계층아키텍처.mmd -o 계층아키텍처.png -b white -s 2
```

⚠️ 내용 없는 `%%` 줄은 mermaid 파서를 깨뜨립니다. 주석에는 항상 내용을 붙입니다.

### ② 발표용 구성도 — HTML (`.html`)

카드 안에 부제·배지·수치를 넣는 **자유 배치**. mermaid 로는 표현할 수 없습니다.

```bash
node scripts/render_html.mjs docs/아키텍처/version1.2/시스템아키텍처.html
node scripts/render_html.mjs docs/아키텍처/version1.2/AI에이전트구성.html
```

⚠️ 한글은 `word-break` 기본값이면 **단어 한가운데서** 끊깁니다(`않는 다` · `열 거`).
`word-break: keep-all` 을 반드시 넣습니다.

⚠️ puppeteer 는 이 저장소가 직접 의존하지 않고 **이미 깔려 있는 `@mermaid-js/mermaid-cli`
것을 빌려 씁니다.** 그림 하나 굽자고 Chrome 을 또 내려받게 만들 이유가 없습니다.
