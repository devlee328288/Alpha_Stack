# ADR-AS-0001: 팀 저장소를 독립으로 세우고, 원본에서 필요한 것만 골라 재구성한다

## 상태

채택됨 (2026-08-25) · 저장소 이관 세션

## 맥락

1차 팀 프로젝트 AlphaStack 이 2026-09-01 에 시작한다. 이동원은 개인 프로젝트
`data-service` 를 이미 가지고 있고, 거기에 KRX 수집·저장·시계열 분석 코드가 있다.
팀 프로젝트는 그 자산 위에서 출발하는 것이 합리적이다.

그런데 착수 시점의 상태가 여러 갈래로 어긋나 있었다.

| 사실 | 문제 |
|---|---|
| `Alpha_Stack/` 에 `.git` 이 없었다 | 저장소가 아니었다. docs 3파일만 있었다 |
| 상위 모노레포에서 `?? team_project/` 로 떠 있었다 | `git add -A` 한 번에 PDF 735KB 가 딸려 들어간다 |
| GitHub `main` 에 커밋 `be4e647` 이 이미 있었다 | `git init` → push 하면 non-fast-forward 로 거절된다 |
| GitLab 은 브랜치 0개 | 두 원격이 갈라져 있었다 |
| GitHub 소유자가 `devlee328288` | 상위 규약이 전제하는 `EST-Bootcamp-Dongwon` 이 아니다 |
| 저장소가 PUBLIC | 시크릿 사고의 대가가 크다 |
| 원본에 ML 라이브러리가 없다 | sklearn·LightGBM·XGBoost 를 새로 들여야 한다 |

무엇보다, **세 차수의 주제가 이미 정해져 있다는 사실**이 구조 결정을 강제했다.

| 차수 | 주제 |
|---|---|
| 1차 | 주가지수 데이터 활용 머신러닝·딥러닝 |
| 2차 | 나만의 로보 어드바이저 개발 **및 성과 검증** |
| 3차 | 나만의 투자 인디케이터 개발 **및 성과 검증** |

셋 다 "개발 및 성과 검증"으로 끝난다. 매 차수 새로 만드는 것은 앞쪽이고 재는 방법은
같다. 검증 코드를 어디에 두느냐가 6주 전체의 비용을 결정한다.

## 결정

### 1. 상위 모노레포와 완전히 분리한다

서브모듈로 엮지 않고, 상위 `.gitignore` 에 `team_project/` 를 넣는다.

**왜 서브모듈이 아닌가.** 팀원 4명이 함께 쓰는 저장소의 gitlink 를 개인 모노레포가
붙들면, 팀원이 push 할 때마다 개인 작업 트리가 더러워진다. 소유자도 신원도 다르다.

### 2. `git init` → `fetch` → `checkout` 으로 기존 커밋 위에 올라탄다

`be4e647` 은 LICENSE 하나뿐인 Initial commit 이라 `docs/` 와 충돌하지 않았다.

```bash
git init -b main
git remote add origin ...
git fetch origin
git checkout -b main --track origin/main   # docs/ 는 untracked 로 살아남는다
```

`reset --soft` 조차 필요 없었다. **`reset --hard` 는 규약상 금지이므로 애초에
후보가 아니었다.**

### 3. 브랜치 전략은 PR — 상위 규약의 예외로 기록한다

상위 규약은 "모든 저장소에서 main 직커밋, 브랜치·PR 금지"다. **이 저장소는 따르지
않는다.** 그 규칙은 1인 저장소를 전제로 쓰였고, 4명이 같은 파일을 만지는 곳에서는
서로의 작업을 덮는다.

⚠️ **`gh pr merge` 는 여전히 절대 금지다** (계정 정지 이력). PR 생성까지만 하고
머지는 사람이 GitHub 웹에서 한다.

### 4. GitLab 은 `main` 만 미러한다

GitLab 은 이동원 개인 계정의 이중 보관용이다. 팀원 3명의 feature 브랜치까지 미러하면
관리하지 못하는 ref 가 쌓이고, 팀원이 GitHub 에서 브랜치를 지워도 GitLab 에는 남는다.
머지되면 어차피 `main` 에 들어오므로 합의된 결과물만 보관하면 충분하다.

### 5. 구조를 원본 그대로 두지 않고 목적에 맞게 재구성한다 ★

원본은 `app/core` · `app/clients` · `app/repositories` · `app/services/*` 다.
그대로 두면 이관은 공짜지만 **1차 프로젝트의 목적이 구조에 드러나지 않는다.**

```
common/  ingest/{clients,store}/  features/  models/  evaluation/  api/  timeseries/
```

핵심은 **`evaluation/` 을 최상위 형제로 뺀 것**이다. `app/services/` 안에 묻으면
"2·3차가 그대로 가져다 쓴다"는 의도가 사라진다.

그리고 `evaluation/` 은 `models/` 를 import 하지 않는다. 받는 것은 포지션 배열과
수익률 배열뿐이다. 무엇이 그 포지션을 만들었는지 모르게 짜 두면 2·3차가 그대로
쓸 수 있다. **이 무지(無知)가 재사용의 조건이다.**

### 6. 패키지를 저장소 루트에 평탄하게 둔다

`Alpha_Stack/alphastack/common/...` 이 아니라 `Alpha_Stack/common/...` 이다.

⚠️ **이 결정이 실제 버그를 고쳤다.** 중첩 구조에서 `ingest/clients/krx_data.py` 의
`parents[2]` 는 저장소 루트가 아니라 `alphastack/` 을 가리켰다. 원본에서 `app` 을
건너뛰도록 센 숫자였는데 폴더가 하나 더 끼었기 때문이다. **`import` 는 멀쩡히
통과했다** — 파일을 실제로 읽는 순간에야 드러났을 것이다.

평탄화하면서 깊이를 다시 셌다.

| 파일 | 원본 | 지금 |
|---|---|---|
| `common/paths.py` · `common/secrets.py` | `parents[2]` | **`parents[1]`** |
| `ingest/clients/*.py` · `ingest/store/*.py` | `parents[2]` | `parents[2]` (그대로 맞다) |

앞으로는 깊이를 세지 말고 [`common/paths.py`](../../common/paths.py) 의 상수를 쓴다.

### 7. 파이썬은 `snake_case` 를 쓴다

이동원의 전역 규약은 "변수/함수는 camelCase" 지만, 이관한 9,000줄이 전부 `snake_case`
이고 `sklearn`·`pandas` API 도 그렇다. 섞으면 한 파일 안에서 두 규약이 부딪힌다.
**프론트엔드·JS 가 생기면 그쪽은 camelCase 다.**

### 8. `.key` 를 `.env` 로 옮기되, 쓰지 않는 자격증명은 빼고 옮긴다

원본 `.key` 에는 12개 키가 있었다. 이관한 코드가 **실제로 참조하는 이름만** 옮겼다.

| | 키 |
|---|---|
| 옮김 (9) | `KRX_API_KEY` `KRX_ID` `KRX_PW` `DART_API_KEY` `ECOS_API_KEY` `FRED_API_KEY` `KOSIS_API_KEY` `FINANCE_SUPERVISORY_API_KEY` `HUGGINGFACE_ACCESS_TOKEN` |
| **뺌 (3)** | `SUPABASE_DB_PASSWORD` `NAVER_API_CLIENT_ID` `NAVER_API_SECRET_KEY` |

`SUPABASE_DB_PASSWORD` 는 **개인 프로젝트 DB 비밀번호**다. 이관 코드가 쓰지 않고,
팀 Supabase 는 새로 파기로 되어 있다. 팀 작업 폴더에 둘 이유가 없다.

`.gitignore` 가 `.env` 를 막는 것을 `git check-ignore` 로 확인했다.

## 무엇을 가져오고 무엇을 버렸나

### 가져온 것 — 9,031줄

| 새 위치 | 원본 | 줄수 |
|---|---|---|
| `common/` | `app/core/` (6/8) | 1,360 |
| `ingest/clients/` | `app/clients/` (전부) | 3,978 |
| `ingest/store/` | `app/repositories/` (5/11) | 1,612 |
| `timeseries/` | `app/services/timeseries/` | 2,063 |
| `scripts/` | `scripts/` (2/18) | 532 |
| `sql/init/` | `sql/init/` (2/3) | 250 |

`clients` 9종을 전부 가져온 이유는 **크롤링 검수·개선을 맡을 팀원**이 있기 때문이다.
`krx_data` 하나만 옮기면 검수할 대상이 없다.

### 버린 것

| 버린 것 | 규모 | 이유 |
|---|---|---|
| `app/services/research/` | 24파일 11,241줄 | 리포트 생성기. 1차 범위 밖 |
| `app/routers/` | 17파일 · 엔드포인트 64개 | 쓰지 않는 엔드포인트는 유지 비용만 만든다 |
| `app/main.py` · `core/api_docs.py` | | 최소 API 를 새로 정의한다 |
| `core/parallel.py` | 117줄 | research 전용 |
| `repositories/` 6파일 | 2,025줄 | clip·report·user·industry·snapshot |
| `sql/init/03-clip.sql` | 105줄 | 대응 코드를 가져오지 않았다 |

## 결과 (실측 · 2026-08-25)

| 항목 | 값 |
|---|---|
| 이관 | 9,031줄 |
| 신규 작성 | 734줄 (`evaluation/` 478 + `tests/` 159 + 계약 97) |
| 모듈 import | **32 / 32 성공** |
| 검증 엔진 테스트 | **20 / 20 통과** (0.23초) |
| `ruff` (새 코드) | **0건** |
| `ruff` (전체) | 36건 — 전부 원본에서 물려받은 것 (원본도 동일 36건) |

### ML 스택 호환성을 실측했다

원본은 `numpy 2.5.1` · `pandas 3.0.5` 로 상당히 앞서 있어, ML 라이브러리와 ABI 가
어긋날 위험이 있었다. **의존성 해석 성공이 import 성공을 보장하지 않는다** —
실제로 설치해서 확인했다.

```
numpy 2.5.1 · pandas 3.0.5 · scipy 1.18.1 · scikit-learn 1.9.0
LightGBM 4.7.0 · XGBoost 3.4.1 · joblib 1.5.3   → 7종 전부 import 성공
```

이 조합을 `pyproject.toml` 에 고정했다. **킥오프 첫날 4명이 환경에서 막히는 것을
막는 것이 이 확인의 목적이다.**

### 이관 중 잡은 사고 둘

1. **상대 import 를 놓칠 뻔했다.** `dart_report.py` 의
   `from ..repositories import tmp_cache` 는 절대 경로 치환 규칙에 걸리지 않았다.
   import 검증을 돌리지 않았다면 팀원이 9/1에 `ModuleNotFoundError` 를 봤을 것이다.
   → `tmp_cache.py` 를 함께 이관하고 절대 경로로 고쳤다.

2. **경로 깊이가 깨져 있었다.** 위 결정 6 참고.

## 대가

- 원본과 코드를 주고받을 때 **import 경로를 손으로 번역해야 한다.**
  구조를 바꾼 값이다. 알고 택했다.
- `common/settings.py`(437줄)에 1차가 쓰지 않는 원본 설정이 섞여 있다.
  팀 Supabase 가 정해진 뒤 한 번에 덜어낸다 → [회의안건 C-1](../회의안건/2026-09-01-킥오프.md)
- 최상위 패키지 이름(`alphastack.*`)이 없어 `common`·`models`·`api` 가 top-level 을
  차지한다. 서드파티와 이름이 부딪힐 이론적 위험이 있으나, 프로젝트 루트가 `sys.path`
  앞쪽에 오므로 실질 위험은 낮다.

## 남은 과제

정해지지 않은 것은 [docs/회의안건/2026-09-01-킥오프.md](../회의안건/2026-09-01-킥오프.md) 에 모았다. 특히 아래 셋이
정해지기 전에는 `features/` 가 첫 줄도 못 나간다.

1. 팀 Supabase — 팔 것인가, 용량은, 권한은
2. 예측 대상과 레이블 정의
3. 데이터 백필 범위 (현재 297거래일은 얇다)
