# AGENTS.md — AlphaStack 팀 협업 규약

> 이 문서가 이 저장소의 **단일 진실 원천**입니다. 사람과 AI 에이전트 모두 여기를 따릅니다.
> 규칙을 바꿔야 하면 이 파일을 먼저 고치고, 결정의 근거는 [ADR](docs/decisions/)에 남깁니다.
>
> 최초 작성 2026-08-25 (저장소 이관 세션)

---

## 0. 이 저장소는 상위 규약의 예외입니다 ★

팀장의 개인 모노레포(`EST-Camp-AI-Quant`)에는 별도의 `AGENTS.md` · `CLAUDE.md` 가 있고,
그 규약은 **"모든 저장소에서 main 직커밋, 브랜치·PR 을 만들지 않는다"** 입니다.

**이 저장소는 그 규칙을 따르지 않습니다.** 이유는 두 가지입니다.

1. 상위 규약은 **1인 저장소를 전제로** 쓰였습니다. 4명이 같은 파일을 만지는 곳에서
   main 직커밋을 하면 서로의 작업을 덮고 non-fast-forward 가 잦아집니다.
2. **소유자가 다릅니다.** 상위 규약은 `EST-Bootcamp-Dongwon` 조직을 전제하지만
   이 저장소는 팀 계정 `devlee328288` 소유입니다.

### 예외로 남기는 기록

| 항목 | 상위 규약 | 이 저장소 |
|---|---|---|
| 브랜치 | main 직커밋 · PR 금지 | **feature 브랜치 → PR** |
| GitHub 소유자 | `EST-Bootcamp-Dongwon` | **`devlee328288`** (개인 계정) |
| 위치 | 모노레포 서브모듈 | **완전 독립** (상위 `.gitignore` 에 `team_project/`) |

**그대로 물려받는 것**: `gh pr merge` 금지 · force push 금지 · `reset --hard` 금지 ·
저장소 삭제/이관/공개범위 변경 금지 · 시크릿·데이터원본 커밋 금지 · 한국어 · KST 기준

---

## 1. Git 흐름

### 1.1 기본 순서

```bash
git pull origin main
git switch -c feat/무엇을-한다        # 브랜치를 먼저 판다
# ... 작업 ...
git status --short                    # ⚠️ push 전 필수 — 아래 1.4 참고
git add -p                            # 의도한 것만 담는다
git commit
git push -u origin feat/무엇을-한다
gh pr create                          # PR 생성까지만
```

**머지는 사람이 GitHub 웹에서 직접 합니다.**

### 1.2 브랜치 이름

| 접두 | 쓰임 | 예 |
|---|---|---|
| `feat/` | 기능 추가 | `feat/rsi-macd-features` |
| `fix/` | 버그 수정 | `fix/walk-forward-gap` |
| `docs/` | 문서만 | `docs/erd-update` |
| `refactor/` | 동작 그대로, 구조만 | `refactor/split-settings` |
| `test/` | 테스트 추가 | `test/metrics-edge-cases` |

### 1.3 커밋 메시지

```
type: 무엇을 했는지 한 줄 (한국어, 명령형)

왜 그렇게 했는지. 무엇을 고민했고 무엇을 버렸는지.
숫자가 있으면 숫자를 적는다 (실측값만 — 추측이면 추측이라고 쓴다).
```

`type` 은 브랜치 접두와 같습니다 (`feat` · `fix` · `docs` · `refactor` · `test` · `chore`).

### 1.4 push 전 확인 — ⚠️ 이 저장소는 PUBLIC 입니다

```bash
git status --short --untracked-files=all
```

아래가 목록에 있으면 **push 하지 않습니다.**

- `.env` · `*.key` · `*.pem` — 자격증명
- `*.csv` · `*.parquet` · `*.db` — 데이터 원본
- `*.pdf` · `*.zip` — 대용량 바이너리
- `.venv/` · `__pycache__/` — 환경·캐시

`.gitignore` 가 막고 있지만 **눈으로 확인합니다.** 한 번 push 된 시크릿은 이력에
영원히 남고, 지우려면 히스토리 재작성이 필요한데 그건 금지입니다.

### 1.5 절대 하지 않는 것 ★

| 금지 | 이유 | 대신 |
|---|---|---|
| `gh pr merge` (모든 옵션) | **계정 정지 이력이 있습니다** | 사람이 GitHub 웹에서 머지 |
| auto-merge 활성화 | 위와 같음 | 쓰지 않음 |
| `gh api` 반복 호출·스크립트 | 위와 같음 | 필요한 정보는 사람에게 요청 |
| force push (`--force`, `-f`) | 남의 커밋이 사라집니다 | `git revert` |
| `git reset --hard` | 되돌릴 수 없습니다 | `git revert` · `git stash` |
| 히스토리 재작성 (`rebase -i`, `filter-branch`) | 4명의 로컬이 전부 깨집니다 | 하지 않음 |
| 저장소 삭제·이관·공개범위 변경 | | 소유자가 웹 UI 에서 직접 |

---

## 2. 두 원격 — GitHub 와 GitLab

| 원격 | 주소 | 범위 | 누가 |
|---|---|---|---|
| `origin` | github.com/devlee328288/Alpha_Stack | **모든 브랜치 + PR** | 팀 전원 |
| `gitlab` | gitlab.com/dev-dongwon05253/alpha_stack | **`main` 만** | 팀장 개인 보관 |

### GitLab 은 왜 main 만인가

GitLab 은 팀장 개인 계정의 이중 보관용입니다. 팀원 3명의 feature 브랜치까지 미러하면

- 팀장이 관리하지 못하는 ref 가 계속 쌓이고
- 팀원이 GitHub 에서 브랜치를 지워도 **GitLab 에는 영원히 남습니다**

머지되면 어차피 `main` 에 들어오므로, 합의된 결과물만 보관하면 충분합니다.

```bash
# 팀장만 · main 이 머지된 뒤
git checkout main && git pull origin main
git push gitlab main
```

⚠️ **팀원은 `gitlab` 원격에 push 하지 않습니다.** 접근 권한도 없습니다.

---

## 3. 계정 분리 — 팀 작업은 팀 계정으로

이 폴더(`team_project/` 아래)에서는 git 신원이 **자동으로 팀 계정**이 됩니다.
`~/.gitconfig` 의 `includeIf` 규칙이 `~/.gitconfig-team` 을 끌어옵니다.

```
user.name  = devlee328288
user.email = 312804146+devlee328288@users.noreply.github.com
```

확인:

```bash
git config user.email     # 팀 이메일이 나와야 합니다
```

⚠️ **`gh auth switch` 를 쓰지 않습니다.** 활성 계정을 바꾸는 대신 폴더 단위로
가르는 것이 이 설정의 목적입니다. 개인 저장소 작업이 팀 계정으로 올라가는 사고를 막습니다.

---

## 4. 코드 규약

### 4.1 파이썬

| 항목 | 규칙 | 비고 |
|---|---|---|
| 버전 | 3.12 | `pyproject.toml` 이 강제 |
| 들여쓰기 | **4 spaces** | PEP8 |
| 함수·변수 | **`snake_case`** | ⚠️ 아래 예외 참고 |
| 클래스 | `PascalCase` | |
| 상수 | `UPPER_SNAKE_CASE` | |
| 줄 길이 | 100 | `ruff` 가 검사 |
| 린터 | `ruff check .` | `E` `F` `W` `I` `B` |

> ⚠️ **네이밍 예외 기록**: 팀장의 전역 규약은 "변수/함수는 camelCase" 입니다.
> 이 저장소는 파이썬에서 `snake_case` 를 씁니다 — 이관한 9,000줄이 전부 그렇고,
> `sklearn` · `pandas` API 도 전부 `snake_case` 라 섞으면 읽기 어려워집니다.
> **프론트엔드·JS 코드가 생기면 그쪽은 camelCase 입니다.**

### 4.2 주석과 문서 — 한국어로, 왜를 적는다

```python
# ❌ 나쁜 주석 — 코드가 이미 말하는 것을 반복한다
# turnover 를 계산한다
turnover = np.abs(pos - previous)

# ✅ 좋은 주석 — 왜 그렇게 했는지, 무엇을 조심해야 하는지
# 맨 앞은 "아무것도 안 들고 있던 상태"이므로 0 이다.
# np.diff 로 짜면 첫 원소가 사라져 매 폴드마다 진입이 공짜가 된다.
previous = np.concatenate(([0.0], pos[:-1]))
```

**특히 이런 것을 반드시 적습니다.**

- 틀려도 **에러가 안 나는** 지점 (look-ahead · 정렬 · 부호)
- 숫자가 **가정**인지 **실측**인지
- 근사·한계 — 있는 척하지 않습니다

### 4.3 테스트

새 코드에는 테스트를 붙입니다. 특히 `evaluation/` 은 **틀려도 에러가 안 나는** 곳이라
테스트가 유일한 안전망입니다.

```bash
.venv/Scripts/python.exe -m pytest tests/ -q
```

테스트 이름은 한국어로 **무엇을 보장하는지** 적습니다.

```python
def test_첫_시점_진입_비용을_빠뜨리지_않는다():
```

### 4.4 SQL

- 테이블·컬럼은 `snake_case`
- 스키마 변경은 `sql/init/` 수정 + **ADR 필수**
- `DROP` · `TRUNCATE` 는 PR 설명에 **왜 필요한지** 반드시 적습니다

---

## 5. 문서 규칙

### 5.1 결정은 ADR 로 남깁니다

되돌리기 어렵거나 나중에 "왜 이렇게 했지"가 나올 결정은 [docs/decisions/](docs/decisions/)에
남깁니다. 형식은 [0001](docs/decisions/0001-repo-bootstrap.md)을 따릅니다.

```
docs/decisions/NNNN-짧은-제목.md
```

### 5.2 미결 사항은 회의안건으로

정해지지 않은 것은 [docs/회의안건.md](docs/회의안건.md)에 모읍니다.
코드 주석에서 `→ docs/회의안건.md` 로 가리킵니다.
**결정되면 ADR 로 옮기고 회의안건에서 지웁니다.**

### 5.3 README 는 상태를 반영합니다

기능이 완성되면 README 의 "현재 상태" 표를 같은 PR 에서 갱신합니다.
**문서와 코드가 어긋난 채 머지되지 않게 합니다.**

### 5.4 숫자는 실측만

문서에 숫자를 적을 때는 **잰 것**만 적습니다. 재지 않았으면 "미측정"이라고 씁니다.
추측값을 적으면 다음 사람이 그것을 근거로 판단합니다.

---

## 6. 실행처 — 로컬이 기본입니다

| 작업 | 실행처 |
|---|---|
| LightGBM 학습·하이퍼파라미터 그리드 | **로컬** (배포판이 GPU 빌드가 아닙니다) |
| sklearn RandomForest · 보정 · bootstrap | **로컬** (CPU 경로) |
| 전처리 · 피처 · EDA · 시각화 | **로컬** (I/O 바운드) |
| 딥러닝 학습 (LSTM/GRU) | Colab Pro |
| 대규모 임베딩 생성 | Colab Pro |

**코드는 언제나 로컬이 정본입니다.** Colab 에서 고쳤으면 곧바로 로컬 파일에 되씁니다.
Colab VM 안에만 남은 결과물은 런타임이 끊기면 사라지고 git 이력과 갈라집니다.

원자료는 Colab 에 올리지 않습니다. 전처리를 로컬에서 마친 뒤 **학습에 필요한 최소
컬럼만** 내보냅니다.

---

## 7. 절대 제약 (프로젝트 전체)

- **LLM 유료 API 비용 0원** — 매매 신호 경로에 LLM 을 두지 않습니다
  (HuggingFace 인코더는 생성형이 아니므로 ⑦ 감성 피처에 쓸 수 있습니다)
- **클라우드 인프라 미사용** — 학습은 로컬, 결과물만 서빙
- **헤비 프론트 프레임워크 금지**
- **CI/CD 를 필수 경로에 두지 않습니다** — 없어도 개발이 굴러가야 합니다

---

## 8. 작업 시작 전 확인

1. `git pull origin main`
2. [docs/회의안건.md](docs/회의안건.md) — 내가 건드릴 부분이 미결인지 확인
3. 브랜치를 판다 (`main` 에서 바로 작업하지 않습니다)
4. 작업 → `pytest` → `ruff check` → `git status --short`
5. PR 생성 (머지는 사람이)
