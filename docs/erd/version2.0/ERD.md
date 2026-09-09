# ERD — AlphaStack 데이터 모델 (v2.0)

> **정본은 [`ingest/store/migrations.py`](../../../ingest/store/migrations.py) 입니다.**
> 이 문서가 코드와 어긋나면 마이그레이션 코드가 맞습니다.
> v2.0 · 2026-09-02 · 이전 판 [v1.0](../version1.0/ERD.md) · [변경사항](변경사항.md)
>
> 실측 기준: `krx_cache.db` **스키마 v8 · 표 16개 · 1,650 MB** (2026-09-02,
> `python scripts/check_db.py` + `PRAGMA table_info` 전수)

⚠️ **v1.0 과는 완전히 다른 문서입니다.** v1.0 은 PostgreSQL `data-service` 설계
(`securities` · `ohlcv` · `watermark` …)를 옮긴 것인데, 그 경로는 실행된 적이 없고
저장소는 **SQLite 파일 하나**(`data/krx_cache.db`)로 확정됐습니다. 왜 그랬는지는
[변경사항](변경사항.md)에 있습니다.

---

## 1. 한눈에 — 표 16개를 다섯 묶음으로

자료 표 5개에 운영 표 11개입니다. **운영 표가 더 많은 것이 이 설계의 특징입니다** —
"무엇을 받았나"만큼 "왜 안 받았나 · 언제 받았나 · 몇 번 불렀나"를 남기기 때문입니다.

```
┌── 자료 (팀원이 쓰는 것) ──────────────────────────────────────────┐
│                                                                  │
│  daily_price      종목 일별시세   9,223,644행  2010-01-04 ~      │
│  index_price      지수 일별시세     196,119행  2010-01-04 ~      │
│  dart_financial   재무 (사업보고서) 662,933행  FY2015 ~ 2025     │
│  macro_series     거시 지표 9종      17,851행  2009-08 ~         │
│  dart_disclosure  공시 목록               0행  (자리만 있다)      │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘
┌── 수집 대장 (다시 받지 않기 위해) ────────┐  ┌── 반입 (남이 준 파일) ──┐
│  collect_log       출처×대상 수집 대장    │  │  inbox_batch    묶음     │
│  fetch_log         옛 종목 대장 (읽기 전용)│  │  inbox_accepted 합격 행  │
│  index_fetch_log   옛 지수 대장 (읽기 전용)│  │  inbox_quarantine 격리 행│
└──────────────────────────────────────────┘  └─────────────────────────┘
┌── 호출 예산·원문 ────────────────────────┐  ┌── 실행 기록 (v8) ────────┐
│  call_budget    오늘 몇 번 불렀나         │  │  ingest_run       실행    │
│  raw_response   응답 원문 (재정규화용)    │  │  ingest_run_stage 단계    │
│  robots_cache   robots.txt 24시간 캐시   │  │  (대시보드가 폴링할 표)   │
└──────────────────────────────────────────┘  └─────────────────────────┘
```

행 수는 2026-09-02 실측입니다. 수집이 돌면 움직입니다.

---

## 2. 자료 표

### 2.1 `daily_price` — 종목 일별 시세 ★

**PK `(bas_dd, code)`** · 보조 인덱스 `(code, bas_dd)`

| 칸 | 형 | 뜻 | 함정 |
|---|---|---|---|
| `bas_dd` | TEXT | 거래일 `YYYYMMDD` | **날짜형이 아니라 문자열**이다. 사전순 = 날짜순이라 `<=` 비교가 그대로 통한다 |
| `code` | TEXT | 종목코드 6자리 | **여섯 자리 숫자가 아니다** — 5·6번째 자리에 영문이 오는 종목이 84종 있다 (`0001B0` 등) |
| `name` | TEXT | 종목명 | 같은 코드도 이름이 바뀐다 |
| `market` | TEXT | `KOSPI` / `KOSDAQ` | 날짜당 시장별로 따로 받는다 |
| `sector` | TEXT | KRX 업종 | WICS 와 다르다. KOSPI 는 빈 문자열이 많다 |
| `open` `high` `low` `close` | INTEGER | 시·고·저·종가 (원) | 🔴 **수정주가가 아니다** — 아래 함정 절 참조 |
| `change` | INTEGER | 전일 대비 (원) | 전일 종가 = `close - change` |
| `change_rate` | REAL | 등락률 (%) | `change/(close-change)×100` 과 정확히 일치 (920만 행 위반 0건 실측) |
| `volume` | INTEGER | 거래량 (주) | 거래정지일은 0 |
| `value` | INTEGER | 거래대금 (원) | |
| `market_cap` | INTEGER | 시가총액 (원) | 1.8경 규모라 float64 유효숫자를 넘는다 — CSV 왕복 시 1 ULP 어긋남 실측 |
| `listed_shares` | INTEGER | 상장주식수 (주) | 증자·감자로 바뀐다. 분할 계수 역산에 쓴다 |

🔴 **수정주가가 아닙니다** (이슈 #51). 액면분할이 있으면 과거 주가가 그대로라
`close` 로 수익률을 계산하면 삼성전자 2018-05-04 가 −98% 로 읽힙니다.
**마이그레이션 v9 에서 `adj_open/high/low/close` 4칸을 추가하기로 확정**했습니다
(원 칸은 그대로 두고 조정 칸을 병행 — 원자료 보존 원칙). 소스는 FDR + 자체 교차검증.

### 2.2 `index_price` — 지수 일별 시세

**PK `(bas_dd, index_name)`** · 보조 인덱스 `(index_name, bas_dd)`

칸은 `daily_price` 와 같은 골격이되 가격이 REAL(포인트)이고, `index_class`(계열)가
있습니다. 1차 예측 대상 **코스피 200** 이 여기 삽니다. 지수는 계산값이라
수정주가 문제가 없습니다.

### 2.3 `dart_financial` — 재무 (전체 재무제표)

**PK `(corp_code, bsns_year, reprt_code, fs_div, sj_div, account_nm, ord, account_detail)`**
· 보조 인덱스 `(stock_code, bsns_year)` · `(corp_code, bsns_year, reprt_code)` · `(rcept_dt, corp_code)`

| 칸 (주요) | 뜻 | 함정 |
|---|---|---|
| `corp_code` | DART 회사 고유번호 | 종목코드와 다르다. 매핑은 `corp_code.zip` 에서 |
| `bsns_year` · `reprt_code` | 사업연도 · 보고서 종류 | 사업보고서 = `11011` |
| `fs_div` · `sj_div` | 연결/별도 · 재무제표 종류 | `CFS`(연결)·`OFS`(별도) × BS/IS/CIS/CF/SCE |
| `account_detail` | 계정 상세 (자본변동표) | 🔴 **PK 에서 빼면 자본변동표 6.4%(22,436행)가 조용히 사라진다** — 같은 계정명이 자본 항목별로 반복되기 때문 |
| `thstrm_amount` | 당기 금액 | 문자열로 오는 `-` 는 결측으로 정규화했다 |
| `rcept_dt` | **접수일** | **시점 정합의 기준.** 결산기준일로 조인하면 미공시 값이 붙는다(미래 누출) |

- 2026-09-02 실측: **662,933행 · 350종 · FY2015~2025 · 접수일 100% 채움**
- DART 는 **2015년 이전 전체계정을 주지 않습니다** (2010~2014 전수 호출로 실측)
- 학습에는 HF `alphastack-dart` 의 **pit/** 층만 씁니다 (시점은 `rcept_dt`).
  **bulk/** 층(44파일)은 접수일이 없어 학습 금지 — 탐색·대조용

### 2.4 `macro_series` — 거시 지표 (v7)

**PK `(indicator_id, period)`** · 보조 인덱스 `(known_at, indicator_id)` 등

| 칸 (주요) | 뜻 | 함정 |
|---|---|---|
| `indicator_id` | 지표 이름 (`base_rate` 등 9종) | |
| `period` | 기준 기간 (`202608` · `20260901`) | 월간·일간이 섞인다 — `cycle` 로 가른다 |
| `known_at` | **언제부터 알 수 있었나** | 🔴 **계산값이다.** ECOS 는 발표일을 안 준다(세 API 전수 확인). 발표 규칙에서 보수적으로 계산했으므로, **규칙을 바꾸면 전량 재수집** |
| `stat_code` · `item_code` | ECOS 원 코드 | 재현·검증용 |

2026-09-02 실측: 17,851행 · 9종 · 2009-08 ~.

### 2.5 `dart_disclosure` — 공시 목록 (자리만)

**PK `rcept_no`**. 0행 — 공시 본문 수집은 3차로 미뤘고 표만 먼저 팠습니다.

---

## 3. 운영 표

### 3.1 `collect_log` — 수집 대장 ★

**PK `(source, target)`**. "이미 받은 것을 다시 받지 않는다"의 근거입니다.
`status` 는 다섯 값 — `ok` · `empty`(휴장) · `error`(3회 재시도) ·
`quota_exhausted`(내일 재개) · `out_of_range`(출처가 안 주는 기간).
**`empty` 와 `error` 를 가르지 않으면 휴장일마다 매일 다시 물어보게 됩니다.**

`fetch_log` · `index_fetch_log` 는 900만 행 백필이 돌던 시절의 옛 대장입니다.
읽기 전용으로 남겨 두었고 새 기록은 `collect_log` 로만 갑니다.

### 3.2 `call_budget` — 호출 예산

**PK `(source, kst_date)`**. 서버가 거절하기 전에 **우리가 먼저 셉니다.**
KRX 는 키당 하루 10,000회(이용약관 제8조 ④).

### 3.3 `raw_response` — 응답 원문

**PK `(source, target, fetched_at)`**. zstd 압축 BLOB + SHA-256.
정규화 규칙이 틀린 것을 나중에 발견해도 **다시 받지 않고 원문에서 다시 만들기** 위한
표입니다. 재정규화는 수집 시각을 건드리지 않습니다(한 번 틀렸던 자리).

### 3.4 `inbox_batch` / `inbox_accepted` / `inbox_quarantine` — 반입 (v6)

남(팀원·외부)이 준 파일을 규격 검사해 들이는 자리입니다.
`inbox_batch` PK 는 `batch_id`(`<끝난시각>-<SHA-256 앞 12자>`) — 같은 파일을 규격
개정 뒤 다시 검사한 이력이 안 덮이게 시각과 지문을 함께 씁니다.
행 표 둘의 PK 는 `(batch_id, row_no)` — 합격이든 격리든 **원래 몇 번째 행이었는지**가
남아 보고서와 이어집니다.

🔴 **`daily_price` 에 바로 넣지 않습니다.** 우리가 받은 것과 남이 준 것을 한 표에
섞으면 "이 값은 누가 가져왔나"를 되짚을 수 없게 됩니다.

### 3.5 `ingest_run` / `ingest_run_stage` — 실행 기록 (v8)

**PK `run_id`** / **`(run_id, stage)`**. `python -m ingest` 한 번이 `ingest_run`
한 행이고, 출처별 단계가 `ingest_run_stage` 에 남습니다. **수집 현황 대시보드가
폴링할 표**로 설계했습니다 (FastAPI/화면 세션에서 사용 예정).

### 3.6 `robots_cache` — robots.txt 캐시

**PK `origin`**. 크롤링 전에 robots.txt 를 보는 가드의 24시간 캐시입니다.

---

## 4. 스키마 버전과 마이그레이션

`PRAGMA user_version` 으로 관리합니다. **번호는 이름이 아니라 선착순 인덱스**입니다 —
배정표는 `migrations.py` 주석표 · `sqlite_db.py` 표 · `NEXT_MIGRATION_VERSION` 세 곳에
있고, 바꿀 때 셋을 함께 고칩니다.

| 버전 | 제목 (코드의 배정표 그대로) |
|---|---|
| v1 | 수집 대장 · 호출 예산 (`collect_log` · `call_budget`) |
| v2 | 수집 대장에 시도 횟수 |
| v3 | 응답 원문 보존 (`raw_response`) |
| v4 | robots.txt 캐시 (`robots_cache`) |
| v5 | 반입 — 남의 자료를 들인 기록 (`inbox_*` 3표) |
| v6 | 공시 시점정합 — 결산기가 아니라 알게 된 날로 세운다 (`dart_financial` · `dart_disclosure`) |
| v7 | 거시 통계 — 기준월이 아니라 공표된 날로 세운다 (`macro_series`) |
| v8 | 수집 실행 기록 — 지금 돌고 있는지 밖에서 볼 수 있게 (`ingest_run` · `ingest_run_stage`) |
| **v9 (예정)** | **`daily_price` 에 `adj_*` 4칸 + 실측 거래일 달력** (이슈 #51) |

---

## 5. 이 문서가 답하지 않는 것

- **칸별 값의 품질** — `python scripts/check_data.py`(시세) ·
  `check_dart.py`(재무)가 답합니다. 스키마가 맞아도 값은 틀릴 수 있습니다.
- **누가 언제 꺼내 쓰나** — [`supply/`](../../../supply/__init__.py) 정문과
  [아키텍처](../../아키텍처/version1.2/아키텍처.md)가 답합니다.
- **HF 배포본의 구조** — [데이터파트 v3.0 명세서](../../데이터파트/version3.0/명세서.md)가 답합니다.
