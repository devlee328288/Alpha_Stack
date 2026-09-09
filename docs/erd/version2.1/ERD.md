# ERD — AlphaStack 데이터 모델 (v2.1)

> **정본은 [`ingest/store/migrations.py`](../../../ingest/store/migrations.py) 입니다.**
> 이 문서가 코드와 어긋나면 마이그레이션 코드가 맞습니다.
> v2.1 · 2026-09-02 · 이전 판 [v2.0](../version2.0/ERD.md) · [v1.0](../version1.0/ERD.md) ·
> [변경사항](변경사항.md)
>
> 실측 기준: `krx_cache.db` **스키마 v9 · 표 17개** (2026-09-02,
> `python scripts/check_db.py` + `PRAGMA table_info` 전수)

⚠️ **v1.0 과는 완전히 다른 문서입니다.** v1.0 은 PostgreSQL `data-service` 설계
(`securities` · `ohlcv` · `watermark` …)를 옮긴 것인데, 그 경로는 실행된 적이 없고
저장소는 **SQLite 파일 하나**(`data/krx_cache.db`)로 확정됐습니다. 왜 그랬는지는
[v2.0 변경사항](../version2.0/변경사항.md)에 있습니다.

---

## 1. 한눈에 — 표 17개를 다섯 묶음으로

자료 표 5개에 운영 표 12개입니다. **운영 표가 더 많은 것이 이 설계의 특징입니다** —
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
┌── 거래일 달력 (v9) ──────────────────────────────────────────────┐
│  trading_calendar  실제로 열린 날   12,306행  (ALL·KOSPI·KOSDAQ) │
│                    ⚠️ daily_price 에서 파생 — 시세가 늘면 다시 깐다│
└──────────────────────────────────────────────────────────────────┘
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
| `adj_open` `adj_high` `adj_low` `adj_close` | REAL | **수정 시·고·저·종가** (v9) | 🔴 **수익률·라벨은 이쪽으로** 계산한다. INTEGER 가 아닌 이유는 아래 |
| `adj_source` | TEXT | `fdr` / `chain` (v9) | 그 행의 수정값이 어디서 왔나. **날짜로는 유추할 수 없다** |

#### 🔴 원가격과 수정가격 — 어느 쪽을 쓰나 (v9)

`close` 는 KRX 원문 그대로라 **액면분할이 조정돼 있지 않습니다** (이슈 #51).
분할일에 가격이 그대로 뚝 떨어지므로 `close` 로 수익률을 계산하면 삼성전자
2018-05-04 이 **−98.04%** 로 읽힙니다 (실제 −2.08%). 분할·병합 1,139건이 806종(21.9%)에
걸쳐 있습니다.

| 쓰임 | 어느 칸 | 왜 |
|---|---|---|
| 수익률 · 라벨 · 변동성 피처 | **`adj_*`** | 분할이 펴져 있다 |
| 시가총액 | **`close`** | `market_cap = close × listed_shares` 는 원가격이라야 맞다 |
| 원문 대조 · 재현 | **`close`** | 원자료 보존 — 언제든 되돌아갈 수 있어야 한다 |

**원 칸을 덮지 않은 이유**가 하나 더 있습니다 — 후방조정 값은 **새 분할이 생기면 과거
전체가 다시 바뀝니다.** append-only 워터마크 반입과 정면으로 충돌하므로, 원본을 덮으면
어제 결과를 재현할 수 없습니다. (재현이 중요한 자리에서는 구간만 보는
`common.corporate_actions.span_factor` 를 쓰는 편이 안전합니다.)

**REAL 인 이유**: 후방조정 값은 정수가 아닙니다. 삼성전자 2010-01-04 은 원종가 809,000 이
아니라 **16,180.00** 이 되고, 분할이 잦았던 종목은 1원 아래로 내려갑니다.
INTEGER 로 두면 조용히 잘립니다.

**`adj_source` 가 필요한 이유**: FinanceDataReader(네이버 fchart)는 **최근 3,000거래일만**
줍니다 — `count` 를 6000·9000 으로 올려도 서버가 자르는 것을 실측했습니다. 그래서 그
경계 앞은 우리가 조정계수로 이어 붙였습니다(`chain`). 그런데 **경계가 종목마다 다릅니다** —
3,000일 창이 오늘이 아니라 **그 종목의 마지막 거래일**에 걸리기 때문에, 2015년에
상장폐지된 종목은 2010년까지 `fdr` 이 닿습니다. 날짜로 유추할 수 없어 칸으로 둡니다.

실측(2026-09-02 · 9,223,644행 **전부** 채움): `fdr` **81.6%** · `chain` **18.4%**.
`chain` 구간의 오차 상한은 **최대 0.39%** 입니다 — FDR 을 앵커 하루만 남기고 나머지
2,999일을 우리 계산으로 채워 진짜 값과 대조해 쟀습니다.

⚠️ **정지일**(`open=high=low=0`)은 `adj_open`·`adj_high`·`adj_low` 가 **NULL** 이고
`adj_close` 만 있습니다. 0 에 배율을 곱해 실으면 "그 날 0원" 이 되고, 고저 검사
(`0 ≤ 0 ≤ 0`)도 통과해 버립니다.

⚠️ **시세를 다시 받아도 `adj_*` 는 지워지지 않습니다.** `krx_store` 의 쓰기가
`INSERT OR REPLACE`(행을 지우고 새로 넣는다) 에서 `ON CONFLICT DO UPDATE`(적힌 칸만
바꾼다) 로 바뀌었기 때문입니다. 전자였다면 하루만 재수집해도 그 날 3천 종목의 수정주가가
사라지는데 **행 수는 그대로**라 어떤 검증에도 안 걸립니다.

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

### 3.7 `trading_calendar` — 실측 거래일 달력 (v9) ★

**PK `(bas_dd, market)`** · 보조 인덱스 `(market, bas_dd)`

| 칸 | 형 | 뜻 |
|---|---|---|
| `bas_dd` | TEXT | 거래일 `YYYYMMDD` |
| `market` | TEXT | `ALL` / `KOSPI` / `KOSDAQ` |
| `stock_count` | INTEGER | 그날 체결된 종목 수 (CHECK `> 0`) |
| `built_at` | TEXT | 언제 채웠나 |

**휴장일을 계산으로 맞히지 않습니다.** 주말만 걸러 세면 개발구간 평일 3,042일 중
**162일(5.3%)** 이 어긋나고, 그 162일은 명절·공휴일이라 하필 실적 발표와 뉴스가
몰리는 연휴 전후입니다. 뉴스의 `eff_dd` 배정이 이 달력 위에서 이뤄지므로 어긋나면
곧 미래참조가 됩니다. **우리가 실제로 받은 날**이 거래일입니다 — 추정이 아니라 기록입니다.

시장을 나눠 두는 이유: 한쪽 시장만 열리는 날을 양쪽 거래일로 잘못 읽지 않기 위해서입니다.
**휴장일은 이 표에 행이 없습니다** — `stock_count = 0` 행을 만들면 "0건으로 받았다" 와
"아직 안 받았다" 가 섞입니다.

⚠️ **`daily_price` 에서 파생된 표라 원본이 늘면 낡습니다.** 시세를 적재한 뒤
`ingest.store.adj_price.rebuild_calendar()` 를 함께 부릅니다.
`common/trading_calendar.py` 는 표가 없거나 비면 **원본을 직접 세는 폴백**으로 갑니다 —
조용히 낡은 답을 주는 것보다 느린 편이 낫기 때문입니다.

왜 표로 옮겼나 (로직은 원래 `common/trading_calendar.py` 에 있었습니다):

```
SELECT DISTINCT bas_dd FROM daily_price   9.2M 행을 훑는다   599ms
SELECT bas_dd FROM trading_calendar       4,102행을 읽는다    13ms   ← 45배
```

두 답이 같음을 실측으로 확인했습니다(4,102일 · 완전 일치).

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
| **v9** | **수정주가 4칸 · 실측 거래일 달력** (`daily_price.adj_*` · `trading_calendar`) — 이슈 #51 |
| v10 | 다음 빈 번호 |

---

## 5. 이 문서가 답하지 않는 것

- **칸별 값의 품질** — `python scripts/check_data.py`(시세) ·
  `check_dart.py`(재무)가 답합니다. 스키마가 맞아도 값은 틀릴 수 있습니다.
- **누가 언제 꺼내 쓰나** — [`supply/`](../../../supply/__init__.py) 정문과
  [아키텍처](../../아키텍처/version1.2/아키텍처.md)가 답합니다.
- **HF 배포본의 구조** — [데이터파트 v3.0 명세서](../../데이터파트/version3.0/명세서.md)가 답합니다.
