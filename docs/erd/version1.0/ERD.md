# ERD — AlphaStack 데이터 모델

> 정본은 [`sql/init/`](../sql/init/) 입니다. 이 문서가 코드와 어긋나면 **SQL 이 맞습니다.**
> 작성 2026-08-25 · 원본 `data-service` 스키마 이식 (clip 제외)

---

## 1. 전체 관계

```
┌──────────────────────────┐
│  securities              │  종목 마스터
│  ─────────────────────   │
│  security_id  PK  (IDENT)│◄────────┐
│  code             종목코드│         │
│  name  market  sector    │         │
│  listed_shares   최신값   │         │  FK
│  industry_code           │         │
│  corp_code   (DART 매핑) │         │
│  universe_tier core|full │         │
│  is_delisted             │         │
│  delisted_date           │         │
└──────────────────────────┘         │
                                     │
┌────────────────────────────────────┴──────┐
│  ohlcv                     ★ 학습 데이터   │
│  ───────────────────────────────────────  │
│  security_id  PK,FK                       │
│  trade_date   PK          ← RANGE 파티션   │
│  open high low close       (정수 · 원)     │
│  change                    전일대비        │
│  change_rate  numeric(12,4) 등락률(%)      │
│  volume  value  market_cap                │
│  listed_shares            ← 그 날의 값      │
└───────────────────────────────────────────┘

┌──────────────────────┐   ┌──────────────────────┐   ┌────────────────────┐
│  trading_calendar    │   │  ohlcv_sync_log      │   │  watermark         │
│  ─────────────────   │   │  ─────────────────   │   │  ───────────────   │
│  trade_date  PK      │   │  trade_date  PK      │   │  source  PK        │
│  is_open             │   │  rows                │   │  last_synced_at    │
│  source              │   │  fetched_at          │   │  last_key          │
│  note                │   │  status              │   │  rows  status      │
└──────────────────────┘   └──────────────────────┘   └────────────────────┘
   거래일인가?               거래일 단위 수집 대장       소스 단위 동기화 지점
```

**관계는 하나뿐입니다**: `ohlcv.security_id → securities.security_id`
나머지 셋은 독립 테이블로, 각자 다른 축의 메타데이터를 담습니다.

---

## 2. 테이블별 상세

### 2.1 `securities` — 종목 마스터

| 컬럼 | 타입 | 설명 |
|---|---|---|
| `security_id` | integer PK | 자동 생성 (IDENTITY) |
| `code` | text | 종목코드 6자리 (KRX 기준) |
| `name` | text | 종목명 |
| `market` | text | KOSPI · KOSDAQ · KONEX |
| `sector` | text | 소속부 |
| `listed_shares` | bigint | 상장주식수 — **최신값** |
| `industry_code` | text | DART 표준산업분류 |
| `fiscal_month` | smallint | 결산월 |
| `corp_code` | text | DART 고유번호 |
| `universe_tier` | text | `core`(KOSPI200+KOSDAQ150 350) · `full`(전종목) |
| `is_delisted` | boolean | 상장폐지 여부 |
| `delisted_date` | date | 상장폐지일 |

#### ⚠️ 상장폐지 종목을 지우지 않습니다 — 생존 편향

```sql
is_delisted boolean NOT NULL DEFAULT false,
```

망한 종목을 유니버스에서 빼면 백테스트가 **실제보다 훨씬 좋게** 나옵니다.
"살아남은 것들"만 보고 과거를 재는 셈이기 때문입니다.

**이건 1차 프로젝트에 직접 영향을 줍니다.** 학습 데이터를 만들 때 `is_delisted` 를
어떻게 다룰지 정해야 합니다 → [회의안건 A-3](회의안건.md)

#### 유일성 제약이 조건부인 이유

```sql
CREATE UNIQUE INDEX securities_code_active_uq ... WHERE NOT is_delisted;
```

같은 코드가 동시에 두 번 살아 있을 수는 없지만, **상장폐지 뒤 코드 재사용은 실제로
일어납니다.** 그래서 "살아 있는 것"에 한해서만 유일성을 겁니다.

---

### 2.2 `ohlcv` — 일별 시세 ★ 학습 데이터의 원천

| 컬럼 | 타입 | 설명 |
|---|---|---|
| `security_id` | integer PK, FK | |
| `trade_date` | date PK | RANGE 파티션 키 |
| `open` `high` `low` `close` | integer | 원 단위 정수 |
| `change` | integer | 전일대비 |
| `change_rate` | numeric(12,4) | 등락률(%) — 유일하게 정수가 아님 |
| `volume` | bigint | 거래량 |
| `value` | bigint | 거래대금 |
| `market_cap` | bigint | 시가총액 (수집 시점 기준) |
| `listed_shares` | bigint | **그 거래일의** 상장주식수 |

#### 왜 `change` 를 저장하나 — 역산하면 틀린다

`close` 차이로 역산하면 **3원씩 어긋나는 종목**이 있습니다. 원본을 그대로 받습니다.

#### 왜 `numeric(12,4)` 인가 — 그릇이 좁아서 안 들어간 행이 있었다

`numeric(8,4)` 는 정수부가 4자리뿐이라 실측 원본 1행이 들어가지 않았습니다.

```
20260209 · 052670 제일바이오 · close 625,000 · change 622,920 · rate 29948.08
```

액면병합·재상장으로 보이고 같은 행의 `close`·`change` 와 산술적으로 일관됩니다.
즉 **원본이 옳고 그릇이 좁았습니다.** 값을 깎는 대신 그릇을 넓혔습니다.

#### `ohlcv.listed_shares` 와 `securities.listed_shares` 는 다릅니다

| | 뜻 |
|---|---|
| `securities.listed_shares` | **최신** 상장주식수 |
| `ohlcv.listed_shares` | **그 거래일의** 상장주식수 |

역할이 다르므로 **둘이 갈리는 것이 정상**입니다. 동기화 오류로 읽지 마세요.

종목당 한 줄로 접으면 액면분할 종목의 **과거 회전율이 10배 틀립니다**(실측 4종목).

#### 파티셔닝

```sql
) PARTITION BY RANGE (trade_date);
```

연 단위 RANGE 파티션입니다 ([`02-partitions.sql`](../sql/init/02-partitions.sql)).
과거 백필로 자료가 늘어날 때 이 구조가 조회 성능을 지킵니다.

---

### 2.3 `trading_calendar` — 거래일인가 아닌가

| 컬럼 | 타입 | 설명 |
|---|---|---|
| `trade_date` | date PK | |
| `is_open` | boolean | 개장 여부 |
| `source` | text | 이 판정이 어디서 왔나 |
| `note` | text | |

#### `source` 가 신뢰도를 밝힙니다 — 근사치를 사실처럼 쓰지 않기 위해

| 값 | 뜻 | 신뢰도 |
|---|---|---|
| `krx_observed` | 실제 시세가 있었다 | 가장 높음 |
| `krx_zero_rows` | KRX 가 0건을 돌려줬다 → 휴장으로 본다 | 중간 |
| `weekday_approx` | 주말 규칙만 적용한 추정 (**공휴일 미반영**) | 낮음 |

⚠️ `weekday_approx` 구간을 그대로 믿고 피처를 만들면 공휴일이 거래일로 섞입니다.

---

### 2.4 `ohlcv_sync_log` — 거래일 단위 수집 대장

| 컬럼 | 타입 | 설명 |
|---|---|---|
| `trade_date` | date PK | |
| `rows` | integer | **0 = 휴장으로 확인됨** |
| `fetched_at` | timestamptz | |
| `status` | text | `ok` · `empty` · `error` |

#### `rows = 0` 은 "행이 없음"과 다릅니다

`0` 은 **휴장일 마커**입니다. 재요청을 억제하는 데 쓰입니다.
행 자체가 없는 것은 "아직 안 받아 봤다"는 뜻입니다. 둘을 같게 다루면 휴장일마다
KRX 를 계속 두드리게 됩니다.

#### ⚠️ `ohlcv` 행수와 `sync_log` 의 rows 합은 다릅니다 — 정상입니다

원본 실측(2026-08-25):

| | 값 |
|---|---|
| Supabase `ohlcv` (core 350종목) | **103,663행** |
| `ohlcv_sync_log` 의 `rows` 합 (전종목) | **821,928** |

`sync_log` 는 종목이 아니라 **거래일** 단위이고 "KRX 가 그 날 몇 행을 줬나"를
기록합니다. 종목으로 거르면 `rows` 가 뜻을 잃고 휴장일 마커 규칙까지 흔들립니다.
**이 어긋남이 정상입니다.**

---

### 2.5 `watermark` — 소스 단위 동기화 지점

| 컬럼 | 타입 | 설명 |
|---|---|---|
| `source` | text PK | `krx_ohlcv` · `dart_financials` · `industry_map` … |
| `last_synced_at` | timestamptz | |
| `last_key` | text | 재개 지점 (corp_code · 날짜 · 커서) |
| `rows` | bigint | 마지막 수집 행 수 |
| `status` | text | `ok` · `partial` · `error` |

`ohlcv_sync_log` 가 **거래일 단위**라면 이쪽은 **소스 단위**입니다.
날짜로 나뉘지 않는 수집물(DART 재무·산업분류·종목마스터)이 여기 들어옵니다.

> 📌 현재 `watermark` 를 읽고 쓰는 코드는 **이관하지 않았습니다**
> (DART 수집기 전용이라 1차 범위 밖). 스키마만 남겨 두었습니다.

---

## 3. 현재 데이터 규모 (**우리 저장소** 실측 · 2026-08-26)

> ✅ **갱신됨.** 이전 판은 팀장 개인 원본 저장소 수치(297거래일 · 821,928행)를 적었습니다.
> 이제 우리 저장소에 실제로 들어온 값입니다 ([ADR-AS-0003](decisions/0003-수집-계층.md)).

### 지수 — 1차 예측 대상 ★

| 항목 | 값 |
|---|---|
| **코스피 200** | **4,097거래일** (2010-01-04 ~ 2026-08-25) |
| 지수 전체 | 51종 · **195,864행** |
| 수집 대장 | 4,343일 (그중 휴장 확인 246일) |

⚠️ 이 표는 `index_price` 표이고, 아래 ERD 의 `ohlcv` 와 **다른 표**입니다.
지수는 하루 1행이고 가격이 **실수**(1096.25)라 스키마가 갈립니다.

### 개별 종목

| 항목 | 값 |
|---|---|
| 시세 | **7,726,776행** · 3,339거래일 · **3,413종목** (2013-01-16 ~ 2026-08-25) |
| core 유니버스 | **350종목** (KOSPI200 200 + KOSDAQ150 150) |

> 📌 **2010-01-04 가 KRX Open API 의 제공 시작일입니다.** 그 이전은 예외가 아니라
> **0행**으로 조용히 돌아옵니다(실측: `20091230` → 0행). 개별종목은 아직
> 2013-01-16 까지만 차 있고, 나머지 약 1,000일은 한도 리셋 후 이어받으면 됩니다.

> 📌 **얇음 문제는 해소됐습니다.** 4,097거래일이면 `min_train=120` 으로도 폴드가
> 충분히 나옵니다. 검증표본 1,230일 기준 기준선 **52.72%** 를 유의하게 이기려면
> **55.07%** 가 필요합니다 — 재현: `python scripts/check_index_data.py`

---

## 4. 1차 프로젝트가 실제로 쓰는 것

| 테이블 | 쓰임 |
|---|---|
| `ohlcv` | ★ 학습 데이터. 피처와 레이블이 전부 여기서 나온다 |
| `securities` | 유니버스 선택 · 생존 편향 처리 |
| `trading_calendar` | 거래일 정렬 · 결측 판정 |
| `ohlcv_sync_log` | 데이터 신선도 확인 |
| `watermark` | (미사용 — 스키마만) |

---

## 5. 아직 정해지지 않은 것

- 팀 Supabase 를 팔 것인가, 이 스키마를 그대로 쓸 것인가 → [회의안건 A-1](회의안건.md)
- 무료 티어 용량으로 충분한가 (전종목이면 821,928행)
- 피처 표를 **테이블로 저장**할 것인가, 매번 계산할 것인가
  → 저장한다면 `features` 테이블 설계가 필요합니다
- 예측 결과·백테스트 성과를 DB 에 남길 것인가, 파일로 둘 것인가
  → 대시보드 연결 방식과 함께 정합니다 → [회의안건 B-3](회의안건.md)
