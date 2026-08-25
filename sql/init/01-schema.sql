-- data-service — 저장 계층 스키마
--
-- 이 파일은 `docker compose --profile local-db up -d` 로 Postgres 컨테이너가
-- **처음 생성될 때만** 실행된다(볼륨이 비어 있을 때). 이미 만든 볼륨에 다시 적용하려면
-- `docker compose down -v` 로 볼륨을 지우고 다시 띄운다.
--
-- ─────────────────────────────────────────────────────────────────────────────
-- 이 스키마는 백지 설계가 아니다. 지금 돌고 있는 SQLite 캐시를 옮겨 온 것이다.
--
--   app/repositories/krx_store.py:81-111   daily_price · fetch_log   ← 원본
--   scripts/build_krx_bundle.py:75-83      stock 표                  ← securities 선례
--   data/krx_derived.json                  trading_days 282일        ← 거래일 캘린더
--   data/industry_map.json                 3,925종목 산업분류         ← securities 확장
--
-- 옮기면서 **반드시 살려야 하는 규칙이 하나** 있다 — 아래 ohlcv_sync_log 주석 참조.
-- ─────────────────────────────────────────────────────────────────────────────

BEGIN;

-- ============================================================================
-- 1. securities — 종목 마스터
-- ============================================================================
-- 현재 SQLite `daily_price` 는 name·market·sector·listed_shares 를 780,484행마다
-- 반복해 담는다(krx_store.py:84-97). 번들 빌드는 이미 그것을 `stock` 표로 분리해
-- 뒀으므로(build_krx_bundle.py:75-83) 그 정의를 그대로 옮긴다.
--
-- 키를 code(text)가 아니라 security_id(integer)로 두는 이유는 세 가지다 (ADR-CT-0009).
--   ① 우선주 — 005930(보통주)과 005935(우선주)는 다른 종목이다
--   ② 종목코드 변경 — 액면분할·상호변경으로 코드가 바뀌어도 같은 기업이다
--   ③ 상장폐지 후 코드 재사용 — 같은 코드가 다른 기업을 가리키게 된다
-- code 를 PK 로 쓰면 이 셋에서 조용히 깨진다.
CREATE TABLE securities (
  security_id    integer GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
  code           text        NOT NULL,          -- 종목코드 6자리 (KRX 기준)
  name           text        NOT NULL,
  market         text,                          -- KOSPI · KOSDAQ · KONEX
  sector         text,                          -- 소속부
  listed_shares  bigint,                        -- 상장주식수
  industry_code  text,                          -- DART 표준산업분류 (industry_map.json)
  fiscal_month   smallint,                      -- 결산월
  corp_code      text,                          -- DART 고유번호 (corp_code.json 매핑)

  -- 유니버스 2단계 (ADR-CT-0010).
  --   core = KOSPI200 + KOSDAQ150 (약 350) — 로컬 + Supabase 양쪽
  --   full = 전종목 + 상장폐지            — 로컬 전용
  universe_tier  text        NOT NULL DEFAULT 'full'
                             CHECK (universe_tier IN ('core', 'full')),

  -- 상장폐지 종목을 지우지 않고 남긴다. 지우면 백테스트에 생존 편향이 생긴다.
  is_delisted    boolean     NOT NULL DEFAULT false,
  delisted_date  date,

  created_at     timestamptz NOT NULL DEFAULT now(),
  updated_at     timestamptz NOT NULL DEFAULT now()
);

-- 같은 코드가 동시에 두 번 살아 있을 수는 없다. 다만 상장폐지 뒤 재사용은 허용해야
-- 하므로 '살아 있는 것'에 한해서만 유일성을 건다.
CREATE UNIQUE INDEX securities_code_active_uq
  ON securities (code) WHERE NOT is_delisted;

CREATE INDEX securities_code_idx         ON securities (code);
CREATE INDEX securities_universe_idx     ON securities (universe_tier) WHERE NOT is_delisted;
CREATE INDEX securities_corp_code_idx    ON securities (corp_code) WHERE corp_code IS NOT NULL;

COMMENT ON TABLE  securities IS '종목 마스터. SQLite daily_price 의 반복 컬럼과 krx_bundle.stock 표를 흡수한다.';
COMMENT ON COLUMN securities.universe_tier IS 'core=KOSPI200+KOSDAQ150 / full=전종목+상장폐지 (ADR-CT-0010)';


-- ============================================================================
-- 2. trading_calendar — 거래일 달력
-- ============================================================================
-- ★ 이 표는 신설이다. 코드에 대응물이 없다.
--
-- CONTEXT.md 는 "core/trading_calendar.py — 한국 휴장일 계산 이미 구현됨"이라 적었지만
-- 실물은 `day.weekday() < 5` 로 **주말만** 거르고, docstring 이 스스로
-- "⚠️ 공휴일은 반영하지 않는다"고 밝힌다(app/core/trading_calendar.py:41-43).
-- 실제 휴장 판정은 세 곳에 근사치로 흩어져 있었다 —
--   ① krx_store.fetch_log 의 0건 기록
--   ② preprocess/calendar.py:40 HOLIDAY_RUN_MAX = 5 (평일 공백 5일 이하를 연휴로 간주)
--   ③ data/krx_derived.json 의 파생 거래일 282일
-- 그것을 한 곳으로 모은다.
CREATE TABLE trading_calendar (
  trade_date  date        PRIMARY KEY,
  is_open     boolean     NOT NULL,

  -- 이 판정이 어디서 왔는지 밝힌다. 근사치를 사실처럼 쓰지 않기 위해서다.
  --   krx_observed   실제 시세가 있었다 (가장 신뢰도 높음)
  --   krx_zero_rows  KRX 가 0건을 돌려줬다 → 휴장으로 본다
  --   weekday_approx 주말 규칙만 적용한 추정 (공휴일 미반영)
  source      text        NOT NULL
                          CHECK (source IN ('krx_observed', 'krx_zero_rows', 'weekday_approx')),
  note        text,
  updated_at  timestamptz NOT NULL DEFAULT now()
);

CREATE INDEX trading_calendar_open_idx ON trading_calendar (trade_date) WHERE is_open;

COMMENT ON TABLE trading_calendar IS
  '거래일 달력. 신설 — 기존 trading_calendar.py 는 주말만 걸렀고 공휴일을 몰랐다.';


-- ============================================================================
-- 3. ohlcv — 일별 시세 (연 단위 RANGE 파티셔닝)
-- ============================================================================
-- OHLC 를 integer 로 둔다. 국내 주가는 원 단위 정수라 numeric 이 필요 없고,
-- 이미 SQLite 쪽도 INTEGER 다(krx_store.py:87-97). numeric 대비 약 25% 절약.
--
-- 파티션 키는 trade_date. 오래된 연도를 통째로 detach 할 수 있어야 용량 관리가 된다.
-- 파티션 테이블의 PK 는 파티션 키를 반드시 포함해야 하므로 (security_id, trade_date).
CREATE TABLE ohlcv (
  security_id  integer  NOT NULL REFERENCES securities(security_id),
  trade_date   date     NOT NULL,

  open         integer  NOT NULL,
  high         integer  NOT NULL,
  low          integer  NOT NULL,
  close        integer  NOT NULL,

  -- 전일대비. close 차이로 역산하면 3원씩 어긋나는 종목이 있어 원본을 그대로 받는다
  -- (scripts/build_krx_bundle.py:64-67 에 그 실측이 남아 있다).
  change       integer,
  -- 등락률(%) — 유일하게 정수가 아니다.
  -- numeric(8,4) 는 정수부가 4자리뿐이라 실측 원본 1행이 들어가지 않는다:
  --   20260209 · 052670 제일바이오 · close 625,000 · change 622,920 · rate 29948.08
  -- 액면병합·재상장으로 보이고 같은 행의 close·change 와 산술적으로 일관된다. 즉 원본이
  -- 옳고 그릇이 좁았다. 값을 깎는 대신 그릇을 넓힌다 —
  -- `scripts/check_migration_fitness.py` 가 매번 이 한계를 다시 잰다.
  change_rate  numeric(12, 4),

  volume       bigint,
  value        bigint,                       -- 거래대금
  market_cap   bigint,                       -- 시가총액 (수집 시점 기준)

  -- 그 거래일의 상장주식수 (ADR-DS-0010). securities.listed_shares 는 **최신** 값이라
  -- 역할이 다르다 — 둘이 갈리는 것이 정상이고, 갈렸다고 동기화 오류로 읽지 않는다.
  -- 종목당 한 줄로 접으면 액면분할 종목의 과거 회전율이 10배 틀린다(실측 4종목).
  -- market_cap / close 로 역산하면 오늘은 780,484행 전부 정확하지만, 그 항등식이
  -- 깨지는 날 역산은 예외 대신 **그럴듯하게 틀린 값**을 낸다. change 컬럼과 같은 이유로
  -- 원본을 그대로 싣는다. 항등식 자체는 check_migration_fitness.py 가 계속 잰다.
  listed_shares bigint,

  PRIMARY KEY (security_id, trade_date)
) PARTITION BY RANGE (trade_date);

-- 종목 하나의 시계열을 훑는 조회가 가장 잦다(캔들·이동평균·수익률).
-- PK 가 (security_id, trade_date) 이므로 그 순서 조회는 PK 로 커버된다.
-- 날짜 하나의 전종목 스냅샷(`/api/krx/stocks`)을 위해 역순 인덱스를 하나만 더 둔다.
-- **인덱스 추가는 곧 용량 추가다.** 필요해질 때까지 늘리지 않는다.
CREATE INDEX ohlcv_trade_date_idx ON ohlcv (trade_date);

COMMENT ON TABLE  ohlcv IS '일별 시세. SQLite daily_price 이식. 연 단위 RANGE 파티셔닝.';
COMMENT ON COLUMN ohlcv.close IS '종가(원). 국내 주가는 정수라 numeric 을 쓰지 않는다.';


-- ============================================================================
-- 4. ohlcv_sync_log — 거래일 단위 수집 대장  ★ 가장 중요한 이식물
-- ============================================================================
-- SQLite `fetch_log(bas_dd, rows, fetched_at)` 를 그대로 옮긴 것이다
-- (krx_store.py:106-110).
--
-- ★★ 이 표에는 반드시 살려야 하는 규칙이 하나 있다.
--
--    **rows = 0 은 "아직 안 받았다"가 아니라 "받아 봤더니 없었다"** 는 뜻이다.
--    즉 휴장일 마커다. krx_store.fetched_dates()(:160-173)가
--        SELECT bas_dd FROM fetch_log WHERE rows > 0 OR bas_dd < (오늘 - 7일)
--    로 '다시 받을 필요 없는 날짜'를 계산한다. 7일(ZERO_ROW_RETRY_DAYS, :157)이 지나면
--    0건을 확정으로 본다 — 그 전에는 데이터가 늦게 올라온 것일 수 있기 때문이다.
--
--    이 규칙을 빼고 "받은 날짜만 기록"하는 단순한 워터마크로 바꾸면,
--    **확정된 휴장일을 매 수집마다 KRX 에 다시 물어보게 된다.** 설·추석·공휴일이
--    연간 15일 안팎이므로 10년이면 150번의 헛된 왕복이 매 수집마다 생긴다.
CREATE TABLE ohlcv_sync_log (
  trade_date  date        PRIMARY KEY,
  rows        integer     NOT NULL,      -- 0 = 휴장으로 확인됨 (없는 행과 다르다)
  fetched_at  timestamptz NOT NULL DEFAULT now(),
  status      text        NOT NULL DEFAULT 'ok'
                          CHECK (status IN ('ok', 'empty', 'error')),
  note        text
);

COMMENT ON TABLE  ohlcv_sync_log IS 'SQLite fetch_log 이식. 거래일 단위 수집 대장.';
COMMENT ON COLUMN ohlcv_sync_log.rows IS
  '0 은 휴장일 마커다 — 재요청 억제에 쓰인다. 행이 없는 것과 의미가 다르다.';


-- ============================================================================
-- 5. watermark — 소스별 최신 동기화 지점
-- ============================================================================
-- 위 ohlcv_sync_log 가 '거래일 단위'라면 이쪽은 '소스 단위'다. 날짜로 나뉘지 않는
-- 수집물(DART 재무·산업분류·종목마스터·사업보고서 색인)이 여기 들어온다.
--
-- 지금은 이 정보가 파일 안에 흩어져 있다 —
--   data/krx_derived.json      generated_at
--   data/market_snapshot.json.gz  as_of · meta.generated_at
--   data/industry_map.json     generated_at
--   report_index.db  meta      facts_built_at · embed_built_at
-- 전부 '사람이 스크립트를 다시 돌려야 갱신되는' 수동 산출물이고, 그 사실이 어디에도
-- 기계가 읽을 수 있는 형태로 없다. 여기 모아서 신선도를 판정할 수 있게 한다.
CREATE TABLE watermark (
  source          text        PRIMARY KEY,   -- 'krx_ohlcv' · 'dart_financials' · 'industry_map' ...
  last_synced_at  timestamptz NOT NULL,
  last_key        text,                      -- 재개 지점 (corp_code · 날짜 · 커서 등)
  rows            bigint,                    -- 마지막 수집이 담은 행 수
  status          text        NOT NULL DEFAULT 'ok'
                              CHECK (status IN ('ok', 'partial', 'error')),
  note            text,
  updated_at      timestamptz NOT NULL DEFAULT now()
);

COMMENT ON TABLE watermark IS
  '소스별 최신 동기화 지점. 거래일 단위 대장은 ohlcv_sync_log 가 따로 맡는다.';

COMMIT;
