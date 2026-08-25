-- ohlcv 연 단위 파티션 생성
--
-- 파티션을 미리 만들어 두지 않으면 그 연도의 INSERT 가
-- "no partition of relation ohlcv found for row" 로 실패한다.
--
-- 범위를 2015~2027 로 잡은 근거:
--   · 백테스트 목표 구간이 10년이다 (CONTEXT §4 거래세율 표가 2016~2026 을 다룬다)
--   · 현재 SQLite 캐시는 2025-06-09 ~ 2026-07-31 뿐이지만(실측 282거래일),
--     과거를 채우는 순간 파티션이 없으면 수집이 통째로 멈춘다
--   · 미래는 내년 것 하나만 미리 연다. 더 열어 두면 빈 파티션이 늘 뿐이다
--
-- 해가 바뀌기 전에 다음 파티션을 열어야 한다. 아래 DO 블록을 다시 돌리면
-- 이미 있는 것은 건너뛰고 없는 것만 만든다.

DO $$
DECLARE
  y integer;
BEGIN
  FOR y IN 2015..2027 LOOP
    EXECUTE format(
      'CREATE TABLE IF NOT EXISTS ohlcv_%s PARTITION OF ohlcv
         FOR VALUES FROM (%L) TO (%L)',
      y,
      make_date(y, 1, 1),
      make_date(y + 1, 1, 1)
    );
  END LOOP;
END $$;

-- 범위 밖 데이터를 받아 내는 그물. 이게 없으면 2014년 이전이나 2028년 이후 행이
-- 들어올 때 INSERT 가 실패한다. 여기 뭔가 쌓이면 파티션을 안 만든 것이므로
-- 정기적으로 비어 있는지 확인한다.
CREATE TABLE IF NOT EXISTS ohlcv_default PARTITION OF ohlcv DEFAULT;

COMMENT ON TABLE ohlcv_default IS
  '파티션 범위 밖 행을 받는 기본 파티션. 비어 있어야 정상이다.';
