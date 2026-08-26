"""지수 수집·저장 계층 테스트

네트워크를 타지 않는다. KRX 실제 응답을 그대로 붙여 넣은 표본으로 정규화를 검사하고,
저장·조회는 임시 DB 파일에서 확인한다.

**왜 이 테스트가 필요한가.** 여기서 틀리면 **에러가 안 난다** —
지수 가격이 정수로 깎이거나, 시계열이 뒤집혀 나오거나, 가격 없는 행이 섞여도
파이프라인은 그냥 돈다. 나중에 성능 숫자만 조용히 이상해진다.
"""

import sqlite3
from datetime import date

import pytest

from ingest.clients import krx_data as api

# ──────────────────────────────────────────────────────────────
# KRX 실제 응답 표본 (2026-08-21 · idx/kospi_dd_trd)
# 필드명·값을 손대지 않고 그대로 옮겼다. 스펙이 아니라 실물이 기준이다.
# ──────────────────────────────────────────────────────────────
REAL_ROW_KOSPI200 = {
    "BAS_DD": "20260821", "IDX_CLSS": "KOSPI", "IDX_NM": "코스피 200",
    "CLSPRC_IDX": "1,096.25", "CMPPREVDD_IDX": "15.27", "FLUC_RT": "1.41",
    "OPNPRC_IDX": "1,066.22", "HGPRC_IDX": "1,103.80", "LWPRC_IDX": "1,063.52",
    "ACC_TRDVOL": "124576013", "ACC_TRDVAL": "24353087983152",
    "MKTCAP": "5308304168709510",
}

# 가격 필드가 **빈 문자열**로 오는 실제 행. 이런 지수가 실재한다.
REAL_ROW_EMPTY_PRICE = {
    "BAS_DD": "20260821", "IDX_CLSS": "KOSPI", "IDX_NM": "코스피 (외국주포함)",
    "CLSPRC_IDX": "", "CMPPREVDD_IDX": "", "FLUC_RT": "",
    "OPNPRC_IDX": "", "HGPRC_IDX": "", "LWPRC_IDX": "",
    "ACC_TRDVOL": "408037433", "ACC_TRDVAL": "28897809870542",
    "MKTCAP": "5712979436874263",
}


# ==============================================================
# 1. 정규화
# ==============================================================
def test_지수_가격을_실수로_보존한다():
    """정수로 깎으면 하루 등락이 통째로 사라진다 — 그런데 에러는 안 난다."""
    item = api.normalize_index_row(REAL_ROW_KOSPI200, "20260821")

    assert item["close"] == 1096.25
    assert item["open"] == 1066.22
    assert item["high"] == 1103.80
    assert item["low"] == 1063.52
    assert item["change"] == 15.27
    assert item["change_rate"] == 1.41
    # 소수점이 살아 있는지를 타입으로도 못 박는다
    for field in ("open", "high", "low", "close", "change", "change_rate"):
        assert isinstance(item[field], float), f"{field} 가 실수가 아니다"


def test_거래량과_시가총액은_정수로_바꾼다():
    item = api.normalize_index_row(REAL_ROW_KOSPI200, "20260821")

    assert item["volume"] == 124_576_013
    assert item["value"] == 24_353_087_983_152
    assert item["market_cap"] == 5_308_304_168_709_510
    assert isinstance(item["volume"], int)


def test_가격이_빈_지수도_행을_버리지_않는다():
    """'코스피 (외국주포함)' 는 거래량만 온다. 행을 버리면 왜 없는지를 알 수 없게 된다."""
    item = api.normalize_index_row(REAL_ROW_EMPTY_PRICE, "20260821")

    assert item["index_name"] == "코스피 (외국주포함)"
    assert item["close"] is None          # 값만 None
    assert item["open"] is None
    assert item["volume"] == 408_037_433  # 거래량은 살아 있다


def test_날짜를_ISO_로_바꾼다():
    item = api.normalize_index_row(REAL_ROW_KOSPI200, "20260821")
    assert item["date"] == "2026-08-21"


def test_예측_대상_지수명은_띄어쓰기까지_맞아야_한다():
    """'코스피200' 이 아니라 '코스피 200' 이다. 한 글자 틀리면 조회가 조용히 0행이 된다."""
    assert api.TARGET_INDEX == "코스피 200"
    assert api.normalize_index_row(REAL_ROW_KOSPI200, "20260821")["index_name"] == api.TARGET_INDEX


def test_지수_엔드포인트가_두_시장을_안다():
    assert set(api.INDEX_APIS) == {"KOSPI", "KOSDAQ"}
    assert api.INDEX_APIS["KOSPI"][0] == "idx/kospi_dd_trd"


def test_모르는_시장은_무엇을_쓸_수_있는지_알려준다():
    """막다른 길로 만들지 않는다 — 예외 메시지에 쓸 수 있는 값이 들어 있어야 한다."""
    with pytest.raises(api.KrxError) as caught:
        api.fetch_index_snapshot("20260821", "NASDAQ")
    assert "KOSPI" in str(caught.value)


def test_날짜_형식이_틀리면_네트워크를_타기_전에_막는다():
    with pytest.raises(api.KrxError):
        api.fetch_index_snapshot("2026-08-21", "KOSPI")


# ==============================================================
# 2. 저장·조회
# ==============================================================
@pytest.fixture
def 임시저장소(tmp_path, monkeypatch):
    """진짜 DB 를 건드리지 않도록 임시 파일로 갈아 끼운다."""
    from ingest.store import krx_index, krx_store

    db = tmp_path / "test_index.db"
    monkeypatch.setattr(krx_store, "DB_PATH", db)
    monkeypatch.setattr(krx_index, "DB_PATH", db)
    krx_index.init_db()
    return krx_index


def _행(bas_dd: str, close: float, name: str = "코스피 200") -> dict:
    return {"index_name": name, "index_class": "KOSPI",
            "open": close, "high": close, "low": close, "close": close,
            "change": 0.0, "change_rate": 0.0,
            "volume": 1, "value": 1, "market_cap": 1}


def test_시계열은_과거에서_현재_순으로_나온다(임시저장소):
    """뒤집혀 나오면 이동평균·차분이 전부 틀리는데 **예외가 나지 않는다.**"""
    store = 임시저장소
    # 일부러 뒤죽박죽 순서로 넣는다
    for bas_dd, close in (("20260103", 100.0), ("20260101", 98.0), ("20260102", 99.0)):
        store._save(bas_dd, "KOSPI", [_행(bas_dd, close)])

    rows = store.series("코스피 200")

    assert [r["date"] for r in rows] == ["2026-01-01", "2026-01-02", "2026-01-03"]
    assert [r["close"] for r in rows] == [98.0, 99.0, 100.0]


def test_가격이_없는_행은_시계열에서_빠진다(임시저장소):
    """섞이면 수익률이 NaN 으로 오염된다."""
    store = 임시저장소
    store._save("20260101", "KOSPI", [
        _행("20260101", 100.0),
        {**_행("20260101", 0.0, name="코스피 (외국주포함)"),
         "open": None, "high": None, "low": None, "close": None},
    ])

    assert len(store.series("코스피 200")) == 1
    assert store.series("코스피 (외국주포함)") == []


def test_같은_날짜를_다시_받아도_중복되지_않는다(임시저장소):
    store = 임시저장소
    store._save("20260101", "KOSPI", [_행("20260101", 100.0)])
    store._save("20260101", "KOSPI", [_행("20260101", 111.0)])   # 다시 받았다

    rows = store.series("코스피 200")
    assert len(rows) == 1
    assert rows[0]["close"] == 111.0      # 나중 값으로 덮인다


def test_받은_날짜는_다시_요청하지_않는다(임시저장소):
    store = 임시저장소
    store._save("20200102", "KOSPI", [_행("20200102", 100.0)])

    assert "20200102" in store.fetched_dates("KOSPI")
    # 시장이 다르면 별개다 — 하나로 합치면 두 백필이 서로의 진행을 지운다
    assert "20200102" not in store.fetched_dates("KOSDAQ")


def test_최근_0건은_다시_확인한다(임시저장소):
    """휴장일은 영원히 0건이지만, 오늘의 0건은 장이 안 끝난 것일 수 있다."""
    store = 임시저장소
    오늘 = date.today().strftime("%Y%m%d")
    store._save(오늘, "KOSPI", [])            # 0건으로 기록
    store._save("20200102", "KOSPI", [])      # 오래된 0건

    fetched = store.fetched_dates("KOSPI")
    assert 오늘 not in fetched                 # 최근 0건 → 다시 받는다
    assert "20200102" in fetched               # 오래된 0건 → 확정 휴장


def test_제공_시작일_이전은_요청하지_않는다(임시저장소, monkeypatch):
    """2010-01-04 이전은 0행으로 조용히 돌아온다 — 아예 부르지 않는다."""
    store = 임시저장소
    assert store.DATA_START == "20100104"

    # sync 가 만드는 대상 목록에 경계 이전 날짜가 없어야 한다.
    # 네트워크를 타지 않도록 fetch_date 를 가로챈다.
    # ⚠️ monkeypatch 로 갈아야 한다 — 직접 대입하면 테스트가 끝나도 모듈이 변조된
    #    채로 남아 뒤따르는 테스트가 조용히 가짜 함수를 쓴다.
    called: list = []
    monkeypatch.setattr(store, "fetch_date",
                        lambda bas_dd, market: called.append(bas_dd) or 0)
    store.sync(days=30, workers=1, end="20100115", markets=("KOSPI",))

    assert called, "아무것도 부르지 않았다 — 테스트 자체가 틀렸다"
    assert all(d >= "20100104" for d in called), f"경계 이전을 불렀다: {sorted(called)[:5]}"


def test_지수_가격이_DB_에서도_실수로_남는다(임시저장소):
    """SQLite 컬럼이 INTEGER 면 1096.25 가 1096 으로 깎인다."""
    store = 임시저장소
    store._save("20260821", "KOSPI", [_행("20260821", 1096.25)])

    with sqlite3.connect(store.DB_PATH) as conn:
        value = conn.execute(
            "SELECT close FROM index_price WHERE index_name = ?", ("코스피 200",)
        ).fetchone()[0]

    assert value == 1096.25


# ==============================================================
# 3. 수집 대장 — 왜 안 받았는지가 남는가
# ==============================================================
def test_받은_날은_대장에_성공으로_남는다(임시저장소):
    from ingest.store import collect_log

    store = 임시저장소
    store._save("20260821", "KOSPI", [_행("20260821", 1096.25)])

    행 = collect_log.entry("krx_index", "KOSPI/20260821", db_path=store.DB_PATH)
    assert 행["status"] == collect_log.OK
    assert 행["rows"] == 1


def test_휴장일은_실패가_아니라_0건으로_남는다(임시저장소):
    """실패로 남기면 배치를 돌릴 때마다 같은 날짜에 호출을 태운다."""
    from ingest.store import collect_log

    store = 임시저장소
    store._save("20260815", "KOSPI", [])       # 광복절

    행 = collect_log.entry("krx_index", "KOSPI/20260815", db_path=store.DB_PATH)
    assert 행["status"] == collect_log.EMPTY
    assert 행["attempts"] == 0                  # 재시도 횟수를 먹지 않았다


def test_저장이_되돌아가면_대장도_되돌아간다(임시저장소, monkeypatch):
    """적재만 롤백되고 대장이 남으면 그 날짜를 영영 다시 안 받는다."""
    from ingest.store import collect_log

    store = 임시저장소
    # ⚠️ 경로를 **미리** 잡아 둔다. `monkeypatch.undo()` 는 이 테스트가 건 것뿐 아니라
    #    fixture 가 건 DB_PATH 교체까지 함께 되돌려, 그 뒤 검증이 진짜 DB 를 읽는다.
    db = store.DB_PATH

    # 대장을 쓴 **직후**에 터뜨린다 — 커밋 직전이라 둘 다 되돌아가야 한다.
    원래 = collect_log.mark_ok

    def 쓰고_터뜨린다(*args, **kwargs):
        원래(*args, **kwargs)
        raise RuntimeError("적재 도중 죽었다")

    monkeypatch.setattr(collect_log, "mark_ok", 쓰고_터뜨린다)
    with pytest.raises(RuntimeError):
        store._save("20260821", "KOSPI", [_행("20260821", 1096.25)])

    monkeypatch.setattr(collect_log, "mark_ok", 원래)
    assert collect_log.entry("krx_index", "KOSPI/20260821", db_path=db) is None
    with sqlite3.connect(db) as conn:
        assert conn.execute("SELECT COUNT(*) FROM index_price").fetchone()[0] == 0


def test_제공_시작일_이전은_대장에_이유가_남는다(임시저장소, monkeypatch):
    """이게 없으면 나중에 '2009년이 왜 비어 있나'에 아무도 답할 수 없다."""
    from ingest.store import collect_log

    store = 임시저장소
    monkeypatch.setattr(store, "fetch_date", lambda bas_dd, market: 0)
    store.sync(days=30, workers=1, end="20100115", markets=("KOSPI",))

    행 = collect_log.entry("krx_index", "KOSPI/20091230", db_path=store.DB_PATH)
    assert 행 is not None, "요청하지 않았다는 사실 자체가 안 남았다"
    assert 행["status"] == collect_log.OUT_OF_RANGE
    assert "20100104" in 행["note"]             # 경계가 어디인지까지 적혀 있다


def test_한도_소진은_실패로_세지_않는다(임시저장소, monkeypatch):
    """예산이 마른 것은 그 날짜의 잘못이 아니다 — 재시도 횟수를 먹으면 안 된다."""
    from ingest.clients import krx_data
    from ingest.store import collect_log

    store = 임시저장소

    def 한도소진(bas_dd, market):
        raise krx_data.KrxQuotaExhausted("오늘 쓸 수 있는 KRX 호출을 다 썼습니다.")

    monkeypatch.setattr(store, "fetch_date", 한도소진)
    결과 = store.sync(days=3, workers=1, end="20260821", markets=("KOSPI",))

    assert 결과["quota_exhausted"] == 3
    assert 결과["failed"] == [], "한도 소진이 실패로 세어졌다"

    행 = collect_log.entry("krx_index", "KOSPI/20260821", db_path=store.DB_PATH)
    assert 행["status"] == collect_log.QUOTA_EXHAUSTED
    assert 행["attempts"] == 0
    # 예산이 풀리면 다시 받아야 한다
    assert collect_log.should_collect("krx_index", "KOSPI/20260821",
                                      db_path=store.DB_PATH) is True
