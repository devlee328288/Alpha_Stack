"""응답 원문 보존 테스트.

**무엇을 지키려는 테스트인가.** 이 표의 목적은 하나다 — *"다시 받지 않고 다시 정규화할
수 있는가."* 그러려면 보존된 것이 **진짜 원문**이어야 한다. 한 글자라도 달라지면
재정규화 결과가 원래와 달라지는데, **예외는 안 난다.** 값만 조용히 틀린다.

    수용 기준
    - 바이트를 그대로 돌려준다 (인코딩 추측이 끼어들지 않는다) ← 핵심
    - euc-kr 처럼 UTF-8 이 아닌 원문도 깨지지 않는다
    - 크롤링 계열 출처는 **저장 자체를 거부한다** (저작권)
    - 지문은 **압축 전** 원문으로 잡는다 (압축은 라이브러리 버전을 탄다)
    - 손상된 원문을 조용히 돌려주지 않는다
    - 다시 받으면 덮지 않고 쌓인다 (출처가 값을 정정한 이력이 증거다)
"""

from __future__ import annotations

import gzip
import hashlib
import json

import pytest

from common import raw_store


@pytest.fixture()
def db(tmp_path):
    return tmp_path / "t.db"


# ── 원문이 원문으로 남는가 ───────────────────────────────────────────────────

def test_바이트를_그대로_돌려준다(db):
    원문 = b'{"OutBlock_1": [{"IDX_NM": "\xec\xbd\x94\xec\x8a\xa4\xed\x94\xbc 200"}]}'

    raw_store.save("krx", "idx/kospi_dd_trd/20260826", 원문, encoding="utf-8", db_path=db)

    보존 = raw_store.load("krx", "idx/kospi_dd_trd/20260826", db_path=db)
    assert 보존["body"] == 원문                 # 바이트 단위로 같다
    assert 보존["bytes"] == len(원문)


def test_UTF8_이_아닌_원문도_깨지지_않는다(db):
    """euc-kr 로 주는 출처가 실재한다. UTF-8 로 가정하면 글자만 조용히 깨진다."""
    글자 = "코스피 200 종목토론"
    원문 = 글자.encode("euc-kr")
    assert 원문 != 글자.encode("utf-8")          # 두 인코딩이 실제로 다른 바이트다

    raw_store.save("krx", "euckr/20260826", 원문, encoding="euc-kr", db_path=db)

    보존 = raw_store.load("krx", "euckr/20260826", db_path=db)
    assert 보존["body"] == 원문
    assert 보존["encoding"] == "euc-kr"
    # 보존된 인코딩으로 되돌리면 원래 글자가 나온다
    assert 보존["body"].decode(보존["encoding"]) == 글자


def test_문자열은_거부한다(db):
    """문자열로 받는 순간 이미 누군가 인코딩을 추측한 뒤다 — 그건 원문이 아니다."""
    with pytest.raises(TypeError) as 오류:
        raw_store.save("krx", "t/20260826", '{"a": 1}', db_path=db)

    assert "디코딩하기" in str(오류.value)       # 무엇을 해야 하는지까지 알려 준다


# ── 무엇을 저장해도 되는가 ───────────────────────────────────────────────────

def test_크롤링_계열은_저장을_거부한다(db):
    """기사 본문 전문 보관은 저작권 문제다. 조용히 넘어가면 안 된다."""
    with pytest.raises(raw_store.SourceNotAllowed) as 오류:
        raw_store.save("naver_news_article", "t/1", b"<html>...</html>", db_path=db)

    메시지 = str(오류.value)
    assert "저작권" in 메시지
    assert "보존 가능" in 메시지                 # 무엇은 되는지도 알려 준다


def test_허용된_출처만_통과한다(db):
    for source in raw_store.ALLOWED_SOURCES:
        raw_store.save(source, "t/20260826", b"{}", db_path=db)
    assert set(raw_store.stats(db_path=db)) == set(raw_store.ALLOWED_SOURCES)


def test_상한을_넘는_응답은_거부한다(db, monkeypatch):
    monkeypatch.setattr(raw_store, "MAX_BODY_BYTES", 10)

    with pytest.raises(raw_store.RawTooLarge):
        raw_store.save("krx", "t/20260826", b"x" * 11, db_path=db)


# ── 지문 ────────────────────────────────────────────────────────────────────

def test_지문은_압축_전_원문으로_잡는다(db):
    """압축 결과는 라이브러리 버전을 탄다. 그걸 지문으로 삼으면 언젠가 갈라진다."""
    원문 = b'{"a": 1}'

    지문 = raw_store.save("krx", "t/20260826", 원문, db_path=db)

    assert 지문 == hashlib.sha256(원문).hexdigest()
    assert 지문 != hashlib.sha256(gzip.compress(원문)).hexdigest()


def test_손상된_원문을_조용히_돌려주지_않는다(db):
    import sqlite3

    raw_store.save("krx", "t/20260826", b'{"a": 1}', db_path=db)

    # 저장된 본문만 바꿔치기한다 — 지문은 그대로 둔다
    conn = sqlite3.connect(db)
    conn.execute("UPDATE raw_response SET body=?", (gzip.compress(b'{"a": 2}'),))
    conn.commit()
    conn.close()

    with pytest.raises(ValueError) as 오류:
        raw_store.load("krx", "t/20260826", db_path=db)

    assert "손상" in str(오류.value)


# ── 여러 판 ─────────────────────────────────────────────────────────────────

def test_다시_받으면_덮지_않고_쌓인다(db):
    """출처가 과거 값을 정정하는 일이 실제로 있다. 무엇이 어떻게 바뀌었는지가 증거다."""
    raw_store.save("krx", "t/20260826", b'{"v": 1}',
                   fetched_at="2026-08-26T10:00:00+09:00", db_path=db)
    raw_store.save("krx", "t/20260826", b'{"v": 2}',
                   fetched_at="2026-08-27T10:00:00+09:00", db_path=db)

    assert raw_store.stats(db_path=db)["krx"]["responses"] == 2
    assert raw_store.stats(db_path=db)["krx"]["targets"] == 1
    # 안 주면 가장 최근 것
    assert raw_store.load("krx", "t/20260826", db_path=db)["body"] == b'{"v": 2}'
    # 옛 판도 그대로 꺼낼 수 있다
    옛판 = raw_store.load("krx", "t/20260826",
                          fetched_at="2026-08-26T10:00:00+09:00", db_path=db)
    assert 옛판["body"] == b'{"v": 1}'


def test_순회는_대상별_최신만_준다(db):
    raw_store.save("krx", "a/1", b'{"v": 1}', fetched_at="2026-08-26T10:00:00+09:00", db_path=db)
    raw_store.save("krx", "a/1", b'{"v": 2}', fetched_at="2026-08-27T10:00:00+09:00", db_path=db)
    raw_store.save("krx", "b/1", b'{"v": 3}', fetched_at="2026-08-26T10:00:00+09:00", db_path=db)

    전부 = list(raw_store.iter_latest("krx", db_path=db))
    assert [r["body"] for r in 전부] == [b'{"v": 2}', b'{"v": 3}']

    # 앞머리로 좁힐 수 있다 — 지수만 다시 정규화하고 싶을 때 쓴다
    좁힘 = list(raw_store.iter_latest("krx", prefix="a/", db_path=db))
    assert [r["target"] for r in 좁힘] == ["a/1"]


# ── 재정규화 왕복 ────────────────────────────────────────────────────────────

def test_보존한_원문으로_같은_정규화_결과가_나온다(db):
    """이게 이 표의 존재 이유다 — 네트워크를 타지 않고 정규화를 다시 돌릴 수 있는가."""
    from ingest.clients import krx_data as api

    응답 = {"OutBlock_1": [{
        "BAS_DD": "20260821", "IDX_CLSS": "KOSPI", "IDX_NM": "코스피 200",
        "CLSPRC_IDX": "1,096.25", "CMPPREVDD_IDX": "15.27", "FLUC_RT": "1.41",
        "OPNPRC_IDX": "1,066.22", "HGPRC_IDX": "1,103.80", "LWPRC_IDX": "1,063.52",
        "ACC_TRDVOL": "124576013", "ACC_TRDVAL": "24353087983152",
        "MKTCAP": "5308304168709510",
    }]}
    원문 = json.dumps(응답, ensure_ascii=False).encode("utf-8")
    처음 = api.normalize_index_row(응답["OutBlock_1"][0], "20260821")

    raw_store.save("krx", "idx/kospi_dd_trd/20260821", 원문,
                   encoding="utf-8", db_path=db)

    # 원문만 가지고 다시 정규화한다
    보존 = raw_store.load("krx", "idx/kospi_dd_trd/20260821", db_path=db)
    다시 = api.normalize_index_row(
        json.loads(보존["body"].decode(보존["encoding"]))["OutBlock_1"][0], "20260821")

    assert 다시 == 처음
    assert 다시["close"] == 1096.25              # 소수점까지 살아 있다


def test_압축률을_함께_보고한다(db):
    """문서에 추측값을 적지 않으려면 실제로 얼마나 줄었는지 코드가 답해야 한다."""
    원문 = json.dumps({"OutBlock_1": [{"a": "1"} for _ in range(200)]}).encode()

    raw_store.save("krx", "t/20260826", 원문, db_path=db)

    현황 = raw_store.stats(db_path=db)["krx"]
    assert 현황["raw_bytes"] == len(원문)
    assert 현황["stored_bytes"] < 현황["raw_bytes"]   # 반복이 많은 JSON 은 잘 줄어든다
    assert 0 < 현황["ratio"] < 1


# ── 수집 경로에 실제로 붙었는가 ──────────────────────────────────────────────

def _가짜응답(payload: dict, charset: str = "utf-8"):
    import io

    body = json.dumps(payload, ensure_ascii=False).encode(charset)

    class _Headers:
        def get_content_charset(self):
            return charset

    class _Res(io.BytesIO):
        headers = _Headers()

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    return _Res(body)


@pytest.fixture()
def 가짜KRX(monkeypatch, db):
    """네트워크를 타지 않고 KRX 호출 한 번을 흉내 낸다."""
    from common import budget, settings
    from ingest.clients import krx_data as api

    응답 = {"OutBlock_1": [{
        "BAS_DD": "20260821", "IDX_CLSS": "KOSPI", "IDX_NM": "코스피 200",
        "CLSPRC_IDX": "1,096.25", "CMPPREVDD_IDX": "15.27", "FLUC_RT": "1.41",
        "OPNPRC_IDX": "1,066.22", "HGPRC_IDX": "1,103.80", "LWPRC_IDX": "1,063.52",
        "ACC_TRDVOL": "124576013", "ACC_TRDVAL": "24353087983152",
        "MKTCAP": "5308304168709510",
    }]}
    monkeypatch.setattr(api, "urlopen", lambda *a, **k: _가짜응답(응답))
    monkeypatch.setattr(api, "load_krx_key", lambda: ("테스트키", "test"))
    monkeypatch.setattr(budget, "try_spend", lambda *a, **k: True)
    api.reset_auth_block()
    # 원문은 테스트 DB 로 간다
    monkeypatch.setattr(raw_store, "save",
                        lambda *a, **k: _원래save(*a, **{**k, "db_path": db}))
    monkeypatch.setattr(settings, "keep_raw_enabled", lambda: True)
    return api


_원래save = raw_store.save


def test_수집하면_원문이_남는다(가짜KRX, db):
    """스위치만 있고 True 로 넘기는 곳이 없으면 아무것도 보존되지 않는다."""
    items = 가짜KRX.fetch_index_snapshot("20260821", "KOSPI")
    assert items[0]["close"] == 1096.25          # 수집 자체는 정상이다

    보존 = raw_store.load("krx", "idx/kospi_dd_trd/20260821", db_path=db)
    assert 보존 is not None, "수집했는데 원문이 안 남았다"
    assert b"1,096.25" in 보존["body"]           # 파싱 전 문자열 그대로 남아 있다


def test_원문_보존을_끄면_남기지_않는다(가짜KRX, db, monkeypatch):
    from common import settings

    monkeypatch.setattr(settings, "keep_raw_enabled", lambda: False)

    가짜KRX.fetch_index_snapshot("20260821", "KOSPI")

    assert raw_store.load("krx", "idx/kospi_dd_trd/20260821", db_path=db) is None


def test_원문_보존이_실패해도_수집은_계속된다(가짜KRX, db, monkeypatch):
    """보존은 나중을 위한 보험이다. 디스크가 찼다고 16년 백필이 멈추면 안 된다."""
    def 터진다(*a, **k):
        raise OSError("디스크가 찼다")

    monkeypatch.setattr(raw_store, "save", 터진다)

    items = 가짜KRX.fetch_index_snapshot("20260821", "KOSPI")

    assert items[0]["close"] == 1096.25          # 수집은 멀쩡히 끝났다
