"""인증 차단기·재시도 동작 테스트

**왜 이 테스트가 필요한가.** 여기 규칙은 두 요구가 정면으로 부딪히는 자리다.

  · 웹 서버: 키가 틀렸으면 사용자 요청마다 KRX 를 두드리지 말아야 한다 → **빨리 차단**
  · 배치 백필: 4,343콜 도중 깜빡임 하나로 멈추면 안 된다 → **쉽게 차단하지 말아야**

한쪽만 보고 고치면 다른 쪽이 조용히 깨진다. 실제로 2026-08-26 에 그렇게 깨졌다 —
간헐적 401 하나가 남은 3,000일을 45초 만에 전멸시켰다(실측).
그래서 두 요구를 **각각 테스트로 박아 둔다.**

네트워크를 타지 않는다. `urlopen` 을 가짜로 갈아 끼워 응답을 우리가 정한다.
"""

import io
import json
from urllib.error import HTTPError

import pytest

from ingest.clients import krx_data as api


@pytest.fixture(autouse=True)
def 차단기초기화():
    """테스트끼리 차단기 상태가 새지 않게 한다. 이건 모듈 전역이다."""
    api.reset_auth_block()
    yield
    api.reset_auth_block()


@pytest.fixture(autouse=True)
def 백오프제거(monkeypatch):
    """재시도 대기를 없앤다. 안 그러면 테스트가 초 단위로 느려진다."""
    monkeypatch.setattr(api.time, "sleep", lambda _s: None)


def _키있음(monkeypatch):
    monkeypatch.setattr(api, "load_krx_key", lambda: ("TESTKEY", "테스트"))


def _응답(rows):
    """urlopen 이 돌려줄 가짜 컨텍스트 매니저."""
    body = json.dumps({"OutBlock_1": rows}).encode("utf-8")

    class _Res(io.BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    return _Res(body)


def _401():
    return HTTPError("http://x", 401, "Unauthorized", {}, io.BytesIO(b"Unauthorized Key"))


def _가짜urlopen(monkeypatch, 시나리오):
    """시나리오는 호출마다 돌려줄 것의 목록. 예외 인스턴스면 던진다."""
    남은 = list(시나리오)
    호출 = {"n": 0}

    def fake(_request, timeout=None):
        호출["n"] += 1
        item = 남은.pop(0) if 남은 else 시나리오[-1]
        if isinstance(item, Exception):
            raise item
        return _응답(item)

    monkeypatch.setattr(api, "urlopen", fake)
    return 호출


# ==============================================================
# 1. 배치 요구 — 깜빡임 하나로 멈추지 않는다
# ==============================================================
def test_간헐적_401_은_재시도로_흡수한다(monkeypatch):
    """실측된 실제 상황: 401 한 번 뒤 곧바로 성공한다."""
    _키있음(monkeypatch)
    호출 = _가짜urlopen(monkeypatch, [_401(), [{"IDX_NM": "코스피 200"}]])

    rows = api.fetch_index_snapshot("20210614", "KOSPI")

    assert len(rows) == 1
    assert 호출["n"] == 2, "재시도가 일어나지 않았다"
    assert api.get_status()["auth_blocked"] is False, "깜빡임으로 차단기가 걸렸다"


def test_두_번_실패해도_세_번째에_성공하면_통과한다(monkeypatch):
    _키있음(monkeypatch)
    호출 = _가짜urlopen(monkeypatch, [_401(), _401(), [{"IDX_NM": "코스피 200"}]])

    assert len(api.fetch_index_snapshot("20210614", "KOSPI")) == 1
    assert 호출["n"] == 3
    assert api.get_status()["auth_blocked"] is False


def test_성공하면_연속_실패_카운터가_0_으로_돌아간다(monkeypatch):
    """안 되돌리면 백필 내내 띄엄띄엄 난 실패가 누적돼 결국 차단된다."""
    _키있음(monkeypatch)

    # 한 번 완전히 실패시켜 카운터를 1 로 만든다
    _가짜urlopen(monkeypatch, [_401()])
    with pytest.raises(api.KrxError):
        api.fetch_index_snapshot("20210614", "KOSPI")
    assert api._auth_failures["consecutive"] == 1

    # 그다음 성공하면 0 이어야 한다
    _가짜urlopen(monkeypatch, [[{"IDX_NM": "코스피 200"}]])
    api.fetch_index_snapshot("20210615", "KOSPI")
    assert api._auth_failures["consecutive"] == 0


# ==============================================================
# 2. 웹 서버 요구 — 진짜로 틀린 키는 결국 차단한다
# ==============================================================
def test_계속_실패하면_결국_차단된다(monkeypatch):
    """차단기의 원래 목적. 이게 없으면 틀린 키로 500번을 두드린다."""
    _키있음(monkeypatch)
    _가짜urlopen(monkeypatch, [_401()])

    for _ in range(api.AUTH_FAIL_THRESHOLD):
        with pytest.raises(api.KrxError):
            api.fetch_index_snapshot("20210614", "KOSPI")

    assert api.get_status()["auth_blocked"] is True


def test_차단된_뒤에는_네트워크를_타지_않는다(monkeypatch):
    """차단의 값어치가 여기 있다 — 상대 서버를 그만 두드린다."""
    _키있음(monkeypatch)
    호출 = _가짜urlopen(monkeypatch, [_401()])

    for _ in range(api.AUTH_FAIL_THRESHOLD):
        with pytest.raises(api.KrxError):
            api.fetch_index_snapshot("20210614", "KOSPI")
    차단시점 = 호출["n"]

    with pytest.raises(api.KrxError):
        api.fetch_index_snapshot("20210615", "KOSPI")

    assert 호출["n"] == 차단시점, "차단된 뒤에도 KRX 를 불렀다"


def test_키가_아예_없으면_곧바로_차단한다(monkeypatch):
    """재시도해도 소용없다 — 임계를 기다릴 이유가 없다."""
    monkeypatch.setattr(api, "load_krx_key", lambda: ("", "none"))
    호출 = _가짜urlopen(monkeypatch, [[{"IDX_NM": "코스피 200"}]])

    with pytest.raises(api.KrxError) as caught:
        api.fetch_index_snapshot("20210614", "KOSPI")

    assert caught.value.unauthorized is True
    assert 호출["n"] == 0, "키가 없는데 네트워크를 탔다"
    assert api.get_status()["auth_blocked"] is True


# ==============================================================
# 3. 재시도 대상을 넓히지 않는다
# ==============================================================
def test_인증_문제가_아닌_실패는_재시도하지_않는다(monkeypatch):
    """HTTP 500 까지 재시도하면 실패가 몇 겹으로 늘어져 진행이 안 보인다."""
    _키있음(monkeypatch)
    오류 = HTTPError("http://x", 500, "Server Error", {}, io.BytesIO(b""))
    호출 = _가짜urlopen(monkeypatch, [오류])

    with pytest.raises(api.KrxError) as caught:
        api.fetch_index_snapshot("20210614", "KOSPI")

    assert caught.value.unauthorized is False
    assert 호출["n"] == 1, "인증 문제가 아닌데 재시도했다"


def test_종목_경로도_같은_재시도를_받는다(monkeypatch):
    """종목과 지수가 같은 `_request_rows` 를 쓴다 — 한쪽만 고치는 사고를 막는다."""
    _키있음(monkeypatch)
    호출 = _가짜urlopen(monkeypatch, [_401(), [{"ISU_CD": "005930"}]])

    rows = api.fetch_snapshot("20210614", "KOSPI")

    assert len(rows) == 1
    assert 호출["n"] == 2
