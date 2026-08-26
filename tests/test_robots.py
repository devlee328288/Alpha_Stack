"""`robots.txt` 가드 테스트.

**무엇을 지키려는 테스트인가.** 판정 자체가 아니라 **못 받았을 때의 방향**이다.

    4xx (robots.txt 가 없다)      →  전면 허용
    5xx (서버가 답을 못 준다)      →  전면 차단   ← RFC 9309 가 MUST 로 정한 것
    네트워크 오류                  →  전면 차단

**이 둘을 거꾸로 짜는 일이 흔하다.** 뭉뚱그려 "못 받았으면 허용" 으로 두면 상대 서버가
잠깐 아픈 사이에 우리가 그 위를 긁는다. 반대로 두면 robots.txt 없는 사이트를 영영 못 읽는다.

    수용 기준
    - 5xx·네트워크 오류에서 **차단**된다 ← 이게 핵심이다
    - 4xx 에서 허용된다
    - `User-agent` **그룹**을 갈라 읽는다 (2026-08-26 에 이걸 잘못 읽어 계획이 뒤집혔다)
    - euc-kr 로 온 규칙이 깨지지 않는다
    - 차단은 예외로 멈춘다 (빈 결과로 조용히 넘어가지 않는다)

네트워크를 타지 않는다 — `fetch()` 를 가로채 상태 코드를 흉내 낸다.
"""

from __future__ import annotations

import pytest

from common import robots


@pytest.fixture()
def db(tmp_path):
    return tmp_path / "robots.db"


def _응답(monkeypatch, status: int, body=None, encoding: str = "utf-8"):
    """`robots.txt` 응답 하나를 흉내 낸다."""
    monkeypatch.setattr(robots, "fetch", lambda origin, **k: (status, body, encoding))


# ── 못 받았을 때의 방향 ─────────────────────────────────────────────────────

@pytest.mark.parametrize("status", [500, 502, 503, 599])
def test_5xx_는_전면_차단이다(monkeypatch, db, status):
    """RFC 9309 §2.3.1.4 — 서버가 답을 못 주면 MUST assume complete disallow."""
    _응답(monkeypatch, status)

    assert robots.can_fetch("https://example.com/any", db_path=db) is False


def test_네트워크_오류도_전면_차단이다(monkeypatch, db):
    """닿지도 못한 것은 5xx 와 같다. 기다린다고 달라지지 않는다."""
    _응답(monkeypatch, robots.UNREACHABLE)

    assert robots.can_fetch("https://example.com/any", db_path=db) is False


@pytest.mark.parametrize("status", [400, 403, 404, 410, 499])
def test_4xx_는_전면_허용이다(monkeypatch, db, status):
    """robots.txt 가 없다는 뜻이다 — 규칙이 없으니 막을 근거도 없다."""
    _응답(monkeypatch, status)

    assert robots.can_fetch("https://example.com/any", db_path=db) is True


def test_두_실패의_방향이_정반대다(monkeypatch, db):
    """한 테스트에 나란히 둔다. 뭉뚱그리면 여기서 잡힌다."""
    _응답(monkeypatch, 404)
    없음 = robots.can_fetch("https://a.example.com/x", db_path=db)
    _응답(monkeypatch, 503)
    아픔 = robots.can_fetch("https://b.example.com/x", db_path=db)

    assert (없음, 아픔) == (True, False)


# ── 그룹을 갈라 읽는가 ──────────────────────────────────────────────────────

#: 2026-08-26 에 실제로 잘못 읽었던 모양. `Allow` 가 **자체 크롤러 그룹** 소속이고,
#: 제3자 봇이 매칭되는 `*` 그룹은 `Disallow: /` 한 줄이다.
NAVER_모양 = """User-agent: yeti
Allow: /item/board.naver

User-agent: *
Disallow: /
"""


def test_다른_그룹의_Allow_를_내_것으로_읽지_않는다(monkeypatch, db):
    """이 한 줄을 잘못 읽어 수집 계획 하나가 통째로 뒤집혔다."""
    _응답(monkeypatch, 200, NAVER_모양)
    url = "https://finance.naver.com/item/board.naver?code=005930"

    assert robots.can_fetch(url, db_path=db) is False
    # 그 규칙의 주인은 통과한다 — 그룹을 갈라 읽고 있다는 증거다
    assert robots.can_fetch(url, user_agent="yeti", db_path=db) is True


def test_와일드카드_금지를_허용으로_읽지_않는다(monkeypatch, db):
    """`urllib.robotparser` 가 여기서 True 를 준다 — 그래서 그걸 쓰지 않는다."""
    _응답(monkeypatch, 200, "User-agent: *\nDisallow: /item/*\n")

    assert robots.can_fetch("https://example.com/item/board?code=1", db_path=db) is False


def test_허용된_경로는_통과한다(monkeypatch, db):
    _응답(monkeypatch, 200, "User-agent: *\nDisallow: /private/\n")

    assert robots.can_fetch("https://example.com/public/page", db_path=db) is True
    assert robots.can_fetch("https://example.com/private/page", db_path=db) is False


def test_본문이_비면_허용이다(monkeypatch, db):
    """200 인데 규칙이 없다 — 없는 것과 같다."""
    _응답(monkeypatch, 200, "")

    assert robots.can_fetch("https://example.com/any", db_path=db) is True


# ── 인코딩 ──────────────────────────────────────────────────────────────────

def test_euc_kr_규칙이_깨지지_않는다():
    """잘못 디코딩하면 **예외 없이** 규칙만 깨진다 — 그러면 없는 것처럼 동작한다."""
    원문 = "# 종목토론방 수집 금지\nUser-agent: *\nDisallow: /item/\n".encode("euc-kr")

    글자, 쓴인코딩 = robots._decode(원문, None)

    assert "종목토론방" in 글자
    assert 쓴인코딩 == "euc-kr"
    assert "Disallow: /item/" in 글자


def test_헤더가_말한_인코딩을_먼저_믿는다():
    원문 = "User-agent: *\nDisallow: /가나/\n".encode("euc-kr")

    글자, 쓴인코딩 = robots._decode(원문, "euc-kr")

    assert 쓴인코딩 == "euc-kr"
    assert "/가나/" in 글자


def test_UTF8_은_그대로_읽는다():
    글자, 쓴인코딩 = robots._decode("User-agent: *\nDisallow: /비공개/\n".encode(), None)

    assert 쓴인코딩 == "utf-8"
    assert "/비공개/" in 글자


# ── 차단은 조용히 넘어가지 않는다 ────────────────────────────────────────────

def test_차단이면_예외로_멈춘다(monkeypatch, db):
    """빈 결과로 넘기면 '수집했는데 없었다' 와 구별되지 않는다."""
    _응답(monkeypatch, 200, "User-agent: *\nDisallow: /\n")

    with pytest.raises(robots.RobotsBlocked) as 오류:
        robots.require("https://example.com/any", db_path=db)

    메시지 = str(오류.value)
    assert "확인한 이름" in 메시지                # 어느 이름으로 판정했는지 알려 준다
    assert "할 일" in 메시지


def test_허용이면_그냥_지나간다(monkeypatch, db):
    _응답(monkeypatch, 200, "User-agent: *\nAllow: /\n")

    robots.require("https://example.com/any", db_path=db)     # 예외가 없다


# ── 캐시 ────────────────────────────────────────────────────────────────────

def test_한_번_받으면_다시_받지_않는다(monkeypatch, db):
    """매 요청마다 받으면 그 확인 자체가 상대 서버를 두드리는 일이 된다."""
    횟수 = {"n": 0}

    def 세면서받기(origin, **k):
        횟수["n"] += 1
        return 200, "User-agent: *\nAllow: /\n", "utf-8"

    monkeypatch.setattr(robots, "fetch", 세면서받기)

    for _ in range(5):
        robots.can_fetch("https://example.com/a", db_path=db)

    assert 횟수["n"] == 1


def test_호스트가_다르면_따로_받는다(monkeypatch, db):
    """robots.txt 는 스킴·호스트·포트마다 따로다. 묶으면 규칙이 새어 든다."""
    받은곳 = []
    monkeypatch.setattr(robots, "fetch",
                        lambda origin, **k: (받은곳.append(origin), (200, "", "utf-8"))[1])

    robots.can_fetch("https://a.example.com/x", db_path=db)
    robots.can_fetch("https://b.example.com/x", db_path=db)
    robots.can_fetch("http://a.example.com/x", db_path=db)      # 스킴이 다르다

    assert 받은곳 == ["https://a.example.com", "https://b.example.com",
                      "http://a.example.com"]


def test_캐시를_비우면_다시_받는다(monkeypatch, db):
    횟수 = {"n": 0}
    monkeypatch.setattr(robots, "fetch",
                        lambda origin, **k: (횟수.update(n=횟수["n"] + 1),
                                             (200, "", "utf-8"))[1])

    robots.can_fetch("https://example.com/a", db_path=db)
    robots.clear_cache(db_path=db)
    robots.can_fetch("https://example.com/a", db_path=db)

    assert 횟수["n"] == 2


def test_절대_URL_이_아니면_거부한다(db):
    with pytest.raises(ValueError):
        robots.can_fetch("/item/board.naver", db_path=db)
