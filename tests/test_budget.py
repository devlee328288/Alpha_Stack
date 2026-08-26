"""호출 예산 테스트 — 한도를 넘기지 않고, 넘었을 때 조용히 멈추는가.

**무엇을 지키려는 테스트인가.** 외부 API 한도를 넘기면 그날 남은 수집이 통째로 막힌다.
16년치 백필처럼 며칠 걸리는 작업에서 이게 터지면 어디서 끊겼는지도 모른 채 멈춘다.

여기서 잠그는 것은 셋이다.
  ① 세는 게 **여러 프로세스·스레드에서도 정확한가** — 배치와 화면 버튼이 따로 돈다
  ② 한도에 닿았을 때 **예외가 아니라 정상 종료인가** — 예외면 진짜 실패와 구별이 안 된다
  ③ 서버가 한도 초과를 알려 왔을 때 **재시도 루프가 저절로 멈추는가**

전부 tmp_path 의 실제 파일 DB 로 돈다 — 원자성은 메모리 DB 로는 증명이 약하다.
"""

from __future__ import annotations

import json
from datetime import date

import pytest

from common import budget


@pytest.fixture
def db(tmp_path):
    """테스트마다 빈 DB 하나."""
    return tmp_path / "budget.db"


@pytest.fixture
def 날짜고정(monkeypatch):
    """`today_kst()` 를 마음대로 움직이는 손잡이."""
    상태 = {"오늘": date(2026, 8, 26)}
    monkeypatch.setattr(budget, "today_kst", lambda: 상태["오늘"])
    return 상태


# ── 기본 ───────────────────────────────────────────────────────────────────

def test_한도_안에서는_쓰고_넘으면_멈춘다(db, 날짜고정):
    for i in range(3):
        assert budget.try_spend("krx", limit=3, db_path=db) is True, f"{i + 1}번째가 막혔다"

    # 네 번째는 한도를 넘는다 — **예외가 아니라 False** 다
    assert budget.try_spend("krx", limit=3, db_path=db) is False


def test_한도_도달은_예외가_아니다(db, 날짜고정):
    """한도 소진은 고장이 아니라 정상적인 하루의 끝이다.

    예외로 만들면 배치가 '실패' 로 기록되고, 그러면 진짜 실패와 구별할 수 없게 된다.
    """
    budget.try_spend("krx", limit=1, db_path=db)
    try:
        결과 = budget.try_spend("krx", limit=1, db_path=db)
    except Exception as exc:                      # noqa: BLE001
        pytest.fail(f"한도 도달로 예외가 났다: {exc!r}")
    assert 결과 is False


def test_한_번에_여러_개도_한도를_넘지_않는다(db, 날짜고정):
    """`n` 을 크게 주면 남은 여유보다 클 수 있다. 부분 소비는 하지 않는다."""
    assert budget.try_spend("krx", 8, limit=10, db_path=db) is True
    # 남은 건 2 인데 5 를 요구하면 거절한다 (2만 쓰고 True 를 주지 않는다)
    assert budget.try_spend("krx", 5, limit=10, db_path=db) is False
    assert budget.usage("krx", db_path=db)["krx"]["used"] == 8, "거절했는데 숫자가 늘었다"
    # 딱 맞는 건 통과한다
    assert budget.try_spend("krx", 2, limit=10, db_path=db) is True


def test_날짜가_바뀌면_저절로_0_부터_센다(db, 날짜고정):
    """`(출처, 날짜)` 가 키라서 자정 리셋 코드가 아예 필요 없다 — 잊어버릴 수가 없다."""
    assert budget.try_spend("krx", limit=1, db_path=db) is True
    assert budget.try_spend("krx", limit=1, db_path=db) is False

    날짜고정["오늘"] = date(2026, 8, 27)

    assert budget.try_spend("krx", limit=1, db_path=db) is True, "다음 날인데 안 풀렸다"
    # 어제 기록은 지워지지 않는다 — 나중에 사용 이력을 볼 수 있어야 한다
    어제 = budget.usage("krx", kst_date="2026-08-26", db_path=db)
    assert 어제["krx"]["used"] == 1


# ── 서버가 거절했을 때 ──────────────────────────────────────────────────────

def test_서버가_한도초과를_알리면_재시도_루프가_멈춘다(db, 날짜고정):
    """우리 계산과 서버 계산이 어긋날 수 있다. 서버가 거절했으면 그쪽이 맞다.

    이 뒤로 `try_spend` 가 전부 False 를 주므로, 호출자가 재시도 로직을 따로 안 고쳐도
    루프가 저절로 접힌다.
    """
    budget.try_spend("krx", limit=1000, db_path=db)

    budget.mark_exhausted("krx", note="429 를 받았다", limit=1000, db_path=db)

    assert budget.try_spend("krx", limit=1000, db_path=db) is False
    상태 = budget.usage("krx", db_path=db)["krx"]
    assert 상태["used"] == 상태["limit"], "소진 처리했는데 여유가 남아 있다"


# ── 경고 ───────────────────────────────────────────────────────────────────

def test_80퍼센트_경고는_한_번만_남는다(db, 날짜고정, caplog):
    """매 호출마다 찍으면 로그에 묻혀서 아무도 못 본다."""
    import logging

    with caplog.at_level(logging.WARNING, logger="common.budget"):
        for _ in range(10):
            budget.try_spend("krx", limit=10, db_path=db)

    경고 = [r for r in caplog.records if "넘었다" in r.getMessage()]
    assert len(경고) == 1, f"경고가 {len(경고)}번 났다 — 한 번이어야 한다"
    # 8/10 = 80% 에서 처음 걸린다
    assert "8/10" in 경고[0].getMessage()


def test_경고_시각이_기록된다(db, 날짜고정):
    for _ in range(8):
        budget.try_spend("krx", limit=10, db_path=db)
    assert budget.usage("krx", db_path=db)["krx"]["warned_at"] is not None


# ── 오타 방어 ───────────────────────────────────────────────────────────────

def test_모르는_출처는_조용히_넘어가지_않는다(db, 날짜고정):
    """`"navar_search"` 같은 오타가 통과하면 한도 없이 마구 부르게 된다."""
    with pytest.raises(budget.UnknownSource) as 오류:
        budget.try_spend("navar_search", db_path=db)
    # 무엇을 해야 하는지까지 알려 주는가
    assert "할 일" in str(오류.value)
    assert "naver_search" in str(오류.value), "아는 출처 목록을 안 보여 줬다"


def test_한도를_직접_주면_모르는_출처도_쓸_수_있다(db, 날짜고정):
    """새 출처를 붙이는 중에 LIMITS 를 아직 안 고쳤을 때 막히지 않게."""
    assert budget.try_spend("실험용출처", limit=2, db_path=db) is True


# ── 여러 스레드가 동시에 ────────────────────────────────────────────────────

def test_동시에_불러도_한도를_넘지_않는다(db, 날짜고정):
    """수집은 스레드 6개로 돈다. 읽고 나서 쓰는 사이에 끼어들면 둘 다 통과해 버린다.

    `BEGIN IMMEDIATE` 로 잠금을 먼저 잡는 이유가 이것이다.
    """
    from concurrent.futures import ThreadPoolExecutor

    한도 = 50
    시도 = 200

    with ThreadPoolExecutor(max_workers=8) as pool:
        결과 = list(pool.map(
            lambda _: budget.try_spend("krx", limit=한도, db_path=db), range(시도)))

    통과 = sum(결과)
    assert 통과 == 한도, f"{시도}번 시도에 {통과}번 통과 — 정확히 {한도}이어야 한다"
    assert budget.usage("krx", db_path=db)["krx"]["used"] == 한도


# ── 읽기 · 리포트 ───────────────────────────────────────────────────────────

def test_한_번도_안_부른_출처도_0_으로_보여_준다(db, 날짜고정):
    """화면에서 '아직 안 쓴 것' 과 '표에서 빠진 것' 이 구별돼야 한다."""
    budget.try_spend("krx", limit=100, db_path=db)

    전체 = budget.usage(db_path=db)

    assert set(전체) >= set(budget.LIMITS), "아는 출처가 목록에서 빠졌다"
    assert 전체["dart"]["used"] == 0
    assert 전체["krx"]["used"] == 1


def test_리포트가_읽을_수_있는_JSON_으로_나온다(db, 날짜고정, tmp_path):
    budget.try_spend("krx", 40, limit=100, db_path=db)

    경로 = budget.write_report(tmp_path / "quota_usage.json", db_path=db)

    실린것 = json.loads(경로.read_text(encoding="utf-8"))
    assert 실린것["kst_date"] == "2026-08-26"
    assert 실린것["sources"]["krx"] == {
        "used": 40, "limit": 100, "ratio": 0.4, "warned_at": None,
    }
    # git diff 로 어제와 비교되도록 정렬·개행이 고정돼야 한다
    본문 = 경로.read_text(encoding="utf-8")
    assert 본문.endswith("\n")
    assert 본문 == json.dumps(실린것, sort_keys=True, indent=2, ensure_ascii=False) + "\n"


def test_한도가_0_이어도_나눗셈에서_죽지_않는다(db, 날짜고정):
    """설정 실수를 ZeroDivisionError 로 키우지 않는다 — 리포트는 계속 나와야 한다."""
    budget.try_spend("실험용출처", limit=0, db_path=db)
    assert budget.usage("실험용출처", db_path=db)["실험용출처"]["ratio"] is None
