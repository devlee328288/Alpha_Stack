"""`text_signal` — 공시 제목 감성이 **시점을 새지 않게** 잠근다.

## 왜 이 시험이 있나

`text_signal` 은 (text_sha, model_id) 가 기본키인 **제목 단위 캐시**고, 팀원에게 나가는
파일은 **접수번호 단위**다. 둘 사이에서 `known_at`(접수일 다음 거래일)이 만들어지는데,
이 규칙이 틀리면 예외가 나지 않고 **성능만 올라간다** — 15:00 접수 공시를 그날 신호로
쓰면 장 마감 30분 전에 알던 것이 된다. 그래서 규칙을 여기서 막는다.

세 가지를 잰다.

1. **해시** — `text_sha` 가 제목의 SHA-256 앞 16자인가 (조인이 틀린 짝을 만들지 않게)
2. **known_at** — 접수일이 거래일이어도 그날이 아니라 *다음* 거래일인가 · 달력 밖은
   지어내지 않는가 · 접수는 개발구간인데 known_at 이 봉인이면 빠지는가 (실측 522행)
3. **표 제약** — 확률 셋의 합이 1 이 아니면 담기지 않는가 (softmax 누락을 DB 가 막는다)

망을 타지 않고 모델도 부르지 않는다. DB 는 `conftest.py` 가 임시 경로로 갈아 끼운다.
"""

from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ingest.store.migrations import migrate_path  # noqa: E402
from scripts import export_text_signal as ets  # noqa: E402
from scripts import score_text_signal as sts  # noqa: E402

# 2024-08 마지막 주 · 20240831(토)·20240901(일)은 휴장 · 홀드아웃 경계는 20240901.
달력 = ("20240826", "20240827", "20240828", "20240829", "20240830",
        "20240902", "20240903")


@pytest.fixture
def db(tmp_path):
    """이 시험만 쓰는 빈 DB. 마이그레이션을 최신(v12 포함)까지 올려 둔다."""
    경로 = tmp_path / "t.db"
    migrate_path(경로)
    conn = sqlite3.connect(경로)
    conn.row_factory = sqlite3.Row
    yield conn
    conn.close()


def 공시(conn, rcept_no: str, rcept_dt: str, report_nm: str, *,
         stock_code: str | None = "005930") -> None:
    conn.execute(
        "INSERT INTO dart_disclosure (rcept_no, corp_code, corp_name, stock_code, "
        "corp_cls, report_nm, flr_nm, rcept_dt, rm, collected_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (rcept_no, "00126380", "삼성전자", stock_code, "Y", report_nm, "삼성전자",
         rcept_dt, "", "2026-09-04T00:00:00+09:00"))


def 신호(conn, report_nm: str, p_neg: float, p_neu: float, p_pos: float, *,
         model_id: str = sts.MODEL_ID) -> None:
    conn.execute(
        "INSERT INTO text_signal (text_sha, report_nm, model_id, revision, "
        "p_neg, p_neu, p_pos, scored_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        (sts.text_sha(report_nm), report_nm, model_id, "f8586286", p_neg, p_neu, p_pos,
         "2026-09-04T14:05:00+09:00"))


# ==================================================
# 1. 해시
# ==================================================

def test_text_sha_는_제목의_SHA256_앞_16자다():
    import hashlib
    제목 = "주요사항보고서(유상증자결정)"
    참 = hashlib.sha256(제목.encode("utf-8")).hexdigest()[:16]
    assert sts.text_sha(제목) == 참
    assert len(sts.text_sha(제목)) == 16


def test_공백_하나만_달라도_다른_제목이다():
    # 정규화하지 않는다 — 원문 그대로가 키다. 정규화는 다른 층의 일이고, 여기서 하면
    # "같다고 본 두 제목" 이 어느 쪽 확률을 받았는지 알 수 없어진다.
    assert sts.text_sha("사업보고서 (2019.12)") != sts.text_sha("사업보고서(2019.12)")


# ==================================================
# 2. known_at — 접수일 다음 거래일
# ==================================================

def test_접수일이_거래일이어도_known_at_은_그날이_아니라_다음_거래일이다():
    # 🔴 핵심. rcept_dt 에는 시각이 없어 15:00 접수를 그날 신호로 쓰면 장중에 알던 것이 된다.
    지도 = ets._next_session_map(["20240829"], 달력)
    assert 지도 == {"20240829": "20240830"}


def test_휴일_접수는_다음_거래일이다():
    지도 = ets._next_session_map(["20240831"], 달력)         # 토요일
    assert 지도 == {"20240831": "20240902"}


def test_금요일_접수는_주말을_건너_월요일이다():
    지도 = ets._next_session_map(["20240830"], 달력)
    assert 지도 == {"20240830": "20240902"}


def test_달력_밖_접수일은_known_at_을_지어내지_않는다():
    # 달력 마지막 날 접수 → 다음 거래일을 모른다 → 키가 없다. 지어낸 시점은 미래참조다.
    지도 = ets._next_session_map(["20240903", "20240910"], 달력)
    assert 지도 == {}


def test_known_at_규칙_이름은_접수일_다음_거래일이다():
    assert ets.KNOWN_RULE == "rceptDt+1session"


# ==================================================
# 3. 표 제약 — 확률은 DB 가 먼저 막는다
# ==================================================

def test_확률_셋의_합이_1_이_아니면_담기지_않는다(db):
    # softmax 를 빠뜨리면 logit 이 그대로 들어온다. 합이 1 이 아니라 여기서 걸린다.
    with pytest.raises(sqlite3.IntegrityError):
        신호(db, "사업보고서", 0.5, 0.5, 0.5)


def test_확률이_범위_밖이면_담기지_않는다(db):
    with pytest.raises(sqlite3.IntegrityError):
        신호(db, "사업보고서", -0.1, 0.6, 0.5)


def test_같은_제목을_다른_모델로_매기면_덮지_않고_나란히_쌓인다(db):
    신호(db, "사업보고서", 0.1, 0.8, 0.1)
    신호(db, "사업보고서", 0.2, 0.6, 0.2, model_id="other/model")
    n = db.execute("SELECT COUNT(*) FROM text_signal WHERE report_nm='사업보고서'").fetchone()[0]
    assert n == 2


# ==================================================
# 4. 안 매긴 제목 — 흔한 것부터, 매긴 것은 빠진다
# ==================================================

def test_안매긴_제목은_흔한_것부터_이고_이미_매긴_것은_빠진다(db):
    for i in range(3):
        공시(db, f"A{i}", "20240826", "임원ㆍ주요주주특정증권등소유상황보고서")
    for i in range(2):
        공시(db, f"B{i}", "20240826", "주식등의대량보유상황보고서(일반)")
    공시(db, "C0", "20240826", "증권발행실적보고서")
    신호(db, "주식등의대량보유상황보고서(일반)", 0.1, 0.8, 0.1)
    assert sts.안매긴_제목(db) == ["임원ㆍ주요주주특정증권등소유상황보고서", "증권발행실적보고서"]


def test_다른_모델로만_매긴_제목은_이_모델에게는_안_매긴_것이다(db):
    공시(db, "A0", "20240826", "사업보고서")
    신호(db, "사업보고서", 0.1, 0.8, 0.1, model_id="other/model")
    assert sts.안매긴_제목(db) == ["사업보고서"]


def test_limit_은_흔한_것_위주로_자른다(db):
    for i in range(3):
        공시(db, f"A{i}", "20240826", "흔한제목")
    공시(db, "B0", "20240826", "드문제목")
    assert sts.안매긴_제목(db, limit=1) == ["흔한제목"]


# ==================================================
# 5. 반출 — 접수번호마다 한 줄 · 자르는 기준은 known_at
# ==================================================

@pytest.fixture
def 달력고정(monkeypatch):
    """반출이 보는 달력을 이 시험의 것으로 바꾼다 (진짜 DB 의 달력 캐시를 타지 않게)."""
    monkeypatch.setattr(ets, "load_session_days", lambda: frozenset(달력))


def test_반출은_접수번호마다_한_줄이고_known_at_은_접수일_다음_거래일이다(db, 달력고정):
    신호(db, "사업보고서", 0.1, 0.8, 0.1)
    공시(db, "R1", "20240828", "사업보고서")
    공시(db, "R2", "20240829", "사업보고서")
    out = ets.build(db)
    assert list(out["rcept_no"]) == ["R1", "R2"]
    assert list(out["known_at"]) == ["20240829", "20240830"]
    assert (out["known_rule"] == "rceptDt+1session").all()


def test_접수는_개발구간인데_known_at_이_봉인이면_뺀다(db, 달력고정):
    # 🔴 실측 522행. 20240830(금) 접수 → known_at 20240902 ≥ HOLDOUT_START.
    #    접수일로만 자르면 "개발구간 자료" 얼굴로 들어오는데 실제로 쓸 수 있는 날은 봉인 구간이다.
    신호(db, "사업보고서", 0.1, 0.8, 0.1)
    공시(db, "R1", "20240829", "사업보고서")
    공시(db, "R2", "20240830", "사업보고서")
    out = ets.build(db)
    assert list(out["rcept_no"]) == ["R1"]
    assert (out["known_at"] < ets.HOLDOUT_START).all()


def test_달력_밖_접수는_행을_만들지_않는다(db, monkeypatch):
    # 달력이 20240829 에서 끝나면 20240830 접수의 다음 거래일을 모른다 → 지어내지 않고 뺀다.
    monkeypatch.setattr(ets, "load_session_days", lambda: frozenset(달력[:4]))
    신호(db, "사업보고서", 0.1, 0.8, 0.1)
    공시(db, "R1", "20240828", "사업보고서")
    공시(db, "R2", "20240830", "사업보고서")
    out = ets.build(db)
    assert list(out["rcept_no"]) == ["R1"]


def test_종목코드_없는_공시는_반출하지_않는다(db, 달력고정):
    # 비상장 법인 공시 — 붙일 시세가 없다.
    신호(db, "사업보고서", 0.1, 0.8, 0.1)
    공시(db, "R1", "20240828", "사업보고서", stock_code="")
    공시(db, "R2", "20240828", "사업보고서", stock_code=None)
    공시(db, "R3", "20240828", "사업보고서")
    out = ets.build(db)
    assert list(out["rcept_no"]) == ["R3"]


def test_안_매긴_제목의_공시는_조용히_빠진다_그래서_검증기가_커버리지를_먼저_본다(db, 달력고정):
    # JOIN 이라 매기지 않은 제목은 반출에 없다. 이게 조용한 구멍이 되지 않게
    # `verify_text_signal.py` 가 "안 매긴 고유 제목 0" 을 게이트로 먼저 확인한다.
    신호(db, "사업보고서", 0.1, 0.8, 0.1)
    공시(db, "R1", "20240828", "사업보고서")
    공시(db, "R2", "20240828", "안매긴제목")
    out = ets.build(db)
    assert list(out["rcept_no"]) == ["R1"]


def test_반출_칸은_열셋이고_순서가_고정이다(db, 달력고정):
    신호(db, "사업보고서", 0.1, 0.8, 0.1)
    공시(db, "R1", "20240828", "사업보고서")
    out = ets.build(db)
    assert list(out.columns) == [
        "code", "rcept_no", "rcept_dt", "known_at", "known_rule", "source",
        "report_nm", "model", "model_rev", "p_pos", "p_neg", "p_neu", "text_sha256"]
    assert out.iloc[0]["source"] == "dart:list.json"
    assert out.iloc[0]["model"] == sts.MODEL_ID


# ==================================================
# 6. 경계값은 정본과 같아야 한다
# ==================================================

def test_홀드아웃_경계는_evaluation_horizon_과_같다():
    # 반출·검증기는 파트 경계 때문에 값을 **베껴 두고 import 하지 않는다.**
    # 베낀 값이 정본에서 멀어지면 여기서 걸린다.
    from evaluation.horizon import HOLDOUT_START
    from scripts import verify_text_signal as vts
    assert ets.HOLDOUT_START == HOLDOUT_START
    assert vts.HOLDOUT_START == HOLDOUT_START
    assert vts.KNOWN_RULE == ets.KNOWN_RULE
