"""공시 텍스트 정문 — 제목 점수를 공시의 날짜로, 접수일 다음 거래일에 붙이는가.

**무엇을 지키려는 시험인가.** `text_signal` 에는 날짜가 없다. 시점은 그 제목을 단 공시의
접수일에서 오고, 규칙은 재무와 같은 *"접수일 다음 거래일"* 이다.

🔴 누수 검사에는 음성 대조군을 둔다 — 접수일 **당일**로 붙인 틀린 구현이 실제로 붉어지는지.
   검사는 정문을 부르지 않고 이 파일의 달력을 이분탐색해 따로 센다.
🔴 `rm`('정'·'철')은 나중에 붙는 표시라 거르지 않는다 — 걸러지지 않았는지 본다.
🔴 수집 구간 밖의 행은 "공시 0건" 이 아니라 결측이다.

망을 타지 않는다. DB 는 임시 경로를 만들어 `db_path` 로 넘긴다.
"""

from __future__ import annotations

import bisect
import hashlib
import sqlite3
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from common import trading_calendar  # noqa: E402
from ingest.store import krx_store  # noqa: E402
from ingest.store.migrations import migrate_path  # noqa: E402
from supply import text  # noqa: E402

#: 평일에서 삼일절(20240301 금)을 뺀 달력.
휴장 = {"20240301"}
달력날 = tuple(d for d in pd.bdate_range("2024-02-01", "2024-04-30").strftime("%Y%m%d")
             if d not in 휴장)
MODEL = "snunlp/KR-FinBert-SC"

#: 제목 → (p_neg, p_neu, p_pos)
제목들 = {
    "주요사항보고서(자기주식취득결정)": (0.1, 0.2, 0.7),
    "[기재정정]사업보고서 (2023.12)": (0.2, 0.6, 0.2),
    "조회공시요구(풍문또는보도)": (0.7, 0.2, 0.1),
}

#: (rcept_no, code, 제목, rcept_dt, rm) — 시험이 심는 원본 사실
공시들 = [
    ("20240312000001", "005930", "주요사항보고서(자기주식취득결정)", "20240312", None),
    ("20240312000002", "005930", "조회공시요구(풍문또는보도)", "20240312", "정"),    # 뒤에 정정됨
    ("20240229000003", "000660", "[기재정정]사업보고서 (2023.12)", "20240229", "연"),  # 목
    ("20240302000004", "000660", "주요사항보고서(자기주식취득결정)", "20240302", None),  # 토
    ("20240315000005", "005930", "조회공시요구(풍문또는보도)", "20240315", "철"),    # 뒤에 철회됨
    ("20240313000006", "035720", "점수가 아직 없는 제목", "20240313", None),
]
종목들 = ("005930", "000660", "035720")


def 다음거래일(day: str) -> str:
    """정문의 `dart_known_at` 을 부르지 않고 **따로** 센다."""
    return 달력날[bisect.bisect_right(달력날, day)]


@pytest.fixture
def db(tmp_path, monkeypatch):
    경로 = tmp_path / "t.db"
    monkeypatch.setattr(krx_store, "DB_PATH", 경로)
    krx_store.init_db()
    migrate_path(경로)

    days = frozenset(달력날)
    monkeypatch.setattr(trading_calendar, "_SESSION_CACHE", days)
    monkeypatch.setattr(trading_calendar, "_SESSION_SPAN", (min(days), max(days)))
    monkeypatch.setattr(trading_calendar, "_SESSION_SORTED", None)
    monkeypatch.setattr(trading_calendar, "_SORTED_FOR", None)

    with sqlite3.connect(경로) as conn:
        for 제목, (neg, neu, pos) in 제목들.items():
            conn.execute(
                "INSERT INTO text_signal (text_sha, report_nm, model_id, revision, p_neg, "
                "p_neu, p_pos, scored_at) VALUES (?,?,?,?,?,?,?,?)",
                (hashlib.sha256(제목.encode("utf-8")).hexdigest()[:16], 제목, MODEL, "rev1",
                 neg, neu, pos, "2026-09-04T14:05:00+09:00"))
        for rn, code, 제목, dt, rm in 공시들:
            conn.execute(
                "INSERT INTO dart_disclosure (rcept_no, corp_code, corp_name, stock_code, "
                "corp_cls, report_nm, flr_nm, rcept_dt, rm, collected_at) "
                "VALUES (?,?,?,?,?,?,?,?,?,?)",
                (rn, "C" + code, code, code, "Y", 제목, code, dt, rm,
                 "2026-09-11T00:00:00+09:00"))
    return 경로


def 패널() -> pd.DataFrame:
    return pd.DataFrame([(d, c) for d in 달력날 for c in 종목들], columns=["bas_dd", "code"])


# ==================================================
# 1. as_of 를 빠뜨릴 수 없다
# ==================================================
def test_as_of_없이_부르면_터진다():
    with pytest.raises(TypeError):
        text.text_as_of("20240313")                                   # type: ignore[call-arg]
    with pytest.raises(TypeError):
        text.attach_text(pd.DataFrame({"bas_dd": ["20240313"], "code": ["005930"]}))  # type: ignore[call-arg]


def test_as_of_보다_뒤의_행을_물으면_세운다(db):
    with pytest.raises(ValueError) as 잡힘:
        text.text_as_of("20240313", as_of="2024-03-13", db_path=db)
    assert "as_of" in str(잡힘.value)
    with pytest.raises(ValueError):
        text.attach_text(패널(), as_of="2024-03-01", db_path=db)


# ==================================================
# 2. 시점 — 접수일 다음 거래일
# ==================================================
def test_접수일_당일_행에는_없고_다음_거래일_행에_새로_보인다(db):
    당일 = text.text_as_of("20240312", as_of="2024-03-13", codes=["005930"], db_path=db)
    다음날 = text.text_as_of("20240313", as_of="2024-03-14", codes=["005930"], db_path=db)
    assert len(당일) == 0
    assert list(다음날["rcept_no"]) == ["20240312000001", "20240312000002"]
    assert set(다음날["known_at"]) == {"20240313"}


def test_휴장일과_주말_접수는_그다음_거래일에_모인다(db):
    """목요일(0229) 접수 → 금요일 삼일절 → 월요일(0304). 토요일(0302) 접수도 월요일."""
    월요일 = text.text_as_of("20240304", as_of="2024-03-05", codes=["000660"], db_path=db)
    assert list(월요일["rcept_no"]) == ["20240229000003", "20240302000004"]


def test_정문이_낸_행은_모두_독립으로_센_다음_거래일과_같다(db):
    """점 조회의 모든 행을 따로 센 다음 거래일과 대조한다 — 누수 0."""
    전부 = text.text_as_of("20240430", as_of="2024-05-01", start="20240201", db_path=db)
    assert len(전부) == len(공시들)
    for rcept_dt, known in zip(전부["rcept_dt"], 전부["known_at"], strict=True):
        assert known == 다음거래일(rcept_dt)


def test_기간으로_물으면_그_사이에_새로_보인_공시를_준다(db):
    사이 = text.text_as_of("20240318", as_of="2024-03-19", start="20240313", db_path=db)
    assert list(사이["rcept_no"]) == ["20240312000001", "20240312000002",
                                      "20240313000006", "20240315000005"]


def test_텍스트_반출과_규칙_모델이_같다():
    """HF 에 나간 텍스트 반출과 시점 규칙·모델 이름이 갈라지면 여기서 걸린다."""
    from scripts import export_text_signal as ets
    from scripts import score_text_signal as sts

    assert text.TEXT_KNOWN_RULE == ets.KNOWN_RULE
    assert text.DEFAULT_MODEL_ID == sts.MODEL_ID


# ==================================================
# 3. 음성 대조군 — 접수일 당일로 붙이면 붉어지는가
# ==================================================
def test_음성대조군_접수일_당일로_붙이면_붉어진다(db):
    """틀린 구현은 **시점 하나만** 다르다. 무엇이 같은가도 수치로 확인한다.

    같아야 하는 것: 거래일에 접수된 공시는 두 구현이 모두 한 번씩 센다(합계가 같다).
    다른 것: 틀린 구현은 토요일 접수를 붙일 행이 없어 놓치고, 나머지는 하루씩 앞선다.
    """
    표 = 패널()
    옳음 = text.attach_text(표, as_of="2024-05-01", db_path=db)

    기대: dict = {}
    틀린: dict = {}
    for _, code, _, dt, _ in 공시들:
        기대[(다음거래일(dt), code)] = 기대.get((다음거래일(dt), code), 0) + 1
        틀린[(dt, code)] = 틀린.get((dt, code), 0) + 1
    틀림 = 표.assign(text_n=[틀린.get((d, c), 0) for d, c in
                             zip(표["bas_dd"], 표["code"], strict=True)])

    # 정문 = 독립 기대값 — 수집 구간 안의 모든 행에서 (밖은 결측이라 따로 본다)
    옳은값 = [기대.get((d, c), 0) for d, c in zip(표["bas_dd"], 표["code"], strict=True)]
    수집안 = 옳음["text_n"].notna()
    assert 수집안.sum() > 0, "대조할 행이 없으면 초록이 무의미하다"
    assert (옳음.loc[수집안, "text_n"].astype(int).to_numpy()
            == pd.Series(옳은값)[수집안.to_numpy()].to_numpy()).all()

    # 틀린 구현이 센 공시는 전부 다음 거래일보다 이른 행에 붙었다 — 샌 행
    샌 = {(d, c) for d, c, n in zip(틀림["bas_dd"], 틀림["code"], 틀림["text_n"], strict=True)
         if n > 0}
    거래일접수 = sum(1 for *_, dt, _ in 공시들 if dt in 달력날)
    print(f"\n  정문 합계 {int(옳음['text_n'].sum())} · 틀린 구현 합계 {int(틀림['text_n'].sum())}"
          f" (거래일 접수 {거래일접수}) · 샌 행 {sorted(샌)}")
    assert 샌 == {("20240229", "000660"), ("20240312", "005930"),
                 ("20240313", "035720"), ("20240315", "005930")}
    assert int(틀림["text_n"].sum()) == 거래일접수 == len(공시들) - 1, "토요일 접수만 놓친다"
    assert int(옳음["text_n"].sum()) == len(공시들), "정문은 여섯 건을 모두 한 번씩 센다"


# ==================================================
# 4. 값 — 건수·평균·결측·rm
# ==================================================
def test_rm_으로_거르지_않는다(db):
    """'정'(뒤에 정정됨)·'철'(뒤에 철회됨) 공시도 그날에는 있었던 공시다."""
    전부 = text.text_as_of("20240430", as_of="2024-05-01", start="20240201", db_path=db)
    assert {"20240312000002", "20240315000005"} <= set(전부["rcept_no"])


def test_제목_해시가_text_sha_와_같다(db):
    """제목으로 이은 결과가 해시로 이은 것과 같다는 계약을 못박는다."""
    전부 = text.text_as_of("20240430", as_of="2024-05-01", start="20240201", db_path=db)
    점수있음 = 전부[전부["text_sha"].notna()]
    assert len(점수있음) == 5
    for 제목, sha in zip(점수있음["report_nm"], 점수있음["text_sha"], strict=True):
        assert hashlib.sha256(제목.encode("utf-8")).hexdigest()[:16] == sha


def test_attach_는_건수와_평균을_붙이고_공시_없는_날은_0이다(db):
    표 = pd.DataFrame({"bas_dd": ["20240313", "20240312", "20240304", "20240314"],
                      "code": ["005930", "005930", "000660", "035720"]})
    붙인 = text.attach_text(표, as_of="2024-05-01", db_path=db)

    assert list(붙인["text_n"]) == [2, 0, 2, 1]
    assert 붙인["text_p_pos"].iloc[0] == pytest.approx((0.7 + 0.1) / 2)
    assert pd.isna(붙인["text_p_pos"].iloc[1]), "공시가 없던 날의 확률은 결측"
    assert 붙인["text_p_neg"].iloc[2] == pytest.approx((0.2 + 0.1) / 2)
    assert pd.isna(붙인["text_p_neg"].iloc[3]), "점수가 없는 제목은 건수에는 들고 확률은 결측"


def test_수집_구간_밖은_0이_아니라_결측이다(db):
    """첫 접수 20240229 → 20240304 부터, 마지막 접수 20240315 → 20240318 까지가 수집 구간이다."""
    표 = pd.DataFrame({"bas_dd": ["20240228", "20240304", "20240318", "20240319"],
                      "code": ["005930"] * 4})
    붙인 = text.attach_text(표, as_of="2024-05-01", db_path=db)
    assert pd.isna(붙인["text_n"].iloc[0])
    assert 붙인["text_n"].iloc[1] == 0
    assert 붙인["text_n"].iloc[2] == 1
    assert pd.isna(붙인["text_n"].iloc[3]), "안 받은 날을 '공시 없음' 으로 읽으면 조용히 틀린다"


def test_모델이_다르면_건수는_같고_확률은_결측이다(db):
    표 = pd.DataFrame({"bas_dd": ["20240313"], "code": ["005930"]})
    붙인 = text.attach_text(표, as_of="2024-05-01", model_id="다른/모델", db_path=db)
    assert 붙인["text_n"].iloc[0] == 2
    assert 붙인[["text_p_neg", "text_p_neu", "text_p_pos"]].isna().all(axis=None)


# ==================================================
# 5. 모양
# ==================================================
def test_빈_결과에도_칸이_남는다(db):
    빈 = text.text_as_of("20240305", as_of="2024-03-06", db_path=db)
    assert len(빈) == 0 and list(빈.columns) == list(text.TEXT_COLUMNS)

    없음 = text.text_as_of("20240313", as_of="2024-03-14", codes=[], db_path=db)
    assert list(없음.columns) == list(text.TEXT_COLUMNS)

    빈표 = text.attach_text(pd.DataFrame({"bas_dd": [], "code": []}), as_of="2024-05-01",
                           db_path=db)
    for col in text.TEXT_DAILY_COLUMNS:
        assert col in 빈표.columns


def test_attach_는_입력_순서를_지키고_점_조회와_같다(db):
    섞은 = pd.DataFrame({"bas_dd": ["20240318", "20240304", "20240313", "20240305"],
                        "code": ["005930", "000660", "005930", "000660"]})
    붙인 = text.attach_text(섞은, as_of="2024-05-01", db_path=db)
    assert list(붙인["bas_dd"]) == list(섞은["bas_dd"])
    for d, c, n in zip(붙인["bas_dd"], 붙인["code"], 붙인["text_n"], strict=True):
        점 = text.text_as_of(d, as_of="2024-05-01", codes=[c], db_path=db)
        assert len(점) == n, f"{d} {c}"
