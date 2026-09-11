"""재무 정문 — 결산기가 아니라 접수일로, 정정본은 정정일로 붙는가.

**무엇을 지키려는 시험인가.** 재무는 결산기와 세상이 알게 된 날이 석 달 벌어진다(실측
2026-09-11 중앙값 87일 · 최대 2,467일). 결산기에 붙이면 미래가 학습에 들어가는데
**예외가 나지 않고 성능만 좋아진다.**

🔴 **누수 검사에는 음성 대조군을 함께 둔다.** 검사가 정문과 같은 계산(`dart_known_at`)을
   쓰면 정문이 틀릴 때 검사도 같이 틀려 초록이 나온다. 그래서 검사는 이 파일 안에서
   달력을 이분탐색해 **따로** 세고, 일부러 틀린 구현 셋이 실제로 붉어지는지 본다.

    ① 결산기 끝으로 붙인다      가장 흔한 실수 — 석 달 미래
    ② 접수일 당일로 붙인다      계약보다 하루 빠르다
    ③ 원본 접수일로 당긴다      정정된 숫자를 원본 날짜에 붙인다 (재작성 누수)

틀린 구현은 **시점 규칙 하나만** 다르고 고르는 규칙(결산이 가장 늦은 것)은 정문과 같다.
다른 것이 하나뿐이어야 붉은 불이 무엇 때문인지 말할 수 있다.

망을 타지 않는다. DB 는 임시 경로를 만들어 `db_path` 로 넘긴다.
"""

from __future__ import annotations

import bisect
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
from supply import clock, financial  # noqa: E402

# ── 시험이 심는 원본 사실 ─────────────────────────────────────────────────────
# 정문이 아니라 **여기서** 답을 꺼낸다. 기대값을 정문에서 뽑으면 항등식이 된다.

#: 평일에서 휴장 둘을 뺀 달력. 날짜 계산이 아니라 달력으로 세는지 보려는 것이다.
#:   20210301(월) 삼일절 — 금요일 접수의 다음 거래일이 화요일이 된다
#:   20211231(금) 연말 휴장
휴장 = {"20210301", "20211231"}
달력날 = tuple(d for d in pd.bdate_range("2015-01-02", "2024-12-31").strftime("%Y%m%d")
             if d not in 휴장)

보고서들 = [
    # 삼성전자처럼 — 정상 제출. IS 와 CIS 에 당기순이익이 둘 다 있다(값을 일부러 다르게).
    dict(code="005930", corp="00126380", year=2022, rcept_no="20230307000542",
         rcept_dt="20230307", fs="CFS", title="사업보고서 (2022.12)",
         lines=[("BS", "ifrs-full_Assets", "자산총계", 100.0),
                ("BS", "ifrs-full_Liabilities", "부채총계", 40.0),
                ("BS", "ifrs-full_Equity", "자본총계", 60.0),
                ("BS", "-표준계정코드 미사용-", "기타자본항목", 7.0),
                ("CIS", "ifrs-full_Revenue", "매출액", 300.0),
                ("CIS", "dart_OperatingIncomeLoss", "영업이익", 50.0),
                ("IS", "ifrs-full_ProfitLoss", "당기순이익", 30.0),
                ("CIS", "ifrs-full_ProfitLoss", "당기순이익", 31.0)]),
    dict(code="005930", corp="00126380", year=2023, rcept_no="20240312000736",
         rcept_dt="20240312", fs="CFS", title="사업보고서 (2023.12)",
         lines=[("BS", "ifrs-full_Assets", "자산총계", 110.0),
                ("CIS", "ifrs-full_Revenue", "매출액", 320.0)]),
    # 두산처럼 — 옛 해(FY2016)가 2022년 정정본으로 늦게 드러난다. 옛 택소노미 `ifrs_`.
    dict(code="000150", corp="00117212", year=2016, rcept_no="20220217000111",
         rcept_dt="20220217", fs="CFS", title="[기재정정]사업보고서 (2016.12)",
         lines=[("BS", "ifrs_Assets", "자산총계", 500.0),
                ("IS", "ifrs_Revenue", "매출액", 900.0)]),
    # 금요일 접수 → 월요일 삼일절 → 화요일부터 보인다
    dict(code="000150", corp="00117212", year=2020, rcept_no="20210226000333",
         rcept_dt="20210226", fs="CFS", title="사업보고서 (2020.12)",
         lines=[("BS", "ifrs-full_Assets", "자산총계", 700.0)]),
    dict(code="000150", corp="00117212", year=2021, rcept_no="20220316000444",
         rcept_dt="20220316", fs="CFS", title="사업보고서 (2021.12)",
         lines=[("BS", "ifrs-full_Assets", "자산총계", 720.0)]),
    # 6월 결산 · 별도 · 매출 줄 없음 · 영문이 섞인 종목코드
    dict(code="0015N0", corp="00154718", year=2015, rcept_no="20150923000335",
         rcept_dt="20150923", fs="OFS", title="사업보고서 (2015.06)",
         lines=[("BS", "ifrs_Assets", "자산총계", 80.0)]),
    dict(code="0015N0", corp="00154718", year=2016, rcept_no="20161005000555",
         rcept_dt="20161005", fs="OFS", title="[기재정정]사업보고서 (2016.06)",
         lines=[("BS", "ifrs_Assets", "자산총계", 85.0)]),
]

#: 정정본의 **원본** — 공시 목록에만 있다. 대조군 ③ 이 이 날짜로 당긴다.
#: `rm='정연'` 은 나중에 붙은 표시다(뒤에 정정됐다). 정문은 이 칸을 읽지 않는다.
원본공시 = {"20220217000111": dict(rcept_no="20170331000222", rcept_dt="20170331",
                                   title="사업보고서 (2016.12)", rm="정연")}

접수일 = {r["rcept_no"]: r["rcept_dt"] for r in 보고서들}
연도 = {r["rcept_no"]: r["year"] for r in 보고서들}
모든종목 = ("005930", "000150", "0015N0", "999990")      # 999990 은 보고서가 없다


def 다음거래일(day: str) -> str:
    """정문의 `dart_known_at` 을 부르지 않고 **따로** 센다."""
    return 달력날[bisect.bisect_right(달력날, day)]


def 결산기끝(r: dict) -> str:
    """제목의 `(YYYY.MM)` 달의 말일. 대조군 ① 이 이 날짜로 붙인다."""
    ym = r["title"].split("(")[-1].rstrip(")")
    y, m = ym.split(".")
    return (pd.Timestamp(int(y), int(m), 1) + pd.offsets.MonthEnd(0)).strftime("%Y%m%d")


def 심는다(conn: sqlite3.Connection) -> None:
    for r in 보고서들:
        for i, (sj, aid, nm, amt) in enumerate(r["lines"]):
            conn.execute(
                "INSERT INTO dart_financial (corp_code, stock_code, corp_name, bsns_year, "
                "reprt_code, fs_div, sj_div, account_id, account_nm, account_detail, ord, "
                "currency, thstrm_nm, thstrm_amount, rcept_no, rcept_dt, report_nm, collected_at) "
                "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (r["corp"], r["code"], r["code"], r["year"], "11011", r["fs"], sj, aid, nm,
                 "-", i, "KRW", "제 1 기", amt, r["rcept_no"], r["rcept_dt"], "사업보고서",
                 "2026-09-11T00:00:00+09:00"))
        conn.execute(
            "INSERT INTO dart_disclosure (rcept_no, corp_code, corp_name, stock_code, corp_cls, "
            "report_nm, flr_nm, rcept_dt, rm, collected_at) VALUES (?,?,?,?,?,?,?,?,?,?)",
            (r["rcept_no"], r["corp"], r["code"], r["code"], "Y", r["title"], r["code"],
             r["rcept_dt"], "연", "2026-09-11T00:00:00+09:00"))
    for 정정, o in 원본공시.items():
        r = next(x for x in 보고서들 if x["rcept_no"] == 정정)
        conn.execute(
            "INSERT INTO dart_disclosure (rcept_no, corp_code, corp_name, stock_code, corp_cls, "
            "report_nm, flr_nm, rcept_dt, rm, collected_at) VALUES (?,?,?,?,?,?,?,?,?,?)",
            (o["rcept_no"], r["corp"], r["code"], r["code"], "Y", o["title"], r["code"],
             o["rcept_dt"], o["rm"], "2026-09-11T00:00:00+09:00"))
    conn.commit()


@pytest.fixture
def db(tmp_path, monkeypatch):
    """이 시험만 쓰는 DB. 달력은 마이그레이션이 끝난 **뒤에** 끼운다 — 앞에 끼우면
    마이그레이션이 달력을 다시 읽어 덮을 수 있다."""
    경로 = tmp_path / "f.db"
    monkeypatch.setattr(krx_store, "DB_PATH", 경로)
    krx_store.init_db()
    migrate_path(경로)

    days = frozenset(달력날)
    monkeypatch.setattr(trading_calendar, "_SESSION_CACHE", days)
    monkeypatch.setattr(trading_calendar, "_SESSION_SPAN", (min(days), max(days)))
    monkeypatch.setattr(trading_calendar, "_SESSION_SORTED", None)
    monkeypatch.setattr(trading_calendar, "_SORTED_FOR", None)

    with sqlite3.connect(경로) as conn:
        심는다(conn)
    return 경로


def 패널(codes=모든종목) -> pd.DataFrame:
    return pd.DataFrame([(d, c) for d in 달력날 for c in codes], columns=["bas_dd", "code"])


def 샌_행(붙인: pd.DataFrame) -> pd.DataFrame:
    """행의 날짜보다 **늦게 알게 된** 보고서가 붙은 행.

    접수일은 정문이 낸 `fin_rcept_dt` 가 아니라 시험이 심은 원본(`접수일`)에서 꺼낸다.
    """
    sub = 붙인.loc[붙인["fin_rcept_no"].notna(), ["bas_dd", "code", "fin_rcept_no"]].copy()
    sub["알게된날"] = sub["fin_rcept_no"].map(lambda rn: 다음거래일(접수일[rn]))
    return sub[sub["알게된날"] > sub["bas_dd"]]


def 규칙으로_붙인다(표: pd.DataFrame, 시점) -> pd.DataFrame:
    """`시점(보고서) -> YYYYMMDD` 로 알게 된 날을 세우고, **결산이 가장 늦은 것**을 고른다.

    고르는 규칙은 정문과 같고 시점 규칙만 바꿔 끼운다. 정문을 부르지 않는다.
    """
    사건: dict = {}
    for r in 보고서들:
        사건.setdefault(r["code"], []).append((시점(r), r["year"], r["rcept_no"]))
    답: dict = {}
    for code, evs in 사건.items():
        evs.sort()
        i, best = 0, None
        for d in 달력날:
            while i < len(evs) and evs[i][0] <= d:
                known, year, rn = evs[i]
                if best is None or (year, known) > (best[0], best[1]):
                    best = (year, known, rn)
                i += 1
            답[(d, code)] = best[2] if best else None
    return 표.assign(fin_rcept_no=[답.get((d, c)) for d, c in
                                   zip(표["bas_dd"], 표["code"], strict=True)])


def 한종목(frame: pd.DataFrame, code: str) -> pd.Series:
    rows = frame[frame["code"] == code]
    assert len(rows) == 1, f"{code} 행이 {len(rows)}개다"
    return rows.iloc[0]


# ==================================================
# 1. as_of 를 빠뜨릴 수 없다
# ==================================================
def test_as_of_없이_부르면_터진다():
    with pytest.raises(TypeError):
        financial.financial_as_of("20240313")                          # type: ignore[call-arg]
    with pytest.raises(TypeError):
        financial.attach_financial(패널(("005930",)))                  # type: ignore[call-arg]
    with pytest.raises(TypeError):
        financial.financial_lines_as_of("20240313")                    # type: ignore[call-arg]


def test_as_of_는_키워드로만_받는다():
    with pytest.raises(TypeError):
        financial.financial_as_of("20240313", "2024-03-14")            # type: ignore[misc]


def test_as_of_보다_뒤의_행을_물으면_세운다(db):
    """빈 표를 주면 *"그날 재무가 없었다"* 로 읽힌다. 세워서 무엇을 해야 하는지 알린다."""
    with pytest.raises(ValueError) as 잡힘:
        financial.financial_as_of("20240313", as_of="2024-03-13", db_path=db)
    assert "as_of" in str(잡힘.value)

    with pytest.raises(ValueError):
        financial.attach_financial(패널(("005930",)), as_of="2020-01-01", db_path=db)


# ==================================================
# 2. 시점 규칙 — 접수일 다음 거래일
# ==================================================
def test_접수일_당일_행에는_안_보이고_다음_거래일_행부터_보인다(db):
    """2024-03-12(화) 접수 → 03-12 행에는 FY2022 → 03-13 행부터 FY2023."""
    당일 = financial.financial_as_of("20240312", as_of="2024-03-13", db_path=db)
    다음날 = financial.financial_as_of("20240313", as_of="2024-03-14", db_path=db)

    assert 한종목(당일, "005930")["fin_bsns_year"] == 2022
    행 = 한종목(다음날, "005930")
    assert 행["fin_bsns_year"] == 2023
    assert 행["fin_rcept_dt"] == "20240312"
    assert 행["fin_known_at"] == "20240313"


def test_휴장일을_건너_다음_거래일에_보인다(db):
    """🔴 금요일(20210226) 접수의 다음 거래일은 월요일이 아니다 — 월요일이 삼일절이다."""
    assert clock.dart_known_at(["20210226"], db_path=db) == {"20210226": "20210302"}

    금요일 = financial.financial_as_of("20210226", as_of="2021-02-27", db_path=db)
    화요일 = financial.financial_as_of("20210302", as_of="2021-03-03", db_path=db)

    assert "000150" not in set(금요일["code"])
    assert 한종목(화요일, "000150")["fin_known_at"] == "20210302"


def test_달력_뒤_접수는_시점을_지어내지_않는다(db):
    """다음 거래일을 모르면 None 이다. 달력 앞은 첫 거래일로 늦춰 둔다(늦는 방향)."""
    답 = clock.dart_known_at(["20241231", "20250105", "20141230"], db_path=db)
    assert 답["20241231"] is None, "마지막 거래일의 다음 거래일은 아직 모른다"
    assert 답["20250105"] is None
    assert 답["20141230"] == 달력날[0]


def test_텍스트_반출과_시점_규칙이_같다(db):
    """재무와 공시 텍스트는 같은 DART 접수일을 쓴다. 규칙이 둘이 되면 안 된다.

    HF 에 이미 나간 텍스트 반출(`scripts/export_text_signal.py`)과 이름·계산을 대조한다.
    """
    from scripts import export_text_signal as ets

    assert clock.DART_KNOWN_RULE == ets.KNOWN_RULE

    날들 = sorted({r["rcept_dt"] for r in 보고서들} | {"20170331", "20211230", "20211231"})
    정문 = clock.dart_known_at(날들, db_path=db)
    반출 = ets._next_session_map(날들, frozenset(달력날))
    assert 정문 == 반출


# ==================================================
# 3. 정정본과 "가장 최근"
# ==================================================
def test_정정본은_원본_날짜가_아니라_정정일까지_안_보인다(db):
    """두산처럼 — FY2016 원본은 2017-03-31 이지만 우리가 가진 숫자는 2022-02-17 정정본이다."""
    원본뒤 = financial.financial_as_of("20170403", as_of="2017-04-04", db_path=db)
    assert "000150" not in set(원본뒤["code"]), "정정된 숫자가 원본 날짜에 붙었다"


def test_늦게_드러난_옛_해가_최신_해를_밀어내지_않는다(db):
    """🔴 20220218 에 FY2016 정정본이 보이기 시작한다. 그래도 붙는 것은 FY2020 이다.

    알게 된 순서로 최신을 고르면(`merge_asof` 를 접수일에 그대로 걸면) FY2016 이 붙는다 —
    그 오답을 이 자리에서 만들어 보여서, 시험이 실제로 가르는 것을 확인한다.
    """
    행 = 한종목(financial.financial_as_of("20220218", as_of="2022-02-19", db_path=db), "000150")
    assert 행["fin_bsns_year"] == 2020
    assert 행["fin_total_assets"] == 700.0

    알게된순 = sorted((다음거래일(r["rcept_dt"]), r["rcept_no"]) for r in 보고서들
                    if r["code"] == "000150" and 다음거래일(r["rcept_dt"]) <= "20220218")
    assert 연도[알게된순[-1][1]] == 2016, "대조: 알게 된 순서로 고르면 FY2016 이 붙는다"

    # attach 도 같은 답이어야 한다
    붙인 = financial.attach_financial(pd.DataFrame({"bas_dd": ["20220218"], "code": ["000150"]}),
                                     as_of="2022-02-19", db_path=db)
    assert 붙인["fin_bsns_year"].iloc[0] == 2020


def test_정정본에는_정정_표시가_붙는다(db):
    """첫 제출본이 아니라는 사실은 그 보고서 자신의 제목에서 읽는다 — 접수 시점에 알 수 있었다."""
    행 = 한종목(financial.financial_as_of("20161006", as_of="2016-10-07", db_path=db), "0015N0")
    assert 행["fin_amended"] is True or 행["fin_amended"] == True        # noqa: E712
    assert 행["fin_report_nm"].startswith("[기재정정]")

    원본 = 한종목(financial.financial_as_of("20150924", as_of="2015-09-25", db_path=db), "0015N0")
    assert 원본["fin_amended"] is False or 원본["fin_amended"] == False  # noqa: E712


# ==================================================
# 4. 음성 대조군 — 틀린 구현은 실제로 붉어지는가
# ==================================================
def test_정문은_독립_구현과_모든_행에서_같고_누수가_0이다(db):
    """정문과 독립 구현(같은 규칙을 이 파일에서 따로 짠 것)이 모든 행에서 같은 보고서를 붙인다."""
    표 = 패널()
    붙인 = financial.attach_financial(표, as_of="2025-01-01", db_path=db)
    기대 = 규칙으로_붙인다(표, lambda r: 다음거래일(r["rcept_dt"]))

    어긋남 = (붙인["fin_rcept_no"].fillna("-") != 기대["fin_rcept_no"].fillna("-")).sum()
    print(f"\n  행 {len(표):,} · 정문 ≠ 독립 구현 {어긋남} · 누수 {len(샌_행(붙인))}")
    assert 어긋남 == 0
    assert len(샌_행(붙인)) == 0
    assert len(샌_행(기대)) == 0, "검사기 자신이 올바른 구현을 붉게 본다"


def test_음성대조군_결산기로_붙이면_붉어진다(db):
    """① 가장 흔한 실수. 붉어져야 하고, **무엇이 같은가**도 수치로 확인한다.

    같아야 하는 것: 모든 보고서가 드러난 마지막 날에는 두 구현이 **같은 보고서**를 붙인다.
    다른 것은 시점뿐이다 — 그래야 붉은 불이 시점 때문이라고 말할 수 있다.
    """
    표 = 패널()
    옳음 = 규칙으로_붙인다(표, lambda r: 다음거래일(r["rcept_dt"]))
    틀림 = 규칙으로_붙인다(표, 결산기끝)
    샘 = 샌_행(틀림)

    마지막 = 표["bas_dd"].max()
    같은가 = (옳음.loc[옳음["bas_dd"] == 마지막, "fin_rcept_no"].fillna("-").to_numpy()
             == 틀림.loc[틀림["bas_dd"] == 마지막, "fin_rcept_no"].fillna("-").to_numpy())
    미래일수 = (pd.to_datetime(샘["알게된날"]) - pd.to_datetime(샘["bas_dd"])).dt.days
    print(f"\n  마지막 날 같은 보고서 {int(같은가.sum())}/{len(같은가)} · "
          f"샌 행 {len(샘):,} · 최대 {int(미래일수.max())}일 미래")

    assert 같은가.all(), "대조군이 시점 말고 다른 것까지 달라졌다"
    assert len(샘) > 0, "결산기로 붙였는데 검사가 못 잡았다"
    # 2024-01-02 에 FY2023(03-12 접수)이 붙는다 — 70일 넘게 미래다
    rows = 샘[(샘["bas_dd"] == "20240102") & (샘["code"] == "005930")]
    assert list(rows["fin_rcept_no"]) == ["20240312000736"]


def test_음성대조군_하루_당기면_붉어진다(db):
    """② 접수일 당일로 붙이면 **접수일 행에서만** 샌다. 어느 행인지 정확히 센다.

    FY2016 정정본 접수일(20220217)은 새지 않는다 — 그날도 FY2020 이 결산이 더 늦어서
    정정본이 붙지 않기 때문이다. 고르는 규칙이 시점 규칙과 맞물린다는 것을 보여 준다.
    """
    틀림 = 규칙으로_붙인다(패널(), lambda r: r["rcept_dt"])
    샘 = {(d, c) for d, c in 샌_행(틀림)[["bas_dd", "code"]].itertuples(index=False)}
    print(f"\n  샌 행 {len(샘)} · {sorted(샘)}")
    assert 샘 == {
        ("20150923", "0015N0"), ("20161005", "0015N0"),
        ("20210226", "000150"), ("20220316", "000150"),
        ("20230307", "005930"), ("20240312", "005930"),
    }


def test_음성대조군_원본_접수일로_당기면_붉어진다(db):
    """③ 정정된 숫자를 원본 날짜에 붙이면 원본 다음 거래일부터 FY2020 이 보일 때까지 샌다."""
    def 원본날(r: dict) -> str:
        o = 원본공시.get(r["rcept_no"])
        return 다음거래일(o["rcept_dt"] if o else r["rcept_dt"])

    틀림 = 규칙으로_붙인다(패널(("000150",)), 원본날)
    샘 = 샌_행(틀림)

    시작 = bisect.bisect_left(달력날, 다음거래일("20170331"))      # 20170403
    끝 = bisect.bisect_left(달력날, 다음거래일("20210226"))        # 20210302 (FY2020 이 이긴다)
    print(f"\n  샌 행 {len(샘):,} · 기대 {끝 - 시작:,} (20170403 ~ 20210301 의 거래일)")
    assert len(샘) == 끝 - 시작
    assert set(샘["fin_rcept_no"]) == {"20220217000111"}


# ==================================================
# 5. 값 — 접두어 · 우선순위 · 결측
# ==================================================
def test_접두어가_달라도_같은_칸에_모인다(db):
    """FY2018 까지 `ifrs_`, FY2019 부터 `ifrs-full_`. 뒷부분으로 모은다."""
    옛 = 한종목(financial.financial_as_of("20150924", as_of="2015-09-25", db_path=db), "0015N0")
    새 = 한종목(financial.financial_as_of("20210302", as_of="2021-03-03", db_path=db), "000150")
    assert 옛["fin_total_assets"] == 80.0
    assert 새["fin_total_assets"] == 700.0


def test_손익이_IS와_CIS에_둘_다_있으면_IS_를_고른다(db):
    """실측으로는 값이 같지만(557건 중 다른 것 0) 순서를 정해 두어야 몇 번을 돌려도 같다."""
    행 = 한종목(financial.financial_as_of("20230308", as_of="2023-03-09", db_path=db), "005930")
    assert 행["fin_net_income"] == 30.0
    assert 행["fin_operating_income"] == 50.0, "dart_ 접두어 영업이익"
    assert 행["fin_revenue"] == 300.0


def test_결측은_결측이다(db):
    """매출 줄이 없는 보고서의 매출은 0 이 아니라 NaN 이다. 보고서가 없는 종목은 빈 칸이다."""
    행 = 한종목(financial.financial_as_of("20150924", as_of="2015-09-25", db_path=db), "0015N0")
    assert pd.isna(행["fin_revenue"])
    assert 행["fin_fs_div"] == "OFS"

    붙인 = financial.attach_financial(pd.DataFrame({"bas_dd": ["20240313"], "code": ["999990"]}),
                                     as_of="2024-03-14", db_path=db)
    assert 붙인[list(financial.FINANCIAL_COLUMNS)].isna().all(axis=None)


def test_결산월을_12월로_가정하지_않는다(db):
    행 = 한종목(financial.financial_as_of("20150924", as_of="2015-09-25", db_path=db), "0015N0")
    assert 행["fin_period_end"] == "20150630"


def test_제목을_못_이으면_모른다고_둔다(db):
    """공시 목록에 없으면 결산월도 정정 여부도 **모른다**. 12월·정정 아님으로 채우지 않는다."""
    with sqlite3.connect(db) as conn:
        conn.execute("DELETE FROM dart_disclosure WHERE rcept_no = '20150923000335'")
    행 = 한종목(financial.financial_as_of("20150924", as_of="2015-09-25", db_path=db), "0015N0")
    assert 행["fin_period_end"] is None
    assert 행["fin_amended"] is None
    assert 행["fin_total_assets"] == 80.0, "제목이 없어도 값은 붙는다"


# ==================================================
# 6. 표의 모양이 계약이다
# ==================================================
def test_빈_결과에도_칸이_남는다(db):
    빈 = financial.financial_as_of("20150105", as_of="2015-01-06", db_path=db)
    assert len(빈) == 0
    assert list(빈.columns) == ["code", *financial.FINANCIAL_COLUMNS]

    없음 = financial.financial_as_of("20240313", as_of="2024-03-14", codes=[], db_path=db)
    assert list(없음.columns) == ["code", *financial.FINANCIAL_COLUMNS]

    줄 = financial.financial_lines_as_of("20150105", as_of="2015-01-06", db_path=db)
    assert list(줄.columns) == list(financial.FINANCIAL_LINE_COLUMNS)


def test_attach_는_입력_순서를_지키고_점_조회와_같다(db):
    표 = pd.DataFrame({"bas_dd": ["20240313", "20150924", "20220218", "20240312", "20170403"],
                      "code": ["005930", "0015N0", "000150", "005930", "000150"]})
    붙인 = financial.attach_financial(표, as_of="2025-01-01", db_path=db)
    assert list(붙인["bas_dd"]) == list(표["bas_dd"])
    assert list(붙인["code"]) == list(표["code"])

    for d, c, rn in zip(붙인["bas_dd"], 붙인["code"], 붙인["fin_rcept_no"], strict=True):
        점 = financial.financial_as_of(d, as_of="2025-01-01", codes=[c], db_path=db)
        기대 = 점["fin_rcept_no"].iloc[0] if len(점) else None
        assert (rn if isinstance(rn, str) else None) == 기대, f"{d} {c}"


def test_긴_표는_같은_보고서의_모든_줄을_준다(db):
    줄 = financial.financial_lines_as_of("20230308", as_of="2023-03-09", codes=["005930"],
                                        db_path=db)
    assert len(줄) == 8
    assert set(줄["fin_rcept_no"]) == {"20230307000542"}
    assert 줄.loc[줄["account_id"] == "-표준계정코드 미사용-", "account_key"].isna().all()
    assert "OperatingIncomeLoss" in set(줄["account_key"])

    재무상태표 = financial.financial_lines_as_of("20230308", as_of="2023-03-09",
                                               codes=["005930"], sj_div=["BS"], db_path=db)
    assert len(재무상태표) == 4
