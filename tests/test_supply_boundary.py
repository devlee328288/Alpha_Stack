"""공급 계층 테스트 — 미래가 역류하지 못하는가.

**무엇을 지키려는 테스트인가.** 저장소는 표에 있는 것을 전부 준다. 표에는 오늘까지
들어 있다. 그걸 2020년 폴드 학습에 그대로 쓰면 미래가 섞이는데 **예외가 나지 않고
성능만 좋아진다.** 그래서 여기서 잠그는 것은 기능이 아니라 **경계**다.

    수용 기준
    - `as_of` 를 빠뜨리면 **터진다** (기본값이 없다) ← 이게 설계의 핵심이다
    - `as_of` 시점에 몰랐던 거래일이 결과에 없다
    - 경계에서 하루를 더 주지 않는다 (하루가 곧 누수다)
    - 쓰는 계층이 저장소를 직접 import 하면 **테스트가 실패한다**

마지막 항목이 이 파일의 이름값이다. 규칙을 문서에만 적어 두면 급할 때 지나가고,
지나간 코드는 티가 안 난다.
"""

from __future__ import annotations

import ast
from datetime import date, datetime
from pathlib import Path

import pytest

import supply
from common.trading_calendar import KST

루트 = Path(__file__).resolve().parents[1]

#: 자료를 **쓰는** 계층. 여기서는 저장소를 직접 부르면 안 된다.
소비계층 = ("features", "models", "evaluation", "timeseries")

#: 직접 부르면 안 되는 내부 계층.
내부계층 = ("ingest",)


# ── 경계 ────────────────────────────────────────────────────────────────────

def _import된_모듈(경로: Path) -> set:
    """한 파일이 import 하는 최상위 패키지 이름들."""
    나무 = ast.parse(경로.read_text(encoding="utf-8"), filename=str(경로))
    이름들 = set()
    for 노드 in ast.walk(나무):
        if isinstance(노드, ast.Import):
            이름들.update(a.name.split(".")[0] for a in 노드.names)
        elif isinstance(노드, ast.ImportFrom):
            # `from . import x` 같은 상대 import 는 패키지 안이라 상관없다
            if 노드.level == 0 and 노드.module:
                이름들.add(노드.module.split(".")[0])
    return 이름들


def test_쓰는_계층은_저장소를_직접_부르지_않는다():
    """`as_of` 없이 표를 통째로 읽을 수 있는 길이 남아 있으면 언젠가 그 길로 간다."""
    위반 = []
    for 패키지 in 소비계층:
        폴더 = 루트 / 패키지
        if not 폴더.is_dir():
            continue
        for 파일 in 폴더.rglob("*.py"):
            겹침 = _import된_모듈(파일) & set(내부계층)
            if 겹침:
                위반.append(f"{파일.relative_to(루트)} → {', '.join(sorted(겹침))}")

    assert not 위반, (
        "쓰는 계층이 저장소를 직접 import 했다:\n  "
        + "\n  ".join(위반)
        + "\n\n  왜 막나: 저장소는 표에 있는 것을 전부 준다 — 오늘까지 들어 있다.\n"
        "  할 일: supply 를 지난다. `from supply import index_series` 후\n"
        "         index_series(as_of=...) 로 부른다."
    )


def test_공급_계층은_저장소를_부를_수_있다():
    """경계는 한 방향이다. supply 가 ingest 를 못 부르면 자료를 낼 수가 없다."""
    쓰는것 = set()
    for 파일 in (루트 / "supply").rglob("*.py"):
        쓰는것 |= _import된_모듈(파일)
    assert "ingest" in 쓰는것


# ── as_of 를 빠뜨릴 수 없다 ─────────────────────────────────────────────────

def test_as_of_없이_부르면_터진다():
    """기본값이 '지금' 이면 빠뜨려도 돌아가고, 빠뜨린 코드가 조용히 미래를 본다."""
    with pytest.raises(TypeError):
        supply.index_series()                    # type: ignore[call-arg]


def test_as_of_는_키워드로만_받는다():
    """자리 인자로 받으면 index_name 자리에 흘려 넣는 실수가 난다."""
    with pytest.raises(TypeError):
        supply.index_series("코스피 200", "2020-06-30")   # type: ignore[misc]


# ── 언제부터 알 수 있었나 ────────────────────────────────────────────────────

def test_거래일_자료는_다음_날부터_알_수_있다():
    """실측 2026-08-26 16:10 — 장 마감 40분 뒤에도 당일 자료가 0행이었다."""
    assert supply.known_at("20260825") == datetime(2026, 8, 26, 0, 0, tzinfo=KST)


def test_경계에서_하루를_더_주지_않는다():
    """`as_of` 가 정확히 0시여도 그날 것은 아직 모른다. 하루가 곧 누수다."""
    assert supply.is_known("20260825", "2026-08-26T00:00:00") is True
    assert supply.is_known("20260826", "2026-08-26T00:00:00") is False
    assert supply.is_known("20260826", "2026-08-26T23:59:59") is False
    assert supply.is_known("20260826", "2026-08-27T00:00:00") is True


def test_as_of_를_날짜로_주면_그날_0시로_본다():
    """'8월 26일 시점' 이라고 말했을 때 그날 하루치를 이미 아는 것으로 치면 미래를 본다."""
    assert supply.to_kst("2026-08-26") == datetime(2026, 8, 26, 0, 0, tzinfo=KST)
    assert supply.to_kst(date(2026, 8, 26)) == datetime(2026, 8, 26, 0, 0, tzinfo=KST)


def test_타임존이_없으면_KST_로_본다():
    """이 프로젝트의 시각은 전부 KST 다. UTC 로 읽으면 9시간이 어긋난다."""
    assert supply.to_kst("2026-08-26T15:30:00") == \
        datetime(2026, 8, 26, 15, 30, tzinfo=KST)


def test_알_수_있었던_마지막_거래일():
    assert supply.latest_known_day("2026-08-26") == "20260825"
    assert supply.latest_known_day("2026-08-26T23:59:59") == "20260825"
    assert supply.latest_known_day("2026-08-27") == "20260826"


def test_거래일_표기를_하나로_맞춘다():
    """🔴 표기가 섞이면 문자열 비교가 조용히 뒤집힌다.

    `'-'`(0x2D) 가 `'0'`(0x30) 보다 작아서 하이픈이 든 쪽이 언제나 작다.
    두 표기를 그대로 `min()` 에 넣으면 항상 ISO 쪽이 이기고, 그 값이 `bas_dd <= ?`
    로 들어가면 결과가 0행이 된다.
    """
    assert min("2026-08-21", "20260825") == "2026-08-21"     # 함정 자체를 못 박는다

    assert supply.as_bas_dd("20260821") == "20260821"
    assert supply.as_bas_dd("2026-08-21") == "20260821"
    assert supply.as_bas_dd(date(2026, 8, 21)) == "20260821"
    assert supply.as_bas_dd(datetime(2026, 8, 21, 15, 30)) == "20260821"
    assert supply.as_bas_dd(None) is None


def test_거래일로_읽을_수_없으면_거부한다():
    """조용히 통과시키면 그 값이 SQL 로 들어가 0행을 만든다."""
    with pytest.raises(ValueError) as 오류:
        supply.as_bas_dd("작년 여름")

    assert "쓸 수 있는 꼴" in str(오류.value)


def test_표기_맞추기가_하루를_미루지_않는다():
    """`as_bas_dd` 는 표기만 바꾼다. '언제부터 알 수 있었나' 는 `latest_known_day` 다.

    둘을 섞으면 경계가 하루씩 어긋나는데 그 하루가 곧 누수다.
    """
    assert supply.as_bas_dd("2026-08-26") == "20260826"
    assert supply.latest_known_day("2026-08-26") == "20260825"


def test_읽을_수_없는_as_of_는_거부한다():
    with pytest.raises(ValueError) as 오류:
        supply.to_kst("작년 여름")

    assert "쓸 수 있는 꼴" in str(오류.value)     # 무엇을 해야 하는지까지 알려 준다


def test_경계값을_코드가_답한다():
    """사전등록에 '홀드아웃은 언제부터' 를 손으로 계산하면 하루씩 어긋난다."""
    경계 = supply.as_of_bounds("2026-08-26")

    assert 경계["last_known_trading_day"] == "20260825"
    assert 경계["as_of"].startswith("2026-08-26T00:00:00")


# ── 실제로 잘리는가 ─────────────────────────────────────────────────────────

@pytest.fixture()
def 채워진저장소(tmp_path, monkeypatch):
    """2026-08-21 ~ 08-25 다섯 거래일이 든 임시 저장소."""
    from ingest.store import krx_index, krx_store

    db = tmp_path / "supply.db"
    monkeypatch.setattr(krx_store, "DB_PATH", db)
    monkeypatch.setattr(krx_index, "DB_PATH", db)
    krx_index.init_db()

    for i, bas_dd in enumerate(("20260821", "20260824", "20260825")):
        krx_index._save(bas_dd, "KOSPI", [{
            "index_name": supply.TARGET_INDEX, "index_class": "KOSPI",
            "open": 1000.0 + i, "high": 1000.0 + i, "low": 1000.0 + i,
            "close": 1000.0 + i, "change": 0.0, "change_rate": 0.0,
            "volume": 1, "value": 1, "market_cap": 1,
        }])
    return krx_index


def test_아직_몰랐던_거래일은_나오지_않는다(채워진저장소):
    """저장소에는 들어 있다. 그런데 그 시점에는 아직 몰랐다 — 이게 잘려야 한다."""
    # 저장소를 직접 부르면 셋 다 나온다
    assert len(채워진저장소.series(supply.TARGET_INDEX)) == 3

    # 정문을 지나면 그 시점에 알 수 있었던 것만 나온다
    df = supply.index_series(as_of="2026-08-25")

    assert list(df["date"]) == ["2026-08-21", "2026-08-24"]
    assert "2026-08-25" not in list(df["date"]), "당일 자료가 새어 나왔다"


def test_as_of_를_하루_늦추면_하루가_더_보인다(채워진저장소):
    어제 = supply.index_series(as_of="2026-08-25")
    오늘 = supply.index_series(as_of="2026-08-26")

    assert len(오늘) == len(어제) + 1
    assert 오늘["date"].iloc[-1] == "2026-08-25"


def test_부르는_쪽이_준_end_보다_as_of_가_이기지_못한다(채워진저장소):
    """end 를 넉넉히 줘도 as_of 를 넘지 못한다 — 둘 중 이른 쪽이 이긴다."""
    df = supply.index_series(as_of="2026-08-25", end="20991231")

    assert list(df["date"]) == ["2026-08-21", "2026-08-24"]


def test_end_가_as_of_보다_이르면_end_가_이긴다(채워진저장소):
    df = supply.index_series(as_of="2026-08-26", end="20260821")

    assert list(df["date"]) == ["2026-08-21"]


def test_ISO_로_준_end_가_결과를_0행으로_만들지_않는다(채워진저장소):
    """🔴 문자열 비교가 조용히 뒤집히던 자리다.

        min('2026-08-21', '20260825') == '2026-08-21'

    `'-'`(0x2D) 가 `'0'`(0x30) 보다 작아서 **하이픈이 든 쪽이 언제나 작다.**
    그 값이 `bas_dd <= ?` 로 들어가면 `'20260821' <= '2026-08-21'` 이 거짓이라
    결과가 0행이 되고, 받은 쪽은 "그 구간에 자료가 없구나" 로 읽는다. 예외는 안 난다.
    """
    여덟자리 = supply.index_series(as_of="2026-08-26", end="20260821")
    ISO = supply.index_series(as_of="2026-08-26", end="2026-08-21")

    assert list(ISO["date"]) == ["2026-08-21"], "ISO 로 준 end 가 0행을 만들었다"
    assert list(ISO["date"]) == list(여덟자리["date"]), "표기에 따라 답이 달라졌다"


def test_언제부터_알_수_있었는지를_행에_붙여_준다(채워진저장소):
    """누수를 의심할 때 눈으로 확인할 수 있어야 한다."""
    df = supply.index_series(as_of="2026-08-26", with_known_at=True)

    assert (df["known_at"] > df["date"]).all(), "알게 된 시각이 거래일보다 앞설 수 없다"
    assert df["known_at"].iloc[-1].startswith("2026-08-26T00:00:00")


# ── 표의 모양이 계약이다 ────────────────────────────────────────────────────

def test_빈_결과에도_칸이_남는다(채워진저장소):
    """🔴 칸 없는 빈 표를 주면 `df["close"]` 가 KeyError 로 터진다.

    부르는 쪽은 그걸 "자료가 없다" 가 아니라 "코드가 깨졌다" 로 읽는다.
    """
    df = supply.index_series(as_of="2010-01-01")

    assert len(df) == 0
    assert "close" in df.columns
    assert list(df.columns) == list(supply.INDEX_COLUMNS)


def test_칸_목록이_저장소_반환을_덮는다(채워진저장소):
    """`INDEX_COLUMNS` 가 실제 반환보다 좁으면 빈 표와 찬 표의 칸이 달라진다."""
    df = supply.index_series(as_of="2026-08-26")

    assert list(df.columns) == list(supply.INDEX_COLUMNS), (
        "저장소가 칸을 바꿨다. supply/market.py 의 INDEX_COLUMNS 도 맞춰야 한다.\n"
        f"  실제: {list(df.columns)}"
    )


def test_bas_dd_를_되살려_준다(채워진저장소):
    """저장소는 `date`(YYYY-MM-DD)만 주는데 달력·플래그는 `YYYYMMDD` 로 말한다.

    부르는 쪽마다 `.str.replace('-','')` 를 쓰게 두면 언젠가 한 곳에서 빠뜨린다.
    """
    df = supply.index_series(as_of="2026-08-26")

    assert list(df["bas_dd"]) == ["20260821", "20260824", "20260825"]
