"""배포본 대조기의 판정 기준을 못박는다 — 무엇을 "달라졌다" 고 부를 것인가.

## 왜 이 테스트가 필요한가

`verify_hf_dataset` 이 답하는 질문은 "HF 에 올린 것을 다시 올려야 하는가" 다. 그런데
**float 를 CSV 로 적었다 읽으면 끝자리가 흔들린다.** 그 흔들림까지 "달라졌다" 로 세면
판정이 매번 참이 되어 아무것도 못 거른다. 거꾸로 여유를 크게 두면 진짜 변화를 놓친다.

그래서 이 파일은 **경계를 고정한다.**

    ① 1 ULP 차이      → 표기의 한계다 (재배포 안 한다)
    ② 눈에 띄는 차이  → 자료의 변화다 (재배포 한다)
    ③ 결측이 엇갈리면 → 크기가 작아도 자료의 변화다
    ④ 빈 문자열↔결측  → CSV 가 빈 칸을 결측으로 읽은 것이다 (표기의 한계)

③ 이 따로 있는 이유는, 값이 생기거나 사라지는 것은 **크기로 잴 수 없는 종류의 변화**
이기 때문이다. `NaN` 과 `0.0` 은 몇 ULP 떨어져 있다고 말할 수 없다.

④ 는 실제로 겪은 일이다. DB 의 소속부는 KOSPI 종목에서 빈 문자열인데, CSV 로 나가면
`NaN` 이 되어 32,226행이 어긋난 것처럼 보였다. 자료는 그대로였다.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def _load():
    """스크립트를 모듈로 읽는다 — `scripts/` 는 패키지가 아니다."""
    spec = importlib.util.spec_from_file_location(
        "verify_hf_dataset", ROOT / "scripts" / "verify_hf_dataset.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


V = _load()


# ══════════════════════════════════════════════════════════════════════════
# compare_column — 표기의 한계와 자료의 변화를 가르는 자리
# ══════════════════════════════════════════════════════════════════════════
def test_같은_값이면_차이가_없다고_말한다():
    a = pd.Series([1.0, 2.0, 3.0])
    assert V.compare_column(a, a.copy()) is None


def test_dtype이_달라도_값이_같으면_같다():
    """DB 는 INTEGER 로, parquet 은 float 로 돌려줄 수 있다. 그건 차이가 아니다."""
    assert V.compare_column(pd.Series([1, 2, 3]), pd.Series([1.0, 2.0, 3.0])) is None


def test_양쪽_다_결측이면_같다():
    """`==` 만 쓰면 NaN != NaN 이라 멀쩡한 결측이 전부 차이로 잡힌다."""
    a = pd.Series([1.0, np.nan, 3.0])
    assert V.compare_column(a, a.copy()) is None


def test_1_ULP_차이는_표기의_한계로_본다():
    """CSV 왕복에서 실제로 나온 크기다 — 시가총액 1.8경에서 0.25 가 1 ULP 였다."""
    원본 = 1807866930879163.0
    a = pd.Series([원본])
    b = pd.Series([np.nextafter(원본, np.inf)])       # 정확히 1 ULP 옆
    d = V.compare_column(a, b)
    assert d is not None and d["행"] == 1
    assert d["표기한계"] is True
    assert d["최대ulp"] == pytest.approx(1.0)


def test_허용치를_넘는_차이는_자료의_변화로_본다():
    원본 = 100.0
    벗어난 = 원본 * (1 + 1e-9)                        # ULP 로 수만 배
    d = V.compare_column(pd.Series([원본]), pd.Series([벗어난]))
    assert d["표기한계"] is False
    assert d["최대ulp"] > V.ULP_TOLERANCE


def test_결측이_엇갈리면_크기와_무관하게_변화다():
    """값이 생기거나 사라지는 것은 ULP 로 잴 수 없다. 항상 실제 차이로 센다."""
    d = V.compare_column(pd.Series([1.0, 2.0]), pd.Series([1.0, np.nan]))
    assert d["결측엇갈림"] == 1
    assert d["표기한계"] is False


def test_빈_문자열이_결측이_된_것은_표기의_한계다():
    """CSV 는 빈 칸을 결측으로 읽는다. DB 의 빈 소속부가 이렇게 어긋났다."""
    d = V.compare_column(pd.Series(["", "우량기업부"]),
                         pd.Series([None, "우량기업부"]))
    assert d["행"] == 1
    assert d["표기한계"] is True


def test_문자열_값이_실제로_다르면_변화다():
    d = V.compare_column(pd.Series(["우량기업부"]), pd.Series(["벤처기업부"]))
    assert d["표기한계"] is False


# ══════════════════════════════════════════════════════════════════════════
# compare_frames — 표 단위 판정
# ══════════════════════════════════════════════════════════════════════════
def _frame(**cols) -> pd.DataFrame:
    return pd.DataFrame(cols)


def test_행_순서가_달라도_키로_정렬해_비교한다():
    """`full/*.parquet` 은 ORDER BY 없이 뽑는다. 순서는 차이가 아니다."""
    a = _frame(bas_dd=["20200102", "20200103"], close=[100.0, 200.0])
    b = _frame(bas_dd=["20200103", "20200102"], close=[200.0, 100.0])
    assert V.compare_frames(a, b, ["bas_dd"], "순서만 다름")["같다"] is True


def test_행_수가_다르면_바로_다르다고_한다():
    a = _frame(bas_dd=["20200102"], close=[100.0])
    b = _frame(bas_dd=["20200102", "20200103"], close=[100.0, 200.0])
    r = V.compare_frames(a, b, ["bas_dd"], "행 수")
    assert r["같다"] is False and r["이유"] == "행 수"


def test_칸_구성이_다르면_다르다고_한다():
    a = _frame(bas_dd=["20200102"], close=[100.0])
    b = _frame(bas_dd=["20200102"], close=[100.0], extra=[1])
    r = V.compare_frames(a, b, ["bas_dd"], "칸 구성")
    assert r["같다"] is False and r["이유"] == "칸 구성"


def test_only_common_이면_한쪽에만_있는_칸을_눈감아_준다():
    """반출본은 DB 에 없는 `date` 칸을 하나 더 붙인다. 그건 어긋남이 아니다."""
    a = _frame(bas_dd=["20200102"], close=[100.0], date=["2020-01-02"])
    b = _frame(bas_dd=["20200102"], close=[100.0])
    assert V.compare_frames(a, b, ["bas_dd"], "date 추가",
                            only_common=True)["같다"] is True


def test_표기_차이만_있으면_같다고_판정한다():
    """재배포 여부를 가르는 자리다 — 1 ULP 때문에 다시 올리지는 않는다."""
    원본 = 1807866930879163.0
    a = _frame(bas_dd=["20200102"], market_cap=[원본])
    b = _frame(bas_dd=["20200102"], market_cap=[np.nextafter(원본, np.inf)])
    r = V.compare_frames(a, b, ["bas_dd"], "표기만")
    assert r["같다"] is True
    assert r["표기차"] and not r["실제차"]


def test_실제_차이가_있으면_다르다고_판정한다():
    a = _frame(bas_dd=["20200102"], close=[100.0])
    b = _frame(bas_dd=["20200102"], close=[101.0])
    r = V.compare_frames(a, b, ["bas_dd"], "실제 차이")
    assert r["같다"] is False
    assert r["실제차"][0]["칸"] == "close"


# ══════════════════════════════════════════════════════════════════════════
# 읽기 전용 — 이 스크립트가 공유 DB 를 건드리지 않는다는 약속
# ══════════════════════════════════════════════════════════════════════════
def test_연결은_읽기_전용이라_쓰기가_거부된다(tmp_path, monkeypatch):
    """확인만 하는 작업이 공유 DB 를 바꾸면 안 된다. 약속을 코드로 못박는다."""
    db = tmp_path / "t.db"
    import sqlite3
    with sqlite3.connect(db) as c:
        c.execute("CREATE TABLE t (a INTEGER)")
        c.execute("INSERT INTO t VALUES (1)")

    monkeypatch.setenv("KRX_DB_PATH", str(db))
    with V.ro_connect() as conn:
        assert conn.execute("SELECT a FROM t").fetchone()[0] == 1
        with pytest.raises(sqlite3.OperationalError, match="readonly"):
            conn.execute("INSERT INTO t VALUES (2)")
