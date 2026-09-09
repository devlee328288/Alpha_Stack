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
import json
import sys
import types
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


# ══════════════════════════════════════════════════════════════════════════
# 낡은 스냅샷 — 거짓 🔴 은 거짓 ✅ 만큼 나쁘다
#
# `verify_manifest` 는 스냅샷을 **자기 자신의** MANIFEST 와 대조하므로 낡은 배포본도
# 자기끼리 일관돼 통과한다. 그 구멍을 `assert_snapshot_is_current` 가 막는다.
# ══════════════════════════════════════════════════════════════════════════
def _snap(tmp_path: Path, generated_at: str) -> Path:
    snap = tmp_path / "snap"
    snap.mkdir()
    (snap / "MANIFEST.json").write_text(
        json.dumps({"generated_at": generated_at, "files": []}), encoding="utf-8")
    return snap


@pytest.fixture
def 서버가(monkeypatch, tmp_path):
    """서버 MANIFEST 를 흉내 낸다. 받아 오는 자리만 갈아 끼운다."""
    def _설정(generated_at: str):
        서버 = tmp_path / "server_MANIFEST.json"
        서버.write_text(json.dumps({"generated_at": generated_at, "files": []}),
                        encoding="utf-8")
        monkeypatch.setitem(sys.modules, "huggingface_hub",
                            types.SimpleNamespace(
                                hf_hub_download=lambda **kw: str(서버)))
        monkeypatch.setitem(sys.modules, "ingest.clients",
                            types.SimpleNamespace(
                                hf_data=types.SimpleNamespace(
                                    load_hf_key=lambda: ("tok", "테스트"))))
    return _설정


def test_스냅샷이_서버와_같으면_지나간다(tmp_path, 서버가):
    서버가("2026-09-07T04:58:00Z")
    snap = _snap(tmp_path, "2026-09-07T04:58:00Z")
    V.assert_snapshot_is_current("repo", snap)      # 예외가 없으면 통과다


def test_스냅샷이_서버보다_낡으면_멈춘다(tmp_path, 서버가):
    """2026-09-08 에 실제로 겪은 일 — 같은 날 두 번째 배포를 못 받은 사본이었다."""
    서버가("2026-09-07T04:58:00Z")
    snap = _snap(tmp_path, "2026-09-07T01:36:00Z")
    with pytest.raises(SystemExit) as e:
        V.assert_snapshot_is_current("repo", snap)
    assert e.value.code == 2


def test_스냅샷에_MANIFEST_가_없으면_멈춘다(tmp_path):
    with pytest.raises(SystemExit, match="MANIFEST"):
        V.assert_snapshot_is_current("repo", tmp_path / "없는곳")


def test_서버를_못_물어보면_막지_않고_알리기만_한다(tmp_path, monkeypatch, capsys):
    """토큰이 없다고 판정을 통째로 못 하게 만들면, 쓰던 사람이 갈 곳이 없어진다."""
    monkeypatch.setitem(sys.modules, "huggingface_hub",
                        types.SimpleNamespace(hf_hub_download=lambda **kw: ""))
    monkeypatch.setitem(sys.modules, "ingest.clients",
                        types.SimpleNamespace(
                            hf_data=types.SimpleNamespace(
                                load_hf_key=lambda: (None, ".env 없음"))))
    snap = _snap(tmp_path, "2026-09-07T04:58:00Z")
    V.assert_snapshot_is_current("repo", snap)      # 멈추지 않는다
    assert "확인하지 못했다" in capsys.readouterr().out


# ══════════════════════════════════════════════════════════════════════════
# 판정은 반출과 같은 입력 위에서 — 연도로 끊으면 경계에서 답이 달라진다
#
# 2026-09-09 실제: 036220 은 2016-05-04 인포피아(상장폐지)의 마지막 행 뒤에
# 2024-03-13 오상헬스케어(신규상장)가 같은 코드로 다시 왔다. "전년도 12월부터" 패드로는
# 그 전일이 안 보여 판정기만 NaN 이 됐고, 반출은 +738.57% 로 `is_adj_suspect` 를 켰다.
# ══════════════════════════════════════════════════════════════════════════
def _daily_price_db(tmp_path: Path, rows):
    import sqlite3
    db = tmp_path / "q.db"
    with sqlite3.connect(db) as c:
        c.execute("CREATE TABLE daily_price (bas_dd TEXT, code TEXT, close REAL, "
                  "adj_close REAL, change_rate REAL)")
        c.executemany("INSERT INTO daily_price VALUES (?, ?, ?, ?, ?)", rows)
    return db


코드_재사용 = [
    ("20160503", "036220", 1500.0, 1500.0, -1.90),
    ("20160504", "036220", 1400.0, 1400.0, -6.67),      # 인포피아 마지막 행
    ("20240313", "036220", 11740.0, 11740.0, 46.75),    # 오상헬스케어 첫 행 — 같은 코드
    ("20240314", "036220", 12000.0, 12000.0, 2.21),
]


def test_품질_판정은_전_구간에서_한_번_만든다(tmp_path, monkeypatch):
    """8년 공백 뒤 행도 반출과 같은 값(전 구간 직전 행 대비)이 나와야 한다."""
    monkeypatch.setenv("KRX_DB_PATH", str(_daily_price_db(tmp_path, 코드_재사용)))
    with V.ro_connect() as conn:
        q = V._quality_table(conn, "20240831")
    행 = q.set_index(["bas_dd", "code"]).loc[("20240313", "036220")]
    assert 행["adj_return_1d"] == pytest.approx((11740.0 / 1400.0 - 1.0) * 100.0)
    assert bool(행["is_adj_suspect"]), "8년 전 값 대비 +738% 는 KRX 등락률과 어긋난다"
    assert not bool(행["is_extreme_return"]), "의심이면 극단으로 치지 않는다"
    assert len(q) == len(코드_재사용), "입력 행이 하나도 빠지거나 늘지 않는다"


def test_전년도_12월_패드로는_코드_재사용의_전일이_안_보인다(tmp_path, monkeypatch):
    """옛 방식이 왜 틀렸는지를 남긴다 — 같은 함수에 잘린 입력을 주면 NaN 이 된다."""
    monkeypatch.setenv("KRX_DB_PATH", str(_daily_price_db(tmp_path, 코드_재사용)))
    with V.ro_connect() as conn:
        잘린 = pd.read_sql_query(
            "SELECT bas_dd, code, close, adj_close, change_rate FROM daily_price "
            "WHERE bas_dd BETWEEN ? AND ?", conn, params=("20231201", "20241231"))
    flags = V.flag_adjustment_quality(잘린)
    첫행 = flags.loc[잘린["bas_dd"].eq("20240313")].iloc[0]
    assert np.isnan(첫행["adj_return_1d"])
    assert not bool(첫행["is_adj_suspect"])


def test_판정_표에_없는_행이_있으면_붙이지_않고_멈춘다(tmp_path, monkeypatch):
    """조용히 False 로 채우면 반출과 다른 표본을 같다고 판정하게 된다."""
    monkeypatch.setenv("KRX_DB_PATH", str(_daily_price_db(tmp_path, 코드_재사용)))
    with V.ro_connect() as conn:
        q = V._quality_table(conn, "20240831")
    db = pd.DataFrame({"bas_dd": ["20240313", "20240399"], "code": ["036220", "036220"],
                       "adj_close": [11740.0, 1.0]})
    monkeypatch.setattr(V, "attach_industry", lambda frame, *, as_of: frame)
    ca = pd.DataFrame(columns=["bas_dd", "code", *V.CORPORATE_ACTION_COLUMNS])
    with pytest.raises(RuntimeError, match="반출과 같은 입력"):
        V._attach_export_derived(db, None, ca, q)
