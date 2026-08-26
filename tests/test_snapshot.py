"""F-02 — 스냅샷 박제 + SHA-256 테스트.

**무엇을 지키려는 테스트인가.** 이 프로젝트의 결론은 *"어떤 자료로 냈는가"* 와 떼어 낼 수
없다. 지문이 흔들리면 재현이 불가능해지고, 그러면 리포트의 모든 숫자가 검증 불가능한
주장이 된다.

    수용 기준 (docs/요구사항.md F-02)
    - data/snapshots/kospi200_YYYYMMDD.parquet 과 snapshot_meta.json 이 생긴다
    - 메타에 행 수·시작·종료·SHA-256·수집 시각 KST 가 있다
    - **다른 날 두 번 돌려도 두 산출 JSON 의 스냅샷 해시가 같다**

DB 를 건드리지 않는다 — `krx_index.series` 를 가짜로 갈아 끼워 빠르고 결정적으로 돈다.
"""

from __future__ import annotations

import json

import pytest

from ingest import snapshot as snap

# ── 픽스처 ─────────────────────────────────────────────────────────────────

def 샘플행(n: int = 5, 시작가: float = 100.0) -> list:
    """`series()` 가 돌려주는 모양 그대로의 가짜 행."""
    return [
        {
            "date": f"2020-01-{i + 1:02d}",
            "index_name": "코스피 200",
            "index_class": "KOSPI",
            "open": 시작가 + i, "high": 시작가 + i + 1.5,
            "low": 시작가 + i - 1.5, "close": 시작가 + i + 0.25,
            "change": 0.25, "change_rate": 0.24,
            "volume": 1_000 + i, "value": 2_000_000 + i, "market_cap": 5_000_000 + i,
        }
        for i in range(n)
    ]


@pytest.fixture
def 격리(tmp_path, monkeypatch):
    """스냅샷 경로를 tmp 로 돌리고, DB 대신 가짜 행을 쓰게 만든다."""
    monkeypatch.setattr(snap, "SNAPSHOT_DIR", tmp_path / "snapshots")
    monkeypatch.setattr(snap, "META_PATH", tmp_path / "snapshots" / "snapshot_meta.json")
    monkeypatch.setattr(snap, "ROOT", tmp_path)

    상태 = {"rows": 샘플행()}

    def 가짜_series(index_name="코스피 200", days=None, start=None, end=None):
        return list(상태["rows"])

    monkeypatch.setattr(snap.krx_index, "series", 가짜_series)
    return 상태


# ── 정규 직렬화 · 해시 ─────────────────────────────────────────────────────

def test_행_순서가_달라도_해시는_같다():
    """자료가 같으면 지문이 같아야 한다. 정렬은 우리가 책임진다."""
    rows = 샘플행()
    assert snap.content_sha256(rows) == snap.content_sha256(list(reversed(rows)))


def test_값이_하나만_달라도_해시가_바뀐다():
    rows = 샘플행()
    바뀐 = [dict(r) for r in rows]
    바뀐[2]["close"] = 바뀐[2]["close"] + 0.01
    assert snap.content_sha256(rows) != snap.content_sha256(바뀐)


def test_결측과_빈_문자열이_구별된다():
    """`None` 과 `""` 를 같은 문자열로 두면 서로 다른 자료가 같은 지문을 갖는다."""
    a = 샘플행(1)
    b = [dict(a[0])]
    a[0]["index_class"] = None
    b[0]["index_class"] = ""
    assert snap.content_sha256(a) != snap.content_sha256(b)


def test_열_순서를_바꾸면_해시가_바뀐다(monkeypatch):
    """헤더를 해시에 넣는 이유. 열 배치가 달라진 자료를 같다고 말하면 안 된다."""
    rows = 샘플행()
    원래 = snap.content_sha256(rows)
    뒤바뀐 = tuple(reversed(snap.CANONICAL_COLUMNS))
    monkeypatch.setattr(snap, "CANONICAL_COLUMNS", 뒤바뀐)
    assert snap.content_sha256(rows) != 원래


def test_실수는_반올림_없이_직렬화된다():
    """f-string 반올림을 쓰면 다른 값이 같은 문자열이 되어 지문이 뭉개진다."""
    assert snap._cell(1054.01) != snap._cell(1054.010000001)
    # repr 은 되읽으면 정확히 같은 값이 나오는 최단 표기다
    assert float(snap._cell(1054.01)) == 1054.01


# ── 박제 ───────────────────────────────────────────────────────────────────

def test_박제하면_parquet_과_메타가_생긴다(격리):
    항목 = snap.freeze(end="20200105")
    assert (snap.SNAPSHOT_DIR / "kospi200_20200105.parquet").exists()
    assert snap.META_PATH.exists()
    assert 항목["path"].endswith("kospi200_20200105.parquet")


def test_메타에_수용기준_항목이_전부_있다(격리):
    snap.freeze()
    meta = json.loads(snap.META_PATH.read_text(encoding="utf-8"))
    항목 = meta["snapshots"]["kospi200_20200105"]
    for 칸 in ("rows", "start", "end", "sha256", "frozen_at_kst"):
        assert 칸 in 항목, f"메타에 {칸} 이 없다 (F-02 수용 기준)"
    assert 항목["rows"] == 5
    assert 항목["start"] == "2020-01-01"
    assert 항목["end"] == "2020-01-05"


def test_수집_시각이_KST_다(격리):
    항목 = snap.freeze()
    assert 항목["frozen_at_kst"].endswith("+09:00")


def test_파일_이름은_오늘이_아니라_자료_종료일을_쓴다(격리):
    """오늘 날짜를 쓰면 같은 자료를 다른 날 뜨기만 해도 파일이 갈라진다 — 박제의 반대다."""
    항목 = snap.freeze()
    assert "20200105" in 항목["path"]


# ── ★ 수용 기준의 핵심: 다른 날 두 번 돌려도 해시가 같은가 ────────────────

def test_두_번_돌려도_해시가_같다(격리):
    첫번째 = snap.freeze()
    두번째 = snap.freeze()
    assert 첫번째["sha256"] == 두번째["sha256"]


def test_다른_날_돌려도_해시가_같다(격리, monkeypatch):
    """'수집 시각'만 다르고 자료는 같은 상황을 흉내 낸다."""
    from datetime import datetime, timedelta

    첫번째 = snap.freeze()
    첫_json = json.loads(snap.META_PATH.read_text(encoding="utf-8"))

    실제_now = datetime.now

    class 다른날(datetime):
        @classmethod
        def now(cls, tz=None):
            return 실제_now(tz) + timedelta(days=3)

    monkeypatch.setattr(snap, "datetime", 다른날)
    두번째 = snap.freeze()
    둘_json = json.loads(snap.META_PATH.read_text(encoding="utf-8"))

    assert 첫번째["sha256"] == 두번째["sha256"]
    assert (첫_json["snapshots"]["kospi200_20200105"]["sha256"]
            == 둘_json["snapshots"]["kospi200_20200105"]["sha256"])


def test_처음_박제한_시각은_다시_돌려도_안_바뀐다(격리):
    """그게 '박제'다. 다시 돌린 사실은 last_verified_at_kst 에 남는다."""
    첫번째 = snap.freeze()
    두번째 = snap.freeze()
    assert 첫번째["frozen_at_kst"] == 두번째["frozen_at_kst"]
    assert "last_verified_at_kst" in 두번째


# ── 자료가 달라졌을 때 ─────────────────────────────────────────────────────

def test_내용이_달라지면_조용히_덮어쓰지_않는다(격리):
    snap.freeze()
    바뀐 = [dict(r) for r in 격리["rows"]]
    바뀐[0]["close"] = 999.99
    격리["rows"] = 바뀐

    with pytest.raises(snap.SnapshotConflictError) as err:
        snap.freeze()
    메시지 = str(err.value)
    assert "해결" in 메시지          # 막다른 길로 두지 않는다
    assert "--force" in 메시지


def test_force_면_덮어쓴다(격리):
    첫번째 = snap.freeze()
    바뀐 = [dict(r) for r in 격리["rows"]]
    바뀐[0]["close"] = 999.99
    격리["rows"] = 바뀐

    두번째 = snap.freeze(force=True)
    assert 두번째["sha256"] != 첫번째["sha256"]


def test_자료가_없으면_무엇을_하라고_말해_준다(격리):
    격리["rows"] = []
    with pytest.raises(ValueError, match="fetch_index"):
        snap.freeze()


# ── parquet 왕복 ───────────────────────────────────────────────────────────

def test_parquet_왕복_후에도_지문이_맞는다(격리):
    """쓰고 되읽었을 때 지문이 달라지면 박제가 성립하지 않는다."""
    snap.freeze()
    결과 = snap.verify()
    assert len(결과) == 1
    assert 결과[0]["ok"], 결과[0]


def test_parquet_이_사라지면_검증이_실패한다(격리):
    snap.freeze()
    (snap.SNAPSHOT_DIR / "kospi200_20200105.parquet").unlink()
    결과 = snap.verify()
    assert not 결과[0]["ok"]
    assert "파일이 없다" in 결과[0]["reason"]


def test_parquet_이_변조되면_검증이_잡아낸다(격리):
    """지문의 존재 이유. 파일이 바뀌었는데 메타가 그대로면 알아채야 한다."""
    import pandas as pd

    snap.freeze()
    경로 = snap.SNAPSHOT_DIR / "kospi200_20200105.parquet"
    df = pd.read_parquet(경로, engine="pyarrow")
    df.loc[0, "close"] = 12345.6
    df.to_parquet(경로, engine="pyarrow", compression="snappy", index=False)

    결과 = snap.verify()
    assert not 결과[0]["ok"]
    assert 결과[0]["reason"] == "내용 해시가 다르다"
