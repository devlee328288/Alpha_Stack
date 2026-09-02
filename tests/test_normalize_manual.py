"""`scripts/normalize_manual.py` — 손으로 받은 자료를 UTF-8 로 맞추는 도구의 시험대.

가장 중요한 두 가지를 본다.

  ① **사람이 붙인 이름을 덮어쓰지 않는가** — 스크립트가 아는 것은 값뿐이고,
     받은 사람이 아는 것이 더 많다. 적게 아는 쪽이 덮어쓰면 정보가 사라진다.
  ② **이미 변환한 것을 건너뛰는가** — 두 번 돌려도 같은 결과여야 한다.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Dict, List

import pytest

_SPEC = importlib.util.spec_from_file_location(
    "normalize_manual", Path(__file__).resolve().parents[1] / "scripts" / "normalize_manual.py")
normalize_manual = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = normalize_manual
_SPEC.loader.exec_module(normalize_manual)


# ── 합성 자료 ───────────────────────────────────────────────────────────────
# 투자자 두 명(가·나)의 닷새치. 순매수 = 매수 − 매도 가 성립하도록 만든다.
DAYS = ["2024/01/02", "2024/01/03", "2024/01/04", "2024/01/05", "2024/01/08"]
매도량 = [[100, 200], [150, 250], [130, 210], [170, 190], [160, 230]]
매수량 = [[120, 180], [140, 300], [155, 205], [160, 240], [175, 215]]


def _rows(values: List[List[int]], scale: int = 1) -> List[List[str]]:
    return [[day] + [str(v * scale) for v in row] for day, row in zip(DAYS, values, strict=True)]


def _net(a: List[List[int]], b: List[List[int]], scale: int = 1) -> List[List[str]]:
    diff = [[y - x for x, y in zip(ra, rb, strict=True)] for ra, rb in zip(a, b, strict=True)]
    return _rows(diff, scale)


@pytest.fixture()
def 여섯갈래() -> Dict[Path, List[List[str]]]:
    """거래량 3개 + 거래대금 3개. 대금은 규모가 훨씬 크다."""
    return {
        Path("data_0001_x.csv"): _rows(매도량),
        Path("data_0002_x.csv"): _rows(매수량),
        Path("data_0003_x.csv"): _net(매도량, 매수량),
        Path("data_0004_x.csv"): _rows(매도량, scale=1000),
        Path("data_0005_x.csv"): _rows(매수량, scale=1000),
        Path("data_0006_x.csv"): _net(매도량, 매수량, scale=1000),
    }


# ── 값으로 정체를 알아내는가 ────────────────────────────────────────────────

def test_여섯갈래를_값만_보고_가른다(여섯갈래):
    """머리글이 같아 파일만 봐서는 구별되지 않는다. 검산이 유일한 근거다."""
    labels = normalize_manual.classify_group(여섯갈래)

    assert labels[Path("data_0001_x.csv")] == "거래량_매도"
    assert labels[Path("data_0002_x.csv")] == "거래량_매수"
    assert labels[Path("data_0003_x.csv")] == "거래량_순매수"
    assert labels[Path("data_0004_x.csv")] == "거래대금_매도"
    assert labels[Path("data_0005_x.csv")] == "거래대금_매수"
    assert labels[Path("data_0006_x.csv")] == "거래대금_순매수"


def test_매도와_매수를_바꿔_넣으면_판정도_바뀐다(여섯갈래):
    """검산이 항등식이면 무엇을 넣어도 같은 답이 나온다 — 그렇지 않은지 본다."""
    바꾼것 = dict(여섯갈래)
    바꾼것[Path("data_0001_x.csv")] = 여섯갈래[Path("data_0002_x.csv")]
    바꾼것[Path("data_0002_x.csv")] = 여섯갈래[Path("data_0001_x.csv")]

    labels = normalize_manual.classify_group(바꾼것)

    # 내용을 맞바꿨으니 이름도 맞바뀌어야 한다
    assert labels[Path("data_0001_x.csv")] == "거래량_매수"
    assert labels[Path("data_0002_x.csv")] == "거래량_매도"


def test_검산이_안_서면_순매수만_붙인다(여섯갈래):
    """근거가 없으면 이름을 붙이지 않는다 — 모르는 것을 아는 척하지 않는다."""
    깨진것 = dict(여섯갈래)
    # 매수 쪽 값을 흐트러뜨려 순매수와 아귀가 맞지 않게 만든다
    깨진것[Path("data_0002_x.csv")] = _rows([[9, 9], [9, 9], [9, 9], [9, 9], [9, 9]])

    labels = normalize_manual.classify_group(깨진것)

    assert labels[Path("data_0003_x.csv")] == "거래량_순매수"      # 음수라 이건 확실하다
    assert labels[Path("data_0001_x.csv")] == ""                   # 매도·매수는 가릴 수 없다
    assert labels[Path("data_0002_x.csv")] == ""


def test_파일이_여섯이_아니면_이름을_안_붙인다():
    """묶음이 온전하지 않으면 판별 근거가 없다."""
    labels = normalize_manual.classify_group({
        Path("a.csv"): _rows(매도량),
        Path("b.csv"): _rows(매수량),
    })

    assert set(labels.values()) == {""}


# ── 사람이 붙인 이름을 지키는가 ─────────────────────────────────────────────

def test_KRX_일련번호만_바꿀_대상이다():
    """`data_4907_20260902` 는 뜻이 없다. 그 무늬만 손댄다."""
    바꿈 = normalize_manual.RAW_NAME
    assert 바꿈.match("data_4907_20260902")
    assert 바꿈.match("data_5021_20260902")


def test_사람이_붙인_이름은_건드리지_않는다():
    """직접 이름을 정해 넣은 파일은 그대로 둔다 — 받은 사람이 더 많이 안다."""
    바꿈 = normalize_manual.RAW_NAME
    for 이름 in ("data_투자자별_순매수상위종목_외국인_20240901-20260901",
                 "수출입 총괄_20260901 (1)",
                 "20260901_전종목시세",
                 "data_4907",                    # 숫자 한 덩이뿐이면 무늬가 아니다
                 "data_4907_20260902_수정본"):
        assert 바꿈.match(이름) is None, 이름


# ── 인코딩 ──────────────────────────────────────────────────────────────────

def test_cp949_한글을_읽어낸다(tmp_path):
    path = tmp_path / "a.csv"
    path.write_bytes("일자,금융투자\n2024/01/02,100\n".encode("cp949"))

    text, enc = normalize_manual.decode(path)

    assert enc == "cp949"
    assert "금융투자" in text


def test_BOM_이_붙은_파일은_utf_8_sig_로_읽는다(tmp_path):
    """utf-8 로 먼저 시도하면 첫 칸 이름에 BOM 이 붙은 채 '성공' 한다 — 조용히 틀린다."""
    path = tmp_path / "b.csv"
    path.write_bytes("﻿기간,수출\n2026,100\n".encode("utf-8"))

    text, enc = normalize_manual.decode(path)

    assert enc == "utf-8-sig"
    assert text.startswith("기간")          # BOM 이 남아 있지 않다


def test_어떤_바이트든_대개_읽힌다(tmp_path):
    """⚠️ `decode` 는 사실상 실패하지 않는다 — 이걸 알고 써야 한다.

    `cp949` 와 `utf-16` 은 거의 모든 바이트열을 받아들인다. 아래 바이트는 UTF-16 의
    BOM(`\\xff\\xfe`)으로 시작해 그대로 디코드된다.

    그래서 **"예외가 안 났으니 인코딩을 맞게 골랐다"고 볼 수 없다.** 순서가 방어선이다 —
    UTF-8 은 엄격해서 아무 바이트나 통과시키지 않으므로 먼저 시도하고, 거기서 걸러진
    것만 cp949 로 내려간다.
    """
    path = tmp_path / "c.csv"
    path.write_bytes(b"\xff\xfe\x00\x00\xff\xff\xfe\xfd")

    _, enc = normalize_manual.decode(path)

    assert enc in normalize_manual.ENCODINGS


def test_후보가_없으면_무엇을_해야_하는지_알려준다(tmp_path, monkeypatch):
    """막다른 길로 만들지 않는다 — 예외만 던지고 끝내지 않는다."""
    monkeypatch.setattr(normalize_manual, "ENCODINGS", ("ascii",))
    path = tmp_path / "d.csv"
    path.write_bytes("한글".encode("cp949"))

    with pytest.raises(ValueError) as err:
        normalize_manual.decode(path)

    assert "할 일" in str(err.value)


# ── 사본 견주기 ─────────────────────────────────────────────────────────────

def test_표기만_다르면_같은_값으로_본다():
    a = [["기간", "금액"], ["총계", "435585248"]]
    b = [["기간", "금액"], ["총계", "4.35585248E8"]]

    assert normalize_manual.sibling_diff(a, b) == []


def test_값이_다른_칸만_짚어낸다():
    a = [["기간", "중량", "금액"], ["총계", "2250940321.0", "7195826806"]]
    b = [["기간", "중량", "금액"], ["총계", "2250940320998", "7195826806"]]

    assert normalize_manual.sibling_diff(a, b) == ["중량"]


def test_머리글이_다르면_견주지_않는다():
    a = [["기간", "금액"], ["총계", "1"]]
    b = [["일자", "금액"], ["총계", "1"]]

    assert normalize_manual.sibling_diff(a, b) is None
