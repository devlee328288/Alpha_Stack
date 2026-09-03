"""반출 폴더의 칸별 통계가 실제 파일과 맞는지 — 손으로 센 값과 대조한다.

여기서 조심할 것이 하나 있다. "프로파일러가 센 결측 수 == 프로파일러가 센 결측 수"
같은 검사는 항등식이라 아무것도 못 잡는다. 그래서 이 파일의 기대값은 **테스트가 만든
표에 무엇을 넣었는지 보고 손으로 적은 수**다. 계산으로 유도하지 않는다.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from common.export_profile import (
    RARE_VALUE_LIMIT,
    SAMPLE_ROWS,
    file_profile,
    load_profile,
    profile_export,
    write_profile,
)

# ══════════════════════════════════════════════════════════════════════════
# 표본 — 무엇을 넣었는지 여기서 다 보인다
# ══════════════════════════════════════════════════════════════════════════

#: 6행짜리 시세 표. 손으로 센 기대값은 아래 테스트에 그대로 적는다.
#:
#:   bas_dd   5일 (20240102 가 두 번)
#:   code     2종 (000020 이 4행 · 0001B0 이 2행)   ← 영문이 낀 코드가 숫자로 안 읽히는지
#:   adj_open 6행 중 2행이 결측                      ← 거래정지일
#:   market   1종
def 시세표() -> pd.DataFrame:
    return pd.DataFrame({
        "bas_dd": ["20240102", "20240102", "20240103", "20240104", "20240105", "20240108"],
        "code": ["000020", "0001B0", "000020", "000020", "000020", "0001B0"],
        "name": ["동화약품", "테스트우", "동화약품", "동화약품", "동화약품", "테스트우"],
        "market": ["KOSPI"] * 6,
        "open": [7540, 1000, 0, 7600, 7700, 0],
        "close": [7800, 1100, 7550, 7650, 7750, 1050],
        "adj_open": [754.0, 1000.0, None, 760.0, 770.0, None],
        "adj_close": [780.0, 1100.0, 755.0, 765.0, 775.0, 1050.0],
        "adj_source": ["fdr", "chain", "fdr", "fdr", "fdr", "chain"],
    })


@pytest.fixture()
def 반출폴더(tmp_path: Path) -> Path:
    """MANIFEST 와 파일 두 개(csv·parquet)를 갖춘 최소 반출 폴더."""
    small, full = tmp_path / "small", tmp_path / "full"
    small.mkdir()
    full.mkdir()

    표 = 시세표()
    표.to_csv(small / "price.csv", index=False, encoding="utf-8-sig")
    표.to_parquet(full / "price.parquet", index=False, compression="zstd")

    manifest = {
        "generated_at": "2026-09-03T10:00:00+09:00",
        "dev_end": "20240831",
        "holdout_start": "20240901",
        "files": [
            {"path": "price.csv", "rows": 6},
            {"path": "price.parquet", "rows": 6},
        ],
    }
    (tmp_path / "MANIFEST.json").write_text(
        json.dumps(manifest, ensure_ascii=False), encoding="utf-8"
    )
    return tmp_path


# ══════════════════════════════════════════════════════════════════════════
# 칸별 통계
# ══════════════════════════════════════════════════════════════════════════


def test_결측률은_실제_결측_행수로_계산된다(반출폴더: Path) -> None:
    """`adj_open` 은 6행 중 2행이 비었다 — 손으로 센 값이다."""
    프로필 = file_profile(반출폴더 / "full/price.parquet")
    칸 = {c["이름"]: c for c in 프로필["칸들"]}

    assert 칸["adj_open"]["결측"] == 2
    assert 칸["adj_open"]["결측률"] == pytest.approx(2 / 6)
    assert 칸["adj_close"]["결측"] == 0
    assert 칸["adj_close"]["결측률"] == 0.0


def test_숫자칸은_최소_최대_평균을_준다(반출폴더: Path) -> None:
    """`close` 는 1050 ~ 7800 이고 평균은 6행의 산술평균이다."""
    프로필 = file_profile(반출폴더 / "full/price.parquet")
    칸 = {c["이름"]: c for c in 프로필["칸들"]}

    assert 칸["close"]["min"] == 1050
    assert 칸["close"]["max"] == 7800
    assert 칸["close"]["평균"] == pytest.approx((7800 + 1100 + 7550 + 7650 + 7750 + 1050) / 6)


def test_결측은_평균에서_빠진다(반출폴더: Path) -> None:
    """`adj_open` 의 평균은 값이 있는 4행만으로 낸다. 결측을 0 으로 세면 안 된다."""
    프로필 = file_profile(반출폴더 / "full/price.parquet")
    칸 = {c["이름"]: c for c in 프로필["칸들"]}

    assert 칸["adj_open"]["평균"] == pytest.approx((754.0 + 1000.0 + 760.0 + 770.0) / 4)


def test_고유값이_적은_칸만_분포를_센다(반출폴더: Path) -> None:
    """`adj_source` 는 2종이라 분포가 있고, `name` 은... 여기선 2종이라 역시 있다.

    분포를 세는 기준은 `RARE_VALUE_LIMIT` 이고, 그 위는 세지 않는다.
    """
    프로필 = file_profile(반출폴더 / "full/price.parquet")
    칸 = {c["이름"]: c for c in 프로필["칸들"]}

    assert 칸["adj_source"]["분포"] == {"fdr": 4, "chain": 2}
    assert 칸["adj_source"]["고유"] == 2
    # 많은 것이 먼저 온다 — 카드에서 잘라 써도 중요한 것이 남는다
    assert list(칸["adj_source"]["분포"]) == ["fdr", "chain"]


def test_고유값이_많으면_분포를_세지_않는다(tmp_path: Path) -> None:
    """한도를 넘는 칸은 분포 칸 자체가 없다. 4,836종 종목명을 카드에 실을 수는 없다."""
    많음 = pd.DataFrame({"x": [f"v{i}" for i in range(RARE_VALUE_LIMIT + 1)]})
    path = tmp_path / "many.parquet"
    많음.to_parquet(path, index=False)

    칸 = file_profile(path)["칸들"][0]
    assert 칸["고유"] == RARE_VALUE_LIMIT + 1
    assert "분포" not in 칸


def test_빈값은_빈값이라는_이름으로_센다(tmp_path: Path) -> None:
    """`sector` 는 KOSPI 에서 빈 문자열이 흔하다. 그대로 두면 표에서 사라져 보인다."""
    표 = pd.DataFrame({"sector": ["", "우량기업부", "", None]})
    path = tmp_path / "sector.parquet"
    표.to_parquet(path, index=False)

    칸 = file_profile(path)["칸들"][0]
    assert 칸["분포"]["(빈값)"] == 3      # 빈 문자열 2 + None 1
    assert 칸["결측"] == 1                # None 만 결측이다


def test_전부_결측인_칸도_기록한다(tmp_path: Path) -> None:
    """비어 있다는 것 자체가 알려야 할 정보라 건너뛰지 않는다."""
    표 = pd.DataFrame({"x": [None, None, None]}, dtype="object")
    path = tmp_path / "empty.parquet"
    표.to_parquet(path, index=False)

    칸 = file_profile(path)["칸들"][0]
    assert 칸["비어있음"] is True
    assert 칸["결측률"] == 1.0
    assert "min" not in 칸


# ══════════════════════════════════════════════════════════════════════════
# 파일 수준
# ══════════════════════════════════════════════════════════════════════════


def test_기간과_종목수를_뽑는다(반출폴더: Path) -> None:
    """`bas_dd` 는 5일(20240102 가 두 번) · `code` 는 2종. 손으로 센 값이다."""
    프로필 = file_profile(반출폴더 / "full/price.parquet")

    assert 프로필["기간"] == {"칸": "bas_dd", "처음": "20240102", "끝": "20240108"}
    assert 프로필["거래일수"] == 5
    assert 프로필["개체"] == {"칸": "code", "수": 2}
    assert 프로필["행"] == 6
    assert 프로필["칸수"] == 9
    assert 프로필["결측있는칸"] == 1        # adj_open 하나뿐이다


def test_앞행을_실물로_담는다(반출폴더: Path) -> None:
    """카드에 실을 예시다. 계산값이 아니라 파일에 있는 그 행이어야 한다."""
    프로필 = file_profile(반출폴더 / "full/price.parquet")

    assert len(프로필["앞행"]) == SAMPLE_ROWS
    assert 프로필["앞행"][0]["code"] == "000020"
    assert 프로필["앞행"][0]["close"] == 7800
    assert 프로필["앞행"][1]["code"] == "0001B0"


def test_CSV_의_종목코드가_숫자로_읽히지_않는다(반출폴더: Path) -> None:
    """🔴 `000020` 이 `20` 이 되면 그 자리에서 종목코드가 아니게 된다.

    5·6번째 자리에 영문이 오는 종목이 84종 있어(`0001B0`) 코드는 애초에 숫자가 아니다.
    """
    프로필 = file_profile(반출폴더 / "small/price.csv")
    칸 = {c["이름"]: c for c in 프로필["칸들"]}

    assert 칸["code"]["형"] == "string"
    assert 칸["code"]["min"] == "000020"
    assert 프로필["앞행"][0]["code"] == "000020"


def test_CSV_와_parquet_이_같은_값을_준다(반출폴더: Path) -> None:
    """같은 표를 두 형식으로 냈으니 통계도 같아야 한다. 읽는 경로가 다르기 때문에 본다."""
    c = {x["이름"]: x for x in file_profile(반출폴더 / "small/price.csv")["칸들"]}
    p = {x["이름"]: x for x in file_profile(반출폴더 / "full/price.parquet")["칸들"]}

    for 이름 in ("code", "close", "adj_open", "adj_source"):
        assert c[이름]["결측"] == p[이름]["결측"], 이름
        assert c[이름]["min"] == p[이름]["min"], 이름
        assert c[이름]["max"] == p[이름]["max"], 이름


# ══════════════════════════════════════════════════════════════════════════
# 폴더 수준
# ══════════════════════════════════════════════════════════════════════════


def test_MANIFEST_에_적힌_파일만_잰다(반출폴더: Path) -> None:
    """`README.md` 처럼 자료가 아닌 파일은 세지 않는다."""
    (반출폴더 / "README.md").write_text("카드", encoding="utf-8")

    프로필 = profile_export(반출폴더)
    assert sorted(f["path"] for f in 프로필["files"]) == ["full/price.parquet",
                                                          "small/price.csv"]


def test_MANIFEST_에_적힌_파일이_없으면_무엇을_할지_알려준다(반출폴더: Path) -> None:
    """막다른 길로 만들지 않는다 — 무엇이 없는지와 다음에 무엇을 할지 함께 말한다."""
    (반출폴더 / "small/price.csv").unlink()

    with pytest.raises(FileNotFoundError) as err:
        profile_export(반출폴더)
    assert "price.csv" in str(err.value)
    assert "export_team_dataset.py" in str(err.value)


def test_PROFILE_은_JSON_으로_되읽힌다(반출폴더: Path) -> None:
    """`NaN` 은 JSON 이 아니다 — 숫자 칸에 결측이 있어도 파일이 깨지면 안 된다."""
    write_profile(반출폴더)

    되읽음 = load_profile(반출폴더)
    assert 되읽음 is not None
    assert len(되읽음["files"]) == 2
    # 파일을 실제로 다시 파싱했는지 — 헐거운 검사가 되지 않게 원문에서 확인한다
    원문 = (반출폴더 / "PROFILE.json").read_text(encoding="utf-8")
    assert "NaN" not in 원문
    assert json.loads(원문)["dev_end"] == "20240831"


def test_PROFILE_이_없으면_None(tmp_path: Path) -> None:
    """없다고 예외를 던지지 않는다 — 부르는 쪽이 무엇을 할지 정한다."""
    assert load_profile(tmp_path) is None


def test_두_번_돌려도_같은_값이_나온다(반출폴더: Path) -> None:
    """재는 일에 부수효과가 없어야 한다. 파일을 읽기만 한다."""
    첫 = write_profile(반출폴더)
    둘 = write_profile(반출폴더)
    assert 첫 == 둘
