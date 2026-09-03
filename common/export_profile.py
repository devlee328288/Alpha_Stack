"""반출 폴더의 파일을 읽어 **칸마다 무엇이 들었는지** 재고 기록한다.

왜 필요한가
----------
팀원이 받는 것은 파일이고, 파일에는 설명이 없다. `MANIFEST.json` 은 칸 **이름**과 행 수를
적어 주지만 "이 칸에 결측이 얼마나 있나 · 값이 어디부터 어디까지인가 · 이 칸은 몇 종류의
값을 갖나" 는 적지 않는다. 그래서 받는 사람은 매번 직접 `df.describe()` 를 돌려 본다.

더 나쁜 것은 **카드에 손으로 적은 숫자**다. 데이터셋 카드에는 `adj_source` 분포 같은
수치가 글자로 박혀 있었는데, 자료가 바뀌어도 글자는 안 바뀐다. 여기서 재서 카드가 그
값을 그대로 쓰게 하면 그런 어긋남이 생길 자리가 없어진다.

어떻게 재나 — 파일을 직접 읽는다
------------------------------
DB 를 다시 조회하지 않고 **반출된 파일 자체를** 읽는다. 카드가 설명해야 하는 것은
"DB 에 무엇이 있나" 가 아니라 "지금 팀원 손에 가는 이 파일에 무엇이 들었나" 이기 때문이다.
둘은 같아야 하지만, 같은지 확인하는 것과 같다고 믿는 것은 다르다.

parquet 은 **칸 하나씩** 읽는다. 302MB·789만 행을 pandas 로 통째로 열면 문자열 칸이
object 로 부풀어 수 GB 를 먹는다. 칸 단위로 읽으면 20칸 전부 도는 데 실측 8.3초다.

무엇을 재나
----------
    칸마다   형 · 결측 수와 비율 · 값 범위(min~max) · 숫자면 평균
             범주형(고유값이 적으면)은 값마다 몇 행인지 분포까지
    파일마다 행 수 · 칸 수 · 기간(거래일 칸의 처음~끝) · 종목 수 · 앞 3행 실물

`RARE_VALUE_LIMIT` 를 넘게 다양한 값이 있는 칸은 분포를 세지 않는다. 종목명처럼 4,836
종류인 칸의 분포는 카드에 실을 수도 없고 읽는 사람에게 도움도 안 된다.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.csv as pacsv
import pyarrow.parquet as pq

#: 고유값이 이 수 이하인 칸만 값별 분포를 센다. `market`(2) · `adj_source`(2) ·
#: `label`(3) · `sector`(9) 처럼 "몇 종류인지" 가 곧 정보인 칸을 담기 위한 눈금이다.
#: 종목명은 4,836종류라 여기 안 걸리고, 걸려서도 안 된다.
RARE_VALUE_LIMIT = 24

#: 카드에 실을 실물 행 수. 세 줄이면 "칸이 이렇게 생겼구나" 가 전달되고, 그 이상은
#: 표만 길어진다.
SAMPLE_ROWS = 3

#: 기간을 읽을 칸 후보. 앞에 있는 것부터 찾는다.
DATE_COLUMNS = ("bas_dd", "date", "period", "rcept_dt")

#: 종목 수를 읽을 칸 후보.
ENTITY_COLUMNS = ("code", "index_name")


def _is_number(t: pa.DataType) -> bool:
    return pa.types.is_integer(t) or pa.types.is_floating(t) or pa.types.is_decimal(t)


def _py(value: Any) -> Any:
    """JSON 으로 나갈 수 있는 값으로 바꾼다. `NaN`·`Infinity` 는 JSON 이 아니다."""
    if isinstance(value, float):
        if value != value or value in (float("inf"), float("-inf")):
            return None
    return value


def column_profile(array: pa.ChunkedArray | pa.Array, name: str) -> Dict[str, Any]:
    """칸 하나를 재서 사전으로 돌려준다.

    결측만 있는 칸에서 `min_max` 는 `None` 을 주고 `mean` 은 예외가 아니라 `None` 이다.
    비어 있다는 사실 자체가 알려야 할 정보라, 그런 칸도 건너뛰지 않고 기록한다.
    """
    행 = len(array)
    결측 = array.null_count
    프로필: Dict[str, Any] = {
        "이름": name,
        "형": str(array.type),
        "행": 행,
        "결측": 결측,
        "결측률": (결측 / 행) if 행 else 0.0,
    }
    if 행 == 0 or 결측 == 행:
        프로필["비어있음"] = True
        return 프로필

    mm = pc.min_max(array).as_py()
    프로필["min"], 프로필["max"] = _py(mm["min"]), _py(mm["max"])

    if _is_number(array.type):
        프로필["평균"] = _py(pc.mean(array).as_py())
    else:
        고유 = pc.count_distinct(array).as_py()
        프로필["고유"] = 고유
        if 고유 <= RARE_VALUE_LIMIT:
            vc = pc.value_counts(array)
            분포: Dict[str, int] = {}
            for 값, 수 in zip(vc.field("values"), vc.field("counts"), strict=True):
                키 = 값.as_py()
                # 🔴 빈 문자열과 `None` 은 **서로 다른 값인데 같은 이름으로 접힌다.**
                #    대입(`=`)으로 담으면 뒤엣것이 앞엣것을 덮어써 한쪽이 조용히
                #    사라진다 — 합계 행 수는 그대로라 눈으로도 안 잡힌다. 더해서 담는다.
                이름 = "(빈값)" if 키 is None or 키 == "" else str(키)
                분포[이름] = 분포.get(이름, 0) + 수.as_py()
            # 많은 것부터 — 카드에서 잘라 쓸 때 중요한 것이 먼저 남는다
            프로필["분포"] = dict(sorted(분포.items(), key=lambda kv: -kv[1]))
    return 프로필


def _read_table(path: Path, columns: Optional[List[str]] = None) -> pa.Table:
    """parquet·CSV 를 같은 방식으로 연다.

    CSV 는 `utf-8-sig` 로 썼다(Excel 이 한글을 깨뜨리지 않게). pyarrow 는 BOM 을
    스스로 걷어내므로 인코딩을 따로 줄 필요가 없다. 다만 **종목코드를 숫자로 읽지 않게**
    막아야 한다 — `000020` 이 `20` 이 되면 그 자리에서 코드가 아니게 된다.
    """
    if path.suffix == ".parquet":
        return pq.read_table(path, columns=columns)
    문자칸 = {c: pa.string() for c in
              ("bas_dd", "date", "code", "name", "market", "sector",
               "index_name", "index_class", "adj_source", "label")}
    표 = pacsv.read_csv(
        path,
        convert_options=pacsv.ConvertOptions(column_types=문자칸),
    )
    return 표.select(columns) if columns else 표


def file_profile(path: Path, *, relative_to: Optional[Path] = None) -> Dict[str, Any]:
    """파일 하나를 재서 사전으로 돌려준다 — 칸마다, 그리고 파일 전체로.

    parquet 은 칸을 하나씩 읽고 바로 버린다. 전부 메모리에 올리면 789만 행짜리
    파일에서 수 GB 를 잡는다.
    """
    이름 = (path.relative_to(relative_to).as_posix() if relative_to else path.name)
    결과: Dict[str, Any] = {"path": 이름, "칸들": []}

    if path.suffix == ".parquet":
        pf = pq.ParquetFile(path)
        칸이름들 = list(pf.schema_arrow.names)
        결과["행"] = pf.metadata.num_rows
        for 칸 in 칸이름들:
            표 = pq.read_table(path, columns=[칸])
            결과["칸들"].append(column_profile(표.column(칸), 칸))
            del 표
        앞 = pf.read_row_group(0).slice(0, SAMPLE_ROWS)
    else:
        표 = _read_table(path)
        칸이름들 = list(표.column_names)
        결과["행"] = 표.num_rows
        for 칸 in 칸이름들:
            결과["칸들"].append(column_profile(표.column(칸), 칸))
        앞 = 표.slice(0, SAMPLE_ROWS)

    결과["칸수"] = len(칸이름들)
    결과["앞행"] = [{k: _py(v) for k, v in row.items()} for row in 앞.to_pylist()]

    보기 = {c["이름"]: c for c in 결과["칸들"]}
    for 후보 in DATE_COLUMNS:
        if 후보 in 보기 and 보기[후보].get("min"):
            결과["기간"] = {"칸": 후보, "처음": 보기[후보]["min"], "끝": 보기[후보]["max"]}
            결과["거래일수"] = 보기[후보].get("고유")
            break
    for 후보 in ENTITY_COLUMNS:
        if 후보 in 보기 and 보기[후보].get("고유"):
            결과["개체"] = {"칸": 후보, "수": 보기[후보]["고유"]}
            break
    결과["결측있는칸"] = sum(1 for c in 결과["칸들"] if c["결측"])
    return 결과


def profile_export(root: Path) -> Dict[str, Any]:
    """반출 폴더 전체를 잰다. `MANIFEST.json` 이 적어 둔 파일만 본다.

    폴더를 `rglob` 으로 훑지 않는 이유 — 반출 폴더에는 `README.md` 처럼 자료가 아닌
    파일도 산다. MANIFEST 에 적힌 것이 "팀원에게 자료로 나가는 것" 의 정의다.
    """
    manifest = json.loads((root / "MANIFEST.json").read_text(encoding="utf-8"))
    적힌것 = {f["path"] for f in manifest["files"]}

    파일들 = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.name not in 적힌것:
            continue
        파일들.append(file_profile(path, relative_to=root))

    빠진것 = 적힌것 - {Path(f["path"]).name for f in 파일들}
    if 빠진것:
        raise FileNotFoundError(
            f"MANIFEST 에 적힌 파일이 폴더에 없다: {sorted(빠진것)}\n"
            f"  할 일: python scripts/export_team_dataset.py 를 다시 돌린다."
        )

    return {
        "generated_at": manifest["generated_at"],
        "dev_end": manifest["dev_end"],
        "holdout_start": manifest["holdout_start"],
        "rare_value_limit": RARE_VALUE_LIMIT,
        "sample_rows": SAMPLE_ROWS,
        "files": 파일들,
    }


def write_profile(root: Path,
                  프로필: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """`PROFILE.json` 을 반출 폴더에 쓰고 **그 내용을 돌려준다.**

    카드도 팀원 코드도 이 파일 하나를 본다.

    이미 잰 것이 있으면 넘겨서 다시 재지 않는다 — 789만 행 parquet 을 두 번 훑으면
    그만큼 그냥 기다리게 된다.
    """
    프로필 = profile_export(root) if 프로필 is None else 프로필
    (root / "PROFILE.json").write_text(
        json.dumps(프로필, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return 프로필


def load_profile(root: Path) -> Optional[Dict[str, Any]]:
    """있으면 읽고 없으면 `None`. 없다고 예외를 던지지 않는다 — 부르는 쪽이 정한다."""
    path = root / "PROFILE.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))
