"""반출 폴더를 HuggingFace private 데이터셋으로 올린다.

왜 드라이브가 아니라 HF 인가
--------------------------
팀원이 `hf_hub_download()` 한 줄로 받아 쓰고, 올린 이력이 커밋으로 남아 "누가 언제
무엇을 바꿨나" 를 되짚을 수 있기 때문이다. 드라이브는 링크를 눌러 받는 것까지는 쉽지만
코드에서 받으려면 인증이 번거롭고, 파일이 바뀌어도 무엇이 바뀌었는지 알 수 없다.

🔴 public 이 되면 그 순간 약관 위반이다
------------------------------------
올리는 것은 KRX 원자료다. 이용약관 제11조 ②가 제3자 제공을 금지하므로 이 데이터셋은
**반드시 private** 여야 하고, 조직(`qurious-quant`) 멤버 밖으로 나가면 안 된다.
그래서 이 스크립트는

  1. 레포를 만들 때 `private=True` 를 준다
  2. 만든 뒤 **서버에 다시 물어 private 인지 확인**하고, 아니면 **한 파일도 올리지 않고 멈춘다**

두 번 확인하는 이유는 1번이 실패해도 예외가 안 날 수 있기 때문이다 — 이미 있는 레포에
`exist_ok=True` 로 붙으면 그 레포가 public 이어도 그냥 성공한다.

토큰
----
`.env` 의 `HUGGINGFACE_ACCESS_TOKEN`. 조직에 `repo.write` 가 있어야 한다.
**팀원에게 이 토큰을 주지 않는다.** 팀원은 조직 멤버로 초대하고 각자 read 토큰을 만든다.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Callable, Dict, List, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common.export_profile import load_profile  # noqa: E402
from ingest.clients import hf_data  # noqa: E402

#: 기본 대상. 조직 이름을 앞에 두면 개인 계정 것과 섞이지 않는다.
DEFAULT_REPO = "qurious-quant/alphastack-krx-dev"


def _api(token: str):
    """`huggingface_hub` 을 늦게 부른다 — 없을 때 무엇을 하라고 알려주기 위해서다.

    맨 위에서 import 하면 `ModuleNotFoundError` 한 줄만 나오고, 받는 사람은 무엇을
    깔아야 하는지 모른 채 검색을 시작한다. 막다른 길로 만들지 않는다.
    """
    try:
        from huggingface_hub import HfApi
    except ImportError:
        print("huggingface_hub 가 없다. 아래를 실행할 것:")
        print("    uv pip install huggingface_hub --python .venv/Scripts/python.exe")
        print("  또는")
        print("    .venv/Scripts/python.exe -m pip install -e .[dev]")
        raise SystemExit(2) from None
    return HfApi(token=token)


#: 칸마다 "이게 무엇이고 어디서 발이 걸리나". 통계는 `PROFILE.json` 이 재 주지만
#: **뜻과 함정은 잴 수 없다.** 재는 것과 아는 것을 갈라 두고, 표에서 다시 합친다.
COLUMN_NOTES: Dict[str, str] = {
    # ── 식별 ──
    "bas_dd": "거래일 `YYYYMMDD`. **문자열**이라 사전순 = 날짜순, `<=` 비교가 그대로 통한다",
    "date": "`bas_dd` 를 `YYYY-MM-DD` 로 다시 쓴 것. 같은 날이다",
    "code": "🔴 종목코드. **숫자가 아니다** — 5·6번째에 영문이 오는 종목이 84종(`0001B0`)",
    "name": "종목명. 같은 코드도 이름이 바뀐다. **코드로 잇고 이름으로 잇지 않는다**",
    "market": "`KOSPI` / `KOSDAQ`",
    "sector": "🔴 KRX **소속부**(중견기업부·벤처기업부…)이지 산업 업종이 **아니다**. "
              "KOSPI 는 100% 빈 값. 업종은 `industry` 를 쓴다",
    "industry": "✅ KRX 업종명 (`전기·전자`·`IT 서비스`…). 사람이 KRX 화면에서 받은 "
                "연 1회 업종분류 현황 스냅샷을 **그 행의 날짜 이전 가장 최근 것**으로 붙였다. "
                "KOSPI 만 있고, 그해 스냅샷 뒤에 상장한 종목은 다음 해까지 빈다(약 1%). "
                "업종지수와 붙일 때는 `supply.sector.index_name_for` 로 이름을 맞춘다",
    "industry_bas_dd": "그 업종이 어느 스냅샷에서 왔나 (`YYYYMMDD`). 최대 1년 전이다",
    "industry_known_at": "그 스냅샷을 언제부터 알 수 있었나 — 스냅샷 날짜의 다음 거래일",
    "index_name": "지수 이름",
    "index_class": "지수 구분",
    # ── 원문 가격 ──
    "open": "시가 (원). 🔴 **분할 미조정**",
    "high": "고가 (원). 🔴 **분할 미조정**",
    "low": "저가 (원). 🔴 **분할 미조정**",
    "close": "종가 (원). 🔴 **분할 미조정** — 시가총액 계산과 KRX 원문 대조에만 쓴다",
    "change": "전일 대비 (원)",
    "change_rate": "등락률 (%). **분할이 반영된 값**이라 원가격끼리 나눈 값과 다르다",
    "volume": "거래량 (주). 거래정지일은 0",
    "value": "거래대금 (원). 거래정지일은 0",
    "market_cap": "시가총액 (원). `close × listed_shares` 라 **원가격**을 쓴다. "
                  "1.8경이라 float64 유효숫자를 넘는다",
    "listed_shares": "상장주식수. 증자·감자·분할로 바뀐다",
    # ── 수정 가격 ──
    "adj_open": "✅ **수정 시가**. 거래정지일은 비어 있다",
    "adj_high": "✅ **수정 고가**. 거래정지일은 비어 있다",
    "adj_low": "✅ **수정 저가**. 거래정지일은 비어 있다",
    "adj_close": "✅ **수정 종가** — 수익률·라벨·모멘텀은 전부 이 칸으로",
    "adj_source": "그 행의 수정값이 어디서 왔나. "
                  "`fdr` 은 외부 실측, `chain` 은 우리가 이어 붙인 값",
    # ── 피처 ──
    "sma_5": "단순이동평균 5일", "sma_20": "단순이동평균 20일", "sma_60": "단순이동평균 60일",
    "ema_12": "지수이동평균 12일", "ema_26": "지수이동평균 26일",
    "rsi_14": "RSI 14일 (0~100)", "macd": "MACD (ema12 − ema26)",
    "macd_signal": "MACD 신호선", "macd_hist": "MACD 히스토그램",
    "bb_mid": "볼린저 중심선", "bb_upper": "볼린저 상단", "bb_lower": "볼린저 하단",
    "bb_bandwidth": "볼린저 폭",
    "true_range": "당일 실질 변동폭", "atr_14": "ATR 14일",
    "hv_20": "역사적 변동성 20일 (연율)", "parkinson_20": "파킨슨 변동성 20일 (고저 기반)",
    "vol_sma_20": "거래량 이동평균 20일", "vol_ratio_20": "거래량 / 20일 평균",
    "obv": "누적 거래량 지표", "vwap_20": "거래량 가중 평균가 20일",
    "vol_roc_5": "거래량 변화율 5일",
    # ── 라벨 ──
    "fwd_return_5d": "진입 t+1 시가 → 청산 t+6 시가 수익률. **예측 대상**",
    "label": "위 수익률의 3분류. **`y` 로 쓸 칸**",
}

#: 그 값이 **실제로 그 파일에 있을 때만** 붙이는 경고. `(칸 이름, 판정, 문구)`.
#:
#: 왜 조건을 다나 — 설명을 칸 이름에만 매달면 파일마다 똑같이 붙는다. "최댓값 669만%"
#: 를 지수 파일(`-15.38 ~ 23.42`)에도 붙이면 **카드가 그 파일에 대해 거짓말을 한다.**
#: 판정은 `PROFILE.json` 이 실제로 잰 값으로 한다.
COLUMN_WARNINGS: List[Tuple[str, Callable[[Dict], bool], str]] = [
    ("close", lambda c: c.get("min") == 1,
     "🔴 최솟값 `1` 은 실제 가격이 아니라 **거래정지 중 표시값**이다"),
    ("change_rate", lambda c: (c.get("max") or 0) > 1000,
     "🔴 최댓값이 오류가 아니다 — 거래정지 중 `1`원으로 적혀 있던 종목이 67,000원에 "
     "재개된 날이다 (008080 · 2013-09-11). **그대로 피처에 넣으면 스케일이 이 한 행에 "
     "끌려간다**"),
    ("open", lambda c: c.get("min") == 0, "최솟값 `0` 은 거래정지일이다"),
    ("high", lambda c: c.get("min") == 0, "최솟값 `0` 은 거래정지일이다"),
    ("low", lambda c: c.get("min") == 0, "최솟값 `0` 은 거래정지일이다"),
    ("volume", lambda c: c.get("min") == 0, "최솟값 `0` 은 거래정지일이다"),
]

#: 파일마다 한 줄 소개. `MANIFEST.json` 의 `note` 를 그대로 쓰되, 여기 있으면 이쪽이 이긴다.
FILE_HEADLINE: Dict[str, str] = {
    "full/daily_price_dev.parquet": "개발구간 전 종목 시세 — **최종 학습용은 이것**",
    "full/index_price_dev.parquet": "개발구간 전 지수",
    "small/features_labels_kospi200_dev.csv": "코스피200 피처+라벨 — 받자마자 `fit` 된다",
    "small/features_labels_stocks30_dev.csv": "표본 30종목 피처+라벨 — 받자마자 `fit` 된다",
}

#: 카드에서 **펼쳐** 보여줄 파일. 나머지는 접는다. 20칸·45칸 표 여덟 개를 그대로 늘어놓으면
#: 첫 화면에서 경고문이 밀린다.
CARD_EXPANDED = ("full/daily_price_dev.parquet",
                 "small/features_labels_stocks30_dev.csv")


def _fmt_num(v) -> str:
    """숫자를 읽기 좋게. 과학표기 대신 자릿수를 살린다 — 팀원이 눈으로 크기를 가늠한다."""
    if v is None:
        return "—"
    if isinstance(v, bool):
        return str(v)
    if isinstance(v, int):
        return f"{v:,}"
    if isinstance(v, float):
        if v.is_integer() and abs(v) < 1e15:
            return f"{v:,.0f}"
        if abs(v) >= 1000:
            return f"{v:,.2f}"
        # `%g` 는 **최단 표현**을 준다 — `6.25` 를 `6.2500` 으로 늘여 쓰지 않는다.
        # 자릿수를 고정하면 등락률 `-98.08` 이 `-98.0800` 이 되어 읽기만 나빠진다.
        return f"{v:g}"
    return str(v)


def _fmt_value(v) -> str:
    """표 한 칸에 들어갈 값. 문자열은 따옴표를 씌워 공백·빈값이 보이게 한다."""
    if v is None:
        return "*(빈값)*"
    if isinstance(v, str):
        return f"`{v}`" if v else "*(빈값)*"
    return _fmt_num(v)


def _fmt_range(c: Dict) -> str:
    """값 범위 한 칸. 범주형이면 범위 대신 분포가 훨씬 유용하다."""
    if c.get("비어있음"):
        return "**전부 비어 있음**"
    분포 = c.get("분포")
    if 분포:
        총 = sum(분포.values()) or 1
        조각 = [f"`{k}` {v / 총:.1%}" for k, v in list(분포.items())[:6]]
        if len(분포) > 6:
            조각.append(f"… 외 {len(분포) - 6}종")
        return " · ".join(조각)
    lo, hi = c.get("min"), c.get("max")
    if isinstance(lo, str):
        return f"`{lo}` ~ `{hi}`"
    범위 = f"{_fmt_num(lo)} ~ {_fmt_num(hi)}"
    if "평균" in c and c["평균"] is not None:
        범위 += f"<br>평균 {_fmt_num(c['평균'])}"
    return 범위


def _column_table(파일: Dict) -> str:
    """칸별 표 하나. 결측이 있는 칸은 눈에 띄게 표시한다."""
    줄 = ["| 칸 | 형 | 결측 | 값 범위 · 분포 | 무엇인가 |", "|---|---|---:|---|---|"]
    for c in 파일["칸들"]:
        결측 = "—" if c["결측"] == 0 else f"**{c['결측률']:.2%}**"
        형 = str(c["형"]).replace("large_string", "문자").replace("string", "문자")
        형 = 형.replace("double", "실수").replace("int64", "정수")
        뜻 = COLUMN_NOTES.get(c["이름"], "")
        경고 = [문구 for 칸, 판정, 문구 in COLUMN_WARNINGS
                if 칸 == c["이름"] and 판정(c)]
        if 경고:
            뜻 = ". ".join([뜻, *경고]) if 뜻 else ". ".join(경고)
        줄.append(f"| `{c['이름']}` | {형} | {결측} | {_fmt_range(c)} | {뜻} |")
    return "\n".join(줄)


def _sample_rows_table(파일: Dict) -> str:
    """앞 3행을 **세로로** 세운 표. 칸이 20~45개라 가로로 두면 읽을 수가 없다."""
    행들 = 파일.get("앞행") or []
    if not 행들:
        return "*(행이 없습니다)*"
    머리 = "| 칸 | " + " | ".join(f"{i + 1}행" for i in range(len(행들))) + " |"
    구분 = "|---|" + "---|" * len(행들)
    줄 = [머리, 구분]
    for 칸 in 행들[0]:
        값들 = " | ".join(_fmt_value(r.get(칸)) for r in 행들)
        줄.append(f"| `{칸}` | {값들} |")
    return "\n".join(줄)


def _file_overview(profile: Dict, manifest: Dict) -> str:
    """파일 목록 표 — 행·크기에 더해 **기간과 종목 수**까지 한눈에."""
    크기 = {f["path"]: f for f in manifest["files"]}
    줄 = ["| 파일 | 행 | 칸 | MB | 기간 | 종목·지수 | 무엇인가 |",
          "|---|---:|---:|---:|---|---:|---|"]
    for f in profile["files"]:
        m = 크기.get(Path(f["path"]).name, {})
        기간 = f.get("기간")
        구간 = f"{기간['처음']} ~ {기간['끝']}" if 기간 else "—"
        개체 = f.get("개체")
        수 = f"{개체['수']:,}" if 개체 else "—"
        설명 = FILE_HEADLINE.get(f["path"], m.get("note", ""))
        줄.append(f"| `{f['path']}` | {f['행']:,} | {f['칸수']} | "
                  f"{m.get('size_mb', 0):.1f} | {구간} | {수} | {설명} |")
    return "\n".join(줄)


def _column_sections(profile: Dict) -> str:
    """파일마다 칸 표. 핵심 둘은 펼치고 나머지는 접는다."""
    조각 = []
    for f in profile["files"]:
        꼬리 = f"{f['행']:,}행 × {f['칸수']}칸 · 결측이 있는 칸 {f['결측있는칸']}개"
        표 = _column_table(f)
        if f["path"] in CARD_EXPANDED:
            조각.append(f"### `{f['path']}` — {꼬리}\n\n{표}\n")
        else:
            # `<summary>` 안에서는 마크다운 백틱이 렌더되지 않는 곳이 있다. HTML 태그로 쓴다.
            조각.append(
                f"<details>\n<summary><code>{f['path']}</code> — {꼬리} · 펼쳐 보기</summary>\n\n"
                f"{표}\n\n</details>\n"
            )
    return "\n".join(조각)


def _adj_source_line(profile: Dict) -> str:
    """`adj_source` 분포를 **재서** 쓴다. 예전에는 이 숫자가 카드에 손으로 박혀 있었다."""
    for f in profile["files"]:
        if f["path"] != "full/daily_price_dev.parquet":
            continue
        for c in f["칸들"]:
            if c["이름"] == "adj_source" and c.get("분포"):
                총 = sum(c["분포"].values()) or 1
                return " · ".join(f"`{k}` {v / 총:.2%} ({v:,}행)"
                                  for k, v in c["분포"].items())
    return "*(PROFILE.json 에 `adj_source` 가 없습니다)*"


def _missing_highlights(profile: Dict) -> str:
    """결측이 있는 칸만 모아 한 표로. 받아서 처음 부딪히는 것이 대개 결측이다."""
    줄 = ["| 파일 | 칸 | 결측 | 비율 |", "|---|---|---:|---:|"]
    개수 = 0
    for f in profile["files"]:
        for c in f["칸들"]:
            if not c["결측"]:
                continue
            개수 += 1
            줄.append(f"| `{f['path']}` | `{c['이름']}` | {c['결측']:,} | {c['결측률']:.2%} |")
    if 개수 == 0:
        return "결측이 있는 칸이 **한 곳도 없습니다.**"
    return "\n".join(줄)


#: 카드에 끼우는 "종류별 지도" 절 — identity/·financial/·macro/·calendar/ 폴더처럼 **이
#: 스크립트가 만들지 않는 반출본**을 설명한다. 카드에 없는 폴더는 없는 것과 같다:
#: `hf_hub_download(filename=...)` 에 적을 이름이 카드에만 있기 때문이다.
#:
#: 🔴 이 파일이 없으면 카드를 다시 만들 때 그 절이 **통째로 사라진다.** 2026-09-03 에
#:    병행 세션이 손으로 끼운 절이 있는 채로 카드를 다시 만들 뻔했다. 그래서 정본을
#:    저장소 안으로 들이고 빌더가 매번 끼운다.
REFERENCE_BLOCK = Path(__file__).resolve().parent / "hf_card_reference_block.md"
#: 끼우는 자리 — 이 제목 **바로 앞**. 맨 끝에 붙이면 55KB 카드의 스크롤 끝이라 아무도 못 본다.
REFERENCE_ANCHOR = "## 🔴 가장 먼저 알아야 할 것 세 가지"
REFERENCE_MARK = "<!-- reference-block -->"


def with_reference_block(card: str) -> str:
    """카드에 종류별 지도 절을 끼운다. 이미 있으면 두 번 끼우지 않는다."""
    if REFERENCE_MARK in card:
        return card
    if not REFERENCE_BLOCK.exists():
        print(f"⚠️ {REFERENCE_BLOCK.name} 이 없다 — 종류별 지도 절 없이 카드를 만든다")
        return card
    block = REFERENCE_BLOCK.read_text(encoding="utf-8").strip() + "\n\n"
    if REFERENCE_ANCHOR not in card:
        print(f"⚠️ 카드에 '{REFERENCE_ANCHOR}' 가 없어 종류별 지도 절을 끝에 붙인다")
        return card.rstrip() + "\n\n" + block
    return card.replace(REFERENCE_ANCHOR, block + REFERENCE_ANCHOR, 1)


def build_dataset_card(root: Path, repo_id: str) -> str:
    """HF 가 데이터셋 첫 화면에 띄우는 카드(README.md)를 만든다.

    팀원이 레포에 들어와 가장 먼저 보는 글이라, **경고를 맨 위에** 둔다.

    숫자는 `MANIFEST.json` 과 `PROFILE.json` 에서만 가져온다 — 카드에 손으로 적지
    않는다. 예전에는 `adj_source` 분포 같은 값이 글자로 박혀 있었고, 자료가 바뀌어도
    글자는 안 바뀌었다. 여기서 그럴 자리를 없앤다.

    반출 폴더에 `CHANGES.md` 가 있으면 **경고 바로 다음에** 그대로 끼운다. 이전 배포본을
    이미 받은 사람이 무엇을 버리고 무엇을 다시 받아야 하는지가 가장 급한 소식이기 때문이다.
    """
    manifest = json.loads((root / "MANIFEST.json").read_text(encoding="utf-8"))
    profile = load_profile(root)
    if profile is None:
        raise FileNotFoundError(
            f"{root}/PROFILE.json 이 없다 — 칸별 통계 없이는 카드를 만들 수 없다.\n"
            f"  할 일: python scripts/profile_export.py --path {root}"
        )

    지수분포 = manifest["stats"].get("kospi200", {}).get("distribution", {})
    종목분포 = manifest["stats"].get("stocks30", {}).get("distribution", {})
    종목기준 = manifest["stats"].get("stocks30", {}).get("price_basis", "adj_close")
    지수기준 = manifest["stats"].get("kospi200", {}).get("price_basis", "close")

    dev = manifest["dev_end"]
    dev_iso = f"{dev[:4]}-{dev[4:6]}-{dev[6:]}"

    def 비율(d: Dict[str, int]) -> str:
        총 = sum(d.values()) or 1
        return " · ".join(f"{k} {v / 총:.2%}" for k, v in d.items())

    변경 = ""
    changes = root / "CHANGES.md"
    if changes.exists():
        변경 = "\n" + changes.read_text(encoding="utf-8").strip() + "\n"

    총칸 = sum(f["칸수"] for f in profile["files"])
    총행 = sum(f["행"] for f in profile["files"])

    return f"""---
license: other
license_name: krx-terms
license_link: https://data.krx.co.kr
language:
  - ko
tags:
  - finance
  - korean-stock
  - time-series
pretty_name: AlphaStack KRX 개발구간
---

# AlphaStack — KRX 개발구간 데이터셋

> 🔴 **이 저장소는 private 이며, 조직 `qurious-quant` 밖으로 나가면 안 됩니다.**
> 담긴 것은 KRX 원자료이고 이용약관 제11조 ②가 제3자 제공을 금지합니다.
> 파일을 다른 곳에 다시 올리거나 공개 저장소에 커밋하지 마세요.

> 🔴 **홀드아웃(`{manifest['holdout_start']}` 이후)은 여기 없습니다. 찾지도 마세요.**
> 이 데이터셋은 `{manifest['dev_end']}` 까지만 담습니다. 봉인 구간을 미리 보면
> "미리 정해 두고 딱 한 번 열어본다" 는 검증 설계가 그 자리에서 무너지고,
> 되돌릴 방법이 없습니다.

생성 시각 `{manifest['generated_at']}` · 파일 {len(profile['files'])}개 ·
합계 {총행:,}행 × {총칸}칸
{변경}
## 무엇이 들어 있나

{_file_overview(profile, manifest)}

## 🔴 가장 먼저 알아야 할 것 세 가지

### ① 수익률은 `close` 가 아니라 `adj_close` 로 계산하세요

`open`·`high`·`low`·`close` 는 **KRX 원문 그대로라 액면분할이 조정돼 있지 않습니다.**
그대로 수익률을 내면 분할일이 폭락으로 읽힙니다.

```
삼성전자 2018-05-04 (50:1 액면분할)
  close     로 계산 →  -98.04%   ← 틀렸다
  adj_close 로 계산 →   -2.08%   ← 맞다 (KRX 공시 등락률과 같다)
```

그래서 조정된 값을 옆에 붙였습니다.

| 무엇을 하려나 | 어느 칸을 쓰나 |
|---|---|
| 수익률 · 라벨 · 모멘텀 | **`adj_close`** |
| 변동성 · 고저 폭 (ATR · Parkinson) | **`adj_high`·`adj_low`·`adj_open`** |
| 시가총액 | **`close`** — `market_cap = close × listed_shares` 라서 |
| KRX 원문과 대조 | **`close`** |

**원 칸을 지우지 않은 이유**: 수정주가는 *현재 가격 기준으로 과거를 눌러 놓은 값*이라
**새 분할이 생기면 과거 값이 전부 바뀝니다.** 원문을 남겨 두어야 언제든 되돌아갈 수 있습니다.

`adj_source` 칸이 그 행의 값이 어디서 왔는지 알려 줍니다 —
{_adj_source_line(profile)}

`chain` 이 있는 이유: FinanceDataReader 는 **최근 3,000거래일만** 줍니다. 우리 자료는
2010년부터라 그 앞이 비어서, 외부 값이 닿는 가장 이른 날을 기준점으로 삼아 과거로 이어
붙였습니다. 그 계산의 오차는 **0.4% 안쪽**입니다 — 기준점 하나만 남기고 나머지
2,998일을 우리 계산으로 채운 뒤 진짜 값과 대조해 실측했습니다 (2026-09-03 기준 최대
0.397%. 외부 창이 날마다 하루씩 밀려 기준점이 바뀌므로 소수 셋째 자리는 움직입니다).

### ② 피처와 라벨은 **이미 수정주가로** 계산돼 있습니다

`features_labels_*.csv` 의 `sma_5`·`rsi_14`·`fwd_return_5d`·`label` 은 다시 계산할
필요가 없습니다. **칸 이름은 그대로 두고 값만** 아래 기준으로 냈습니다.

| 파일 | 계산 기준 | 왜 |
|---|---|---|
| `features_labels_stocks30_dev.csv` | **`{종목기준}`** (수정주가) | 종목에는 액면분할이 있다 |
| `features_labels_kospi200_dev.csv` | `{지수기준}` (원문) | 지수에는 분할이라는 사건 자체가 없다 |

⚠️ **2026-09-03 이전에 받으셨다면 종목 피처·라벨을 버리고 다시 받으세요.** 그때까지는
원문 가격으로 계산돼 있었습니다. 표본 30종목 기준 라벨 370행이 뒤집혔고(삼성전자
2018-04-25 는 `-97.90%` 하락 → `+5.12%` 상승으로 **정반대**), `rsi_14` 는 분할 한 건에
198행(5.49%)이 어긋났습니다. 지수(`kospi200`)는 처음부터 영향이 없었습니다.

### ③ 한글이 깨져 보이면 파일이 아니라 **읽는 방법**입니다

파일은 전부 UTF-8 입니다. 한국어 Windows 의 파이썬 기본 인코딩은 `cp949` 라서,
`encoding=` 을 안 주면 UTF-8 파일을 cp949 로 해석해 `ê¸°ì¤ì¼` 같은 글자가 나옵니다.

```python
# 🔴 이렇게 하면 깨집니다
meta = json.load(open(path))

# ✅ 이렇게 하세요
with open(path, encoding="utf-8") as f:
    meta = json.load(f)

# CSV 는 pandas 가 기본 UTF-8 이라 그냥 읽힙니다
df = pd.read_csv(path)               # ✅
# parquet 은 UTF-8 이 내장이라 애초에 안 깨집니다
```

콘솔 출력까지 깨진다면 — PowerShell `$env:PYTHONUTF8=1` · cmd `chcp 65001`.

## 바로 쓰기

```python
from huggingface_hub import hf_hub_download
import pandas as pd

path = hf_hub_download(
    repo_id="{repo_id}",
    filename="small/features_labels_kospi200_dev.csv",
    repo_type="dataset",
)
df = pd.read_csv(path)

FEATURES = [c for c in df.columns if c not in
            ("bas_dd", "date", "index_name", "index_class",
             "open", "high", "low", "close", "change", "change_rate",
             "volume", "value", "market_cap", "fwd_return_5d", "label")]
X, y = df[FEATURES], df["label"]
```

⚠️ `X` 와 `y` 를 무작위로 섞어 나누지 마세요. 시계열이라 **시간 순서로** 잘라야 합니다.

🔴 종목 파일(`features_labels_stocks30_dev.csv`)에서는 `code` 를 **문자열로** 읽어야
합니다. 안 그러면 `000020` 이 `20` 이 됩니다.

```python
df = pd.read_csv(path, dtype={{"code": str, "bas_dd": str}})
```

## 예측 대상

- 진입 **t+1 시가** → 청산 **t+6 시가** ({manifest['horizon']}거래일)
- 3분류 중립 밴드: 지수 ±{manifest['band']['index']:.1%} · 종목 ±{manifest['band']['stock']:.1%}
- 지수 라벨 분포: {비율(지수분포)}
- 표본 종목 라벨 분포: {비율(종목분포)}
  ⚠️ 표본 30종목의 분포입니다. 거래정지·상장폐지 사례를 **일부러 섞어** 뽑았기 때문에
  전 종목 분포와 다릅니다. 전체 통계로 인용하지 마세요.

## 개발구간이 어디까지인가 — 오해가 잦은 곳

**개발구간에 하한은 없습니다.** `{manifest['dev_end']}` 는 **끝**이지 시작이 아닙니다.

```
{dev_iso} 로 경계가 바뀐 것은 "끝이 뒤로 밀린 것"입니다.
시작은 여전히 2010-01-04 입니다.

  2010-01-04 ─────────────────────── {dev_iso}  │  봉인 {manifest['holdout_start']} ~
             ↑ 여기부터 전부 쓸 수 있습니다        │  (여기 없습니다)
```

늘어난 구간은 기존 구간을 **대체하는 것이 아니라 더해지는 것**입니다.
뒤쪽 몇 년만 잘라 쓰면 학습 자료의 대부분을 버리게 됩니다.

## 결측이 있는 칸 — 받아서 처음 부딪히는 것

{_missing_highlights(profile)}

거래정지일(`open=high=low=0`)에는 `adj_open`·`adj_high`·`adj_low` 가 **비어 있고**
`adj_close` 만 있습니다. 그 날은 체결이 없었으므로 시·고·저가가 존재하지 않습니다.
`0` 으로 채우면 "그 날 가격이 0원" 이 되니 그대로 두거나 걸러 내세요.

지수 쪽 결측은 성격이 다릅니다 — **지수마다 주는 항목이 다릅니다.** 어떤 지수는
시·고·저가를 아예 주지 않고, 거래량·거래대금이 없는 지수도 있습니다. 특정 지수를
쓰기 전에 그 지수에 그 칸이 있는지부터 확인하세요.

## 칸마다 무엇이 들었나

아래 표의 결측률·값 범위·분포는 **올린 파일을 그 자리에서 다시 읽어 잰 값**입니다
(`PROFILE.json`). 손으로 적은 숫자가 아니라서 자료와 어긋날 수 없습니다.

{_column_sections(profile)}

## 행이 실제로 어떻게 생겼나

`full/daily_price_dev.parquet` 의 **맨 앞 {profile['sample_rows']}행**입니다.
가공하지 않은 실물이라, 코드에서 어떤 형으로 받게 되는지 그대로 보입니다.

{_sample_rows_table([f for f in profile['files']
                     if f['path'] == 'full/daily_price_dev.parquet'][0])}

<details>
<summary><b>피처·라벨 파일도 앞 {profile['sample_rows']}행 보기</b></summary>

{_sample_rows_table([f for f in profile['files']
                     if f['path'] == 'small/features_labels_stocks30_dev.csv'][0])}

</details>

## 통계를 코드로 쓰고 싶다면 — `PROFILE.json`

위 표의 모든 숫자가 기계가 읽는 형태로도 들어 있습니다.

```python
import json
from huggingface_hub import hf_hub_download

with open(hf_hub_download(repo_id="{repo_id}", filename="PROFILE.json",
                          repo_type="dataset"), encoding="utf-8") as f:
    prof = json.load(f)

for 파일 in prof["files"]:
    print(파일["path"], f'{{파일["행"]:,}}행 × {{파일["칸수"]}}칸')
    for 칸 in 파일["칸들"]:
        if 칸["결측"]:
            print("   결측", 칸["이름"], f'{{칸["결측률"]:.2%}}')
```

고유값이 {profile['rare_value_limit']}종 이하인 칸에는 `분포` 가 함께 들어 있습니다.
그보다 다양한 칸(종목명 등)은 세지 않았습니다 — 표에 실을 수도 없고 도움도 안 되기 때문입니다.

## 무결성 확인

`MANIFEST.json` 에 파일마다 SHA-256 이 있습니다. 받은 파일이 보낸 것과 같은지
맞춰 볼 수 있습니다.

```python
import hashlib, json

with open("MANIFEST.json", encoding="utf-8") as f:      # encoding 필수
    manifest = json.load(f)
for item in manifest["files"]:
    print(item["path"], item["sha256"][:16], f'{{item["rows"]:,}}행')
```

## 만든 방법

`scripts/export_team_dataset.py` → `scripts/profile_export.py` → `scripts/upload_to_hf.py`
(저장소 `devlee328288/Alpha_Stack`). 같은 커밋에서 다시 돌리면 같은 파일이 나옵니다 —
실제로 이번 반출에서 8개 중 7개가 직전 반출과 SHA-256 까지 같았습니다.

이 카드는 **업로드할 때마다 `MANIFEST.json` 과 `PROFILE.json` 에서 자동으로 다시
만들어집니다.** 파일 목록·행 수·칸별 결측률·값 범위·분포·홀드아웃 경계·라벨 분포는
항상 지금 올라간 자료의 값입니다.

문제가 보이면 저장소에 이슈로 남겨 주세요. 데이터 파트(이동원)가 봅니다.
"""


def main() -> int:
    parser = argparse.ArgumentParser(description="반출 폴더를 HF private 데이터셋으로 올린다")
    parser.add_argument("--repo", default=DEFAULT_REPO, help=f"기본 {DEFAULT_REPO}")
    parser.add_argument("--path", default=None,
                        help="올릴 폴더 (기본: data/outbox 의 가장 최근 날짜)")
    parser.add_argument("--path-in-repo", default="",
                        help="레포 안의 하위 폴더 (기본: 루트). "
                             "🔴 같은 레포에 성격이 다른 자료를 함께 둘 때 반드시 준다 — "
                             "루트에 올리면 기존 README.md·MANIFEST.json 을 덮어쓴다")
    parser.add_argument("--no-card", action="store_true",
                        help="데이터셋 카드를 만들지 않는다. 🔴 build_dataset_card 는 "
                             "시세 반출(MANIFEST 의 stats.kospi200)을 전제하므로 "
                             "다른 종류의 반출은 자기 README.md 를 들고 와야 한다")
    parser.add_argument("--note", default="",
                        help="커밋 메시지에 덧붙일 한 줄")
    parser.add_argument("--dry-run", action="store_true",
                        help="올리지 않고 무엇을 올릴지만 보여준다")
    args = parser.parse_args()

    if args.path:
        root = Path(args.path)
    else:
        # 🔴 **날짜 모양(YYYY-MM-DD)인 폴더만 본다.**
        #
        # 예전에는 `glob("*")` 을 그냥 정렬해 마지막을 골랐다. 그런데 `data/outbox` 에는
        # 성격이 다른 반출도 함께 산다(`dart_20260902` 등). 사전순으로 정렬하면
        # `'d' > '2'` 라서 **`dart_20260902` 가 `2026-09-02` 를 이긴다.**
        #
        # 실제로 2026-09-02 에 이 일이 났다. 시세 반출을 올리려는데 기본값이 DART 폴더를
        # 골랐고, 그대로 갔으면 **DART 자료가 시세 레포의 루트를 덮을 뻔했다**
        # (README.md·MANIFEST.json 까지). `--dry-run` 이 잡아서 막았다.
        날짜모양 = "[0-9][0-9][0-9][0-9]-[0-9][0-9]-[0-9][0-9]"
        후보 = sorted(p for p in Path("data/outbox").glob(날짜모양) if p.is_dir())
        if not 후보:
            print("data/outbox 에 날짜(YYYY-MM-DD) 폴더가 없다.")
            print("  할 일: python scripts/export_team_dataset.py 를 먼저 돌린다.")
            print("  다른 종류의 반출을 올리려면 --path 로 폴더를 직접 지정한다.")
            return 1
        root = 후보[-1]
        print(f"(날짜 폴더 {len(후보)}개 중 가장 최근을 골랐다. 다른 것을 올리려면 --path)")

    if not (root / "MANIFEST.json").exists():
        print(f"{root}/MANIFEST.json 이 없다. 반출이 끝나지 않았다")
        return 1

    token, source = hf_data.load_hf_key()
    if not token:
        print("HUGGINGFACE_ACCESS_TOKEN 이 없다 (.env 확인)")
        return 1
    print(f"토큰 출처: {source}")
    print(f"올릴 폴더: {root}")
    print(f"대상 레포: {args.repo}"
          + (f"  (하위 폴더 {args.path_in_repo}/)" if args.path_in_repo else "  (루트)"))
    print()

    api = _api(token)

    # ── 레포 준비 ────────────────────────────────────────────────────
    api.create_repo(repo_id=args.repo, repo_type="dataset", private=True, exist_ok=True)

    # 🔴 만들었다고 믿지 않는다. 서버에 다시 물어본다.
    info = api.repo_info(repo_id=args.repo, repo_type="dataset")
    if not info.private:
        print("🔴 중단 — 이 데이터셋이 public 이다. KRX 원자료를 올릴 수 없다.")
        print(f"   https://huggingface.co/datasets/{args.repo}/settings 에서")
        print("   private 으로 바꾼 뒤 다시 실행할 것.")
        return 1
    print("✅ private 확인")

    # ── 데이터셋 카드 ────────────────────────────────────────────────
    if args.no_card:
        있나 = (root / "README.md").exists()
        print(f"✅ 카드 생성 건너뜀 (반출본의 README.md {'있음' if 있나 else '없음'})")
    else:
        card = with_reference_block(build_dataset_card(root, args.repo))
        (root / "README.md").write_text(card, encoding="utf-8")
        print("✅ 데이터셋 카드 생성")

    올릴것 = [p for p in sorted(root.rglob("*")) if p.is_file()]
    총MB = sum(p.stat().st_size for p in 올릴것) / 1024 / 1024
    print()
    print(f"올릴 파일 {len(올릴것)}개 · 합계 {총MB:,.1f} MB")
    for p in 올릴것:
        print(f"   {p.relative_to(root).as_posix():46s} {p.stat().st_size / 1024 / 1024:>8.2f} MB")

    if args.dry_run:
        print()
        print("--dry-run 이라 여기서 멈춘다")
        return 0

    print()
    print("업로드 중… (142MB parquet 이 있어 몇 분 걸린다)")
    메시지 = f"데이터 반출 {root.name}"
    if args.note:
        메시지 += f" — {args.note}"
    api.upload_folder(
        folder_path=str(root),
        repo_id=args.repo,
        repo_type="dataset",
        path_in_repo=args.path_in_repo or None,
        commit_message=메시지,
    )

    올라간 = api.list_repo_files(repo_id=args.repo, repo_type="dataset")
    print()
    print(f"✅ 업로드 완료 — 서버에 파일 {len(올라간)}개")
    print(f"   https://huggingface.co/datasets/{args.repo}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
