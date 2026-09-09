"""텍스트 신호를 팀원용 parquet 으로 반출한다.

    python scripts/export_text_signal.py            # data/outbox/text_signal_<오늘>/
    python scripts/export_text_signal.py --out 경로

## 🔴 DB 는 제목 단위, 반출은 **접수번호 단위**다

`text_signal` 표는 `(text_sha, model_id)` 가 기본키다 — 같은 제목을 두 번 매기지
않으려는 캐시다. 그런데 팀원이 조인하는 것은 **종목과 날짜**다. 그래서 반출 파일은
접수번호마다 한 줄로 편다.

    DB     18,600행  (고유 제목)
    반출 1,295,698행  (개발구간 공시 · 접수번호마다)

## 🔴 `known_at` — 접수일 다음 거래일

`rcept_dt` 에는 **시각이 없다.** 15:00 접수 공시를 그날 신호로 쓰면 장 마감 30분
전에 알던 것이 되어 시세의 T+1 규약보다 앞선다. 그래서 접수일 **다음 거래일**부터
쓴다(`known_rule = 'rceptDt+1session'`).

    code + bas_dd  ≥  known_at   으로 바로 붙는다

`common.trading_calendar.next_session` 을 쓰고, 접수일이 4,200종뿐이라 날짜별로
한 번만 계산한다.

⚠️ **달력 밖(마지막 거래일 이후) 접수는 행을 만들지 않는다.** 다음 거래일을 모르면
   `known_at` 을 지어내는 수밖에 없는데, 지어낸 시점은 미래참조가 된다.

## 🔴 이 신호는 5일 방향과 관계가 없다 (실측 2026-09-04)

카이제곱 독립성 검정에서 **Cramer's V = 0.026** 이 나왔다. p 값은 0에 가깝지만
표본이 126만이라 아주 작은 차이도 유의해진다 — **효과크기를 봐야 한다.**

    기준선 상승 30.54% · 부정 28.83% · 중립 31.06% · 긍정 29.20%

긍정 공시가 중립보다 상승을 **덜** 맞힌다. 그래서 이 칸을 5일 방향 피처로 쓰면
성능이 오르지 않는다. **그런데도 반출하는 이유**는 이벤트 타임라인으로서 값이 있고,
"신호가 없다" 도 재현 가능해야 하는 결과이기 때문이다.

⚠️ **제목별로 보면 큰 차이가 나는데 그건 함정이다.** 상위·하위가 전부 정기보고서다
   ("사업보고서 (2019.12)" 상승 81% · "반기보고서 (2015.06)" 상승 4.6%). 제목에 연월이
   박혀 있어 **그 시점의 시장 전체 움직임**을 가리킨 것이지 감성이 아니다.
   제목을 그대로 피처로 쓰면 모델이 감성이 아니라 시점을 외운다.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd  # noqa: E402

from common.paths import krx_db_path  # noqa: E402
from common.trading_calendar import load_session_days  # noqa: E402
from scripts.score_text_signal import MODEL_ID  # noqa: E402

#: 개발구간 경계. 정본은 `evaluation/horizon.py` 지만 그쪽은 강민석 파트라
#: 값만 맞춰 두고 import 하지 않는다(파트 경계).
HOLDOUT_START = "20240901"

KNOWN_RULE = "rceptDt+1session"
_KST = timezone(timedelta(hours=9))

#: 라이선스 표기가 없는 모델이라 인용으로 대신한다 (모델_라이선스_대장 §3.1).
CITATION = (
    "Kim, Eunhee and Hyopil Shin. KR-FinBert: Fine-tuning KR-FinBert for "
    "Sentiment Analysis. 2022. https://huggingface.co/snunlp/KR-FinBert-SC"
)
LICENSE_NOTE = (
    "🔴 HuggingFace 카드에 라이선스 필드가 없다(미표기). HF 모델의 약 70%가 그렇고, "
    "관례는 라이선스 대신 인용이다. 임의로 '오픈소스' 라고 적지 않는다."
)


def _next_session_map(접수일들, 달력) -> Dict[str, str]:
    """접수일 → 다음 거래일. 날짜가 4,200종뿐이라 한 번만 계산한다."""
    정렬 = sorted(달력)
    import bisect
    out = {}
    for d in 접수일들:
        i = bisect.bisect_right(정렬, d)
        if i < len(정렬):
            out[d] = 정렬[i]
    return out


def build(conn: sqlite3.Connection) -> pd.DataFrame:
    df = pd.read_sql(f"""
        SELECT d.stock_code AS code, d.rcept_no, d.rcept_dt, d.corp_code,
               d.report_nm, t.model_id, t.revision, t.text_sha,
               t.p_pos, t.p_neg, t.p_neu
          FROM dart_disclosure d
          JOIN text_signal t
            ON t.report_nm = d.report_nm AND t.model_id = ?
         WHERE d.rcept_dt < '{HOLDOUT_START}'
           AND d.stock_code IS NOT NULL AND d.stock_code <> ''
         ORDER BY d.rcept_dt, d.rcept_no""", conn, params=(MODEL_ID,))
    print(f"  개발구간 공시 {len(df):,}행")

    달력 = load_session_days()
    지도 = _next_session_map(df["rcept_dt"].unique(), 달력)
    df["known_at"] = df["rcept_dt"].map(지도)
    밖 = df["known_at"].isna().sum()
    if 밖:
        print(f"  ⚠️ 달력 밖이라 뺀 행 {밖:,} (다음 거래일을 몰라 known_at 을 못 낸다)")
    df = df.dropna(subset=["known_at"]).copy()

    # 🔴 **자르는 기준은 접수일이 아니라 `known_at` 이다.**
    #
    # 접수일이 개발구간 마지막 며칠이면 다음 거래일이 홀드아웃으로 넘어간다. 접수일로만
    # 자르면 그 행이 "개발구간 자료" 라는 얼굴로 들어오는데, 실제로 그 신호를 쓸 수
    # 있는 날은 봉인 구간이다. 실측에서 **522행**이 그랬다.
    #
    # 반출 뒤 검사가 이걸 잡았다 — 검사를 먼저 만들지 않았다면 그대로 나갔을 것이다.
    넘어간것 = (df["known_at"] >= HOLDOUT_START).sum()
    if 넘어간것:
        print(f"  ⚠️ 접수는 개발구간인데 known_at 이 봉인으로 넘어간 행 {넘어간것:,} — 뺀다")
        df = df[df["known_at"] < HOLDOUT_START].copy()

    df["known_rule"] = KNOWN_RULE
    df["source"] = "dart:list.json"
    df = df.rename(columns={"revision": "model_rev", "model_id": "model",
                            "text_sha": "text_sha256"})
    칸 = ["code", "rcept_no", "rcept_dt", "known_at", "known_rule", "source",
          "report_nm", "model", "model_rev", "p_pos", "p_neg", "p_neu",
          "text_sha256"]
    return df[칸]


def main() -> int:
    ap = argparse.ArgumentParser(description="텍스트 신호 반출")
    ap.add_argument("--out", default="", help="출력 폴더")
    args = ap.parse_args()

    오늘 = datetime.now(_KST).strftime("%Y%m%d")
    root = Path(args.out) if args.out else Path("data/outbox") / f"text_signal_{오늘}"
    (root / "text").mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(krx_db_path())
    print("── 텍스트 신호 반출 ──")
    df = build(conn)

    out = root / "text" / "text_signal_dev.parquet"
    df.to_parquet(out, index=False, compression="zstd")
    raw = out.read_bytes()
    print(f"  ✅ {out.name}  {len(df):,}행 · {len(raw) / 1024 / 1024:.2f} MB")

    # 🔴 홀드아웃 누수 검사 — 반출 뒤에 **파일을 다시 열어** 확인한다.
    #    메모리의 데이터프레임을 보면 "쓴 것" 이 아니라 "쓰려던 것" 을 보는 것이다.
    되읽음 = pd.read_parquet(out)
    샌것 = (되읽음["rcept_dt"] >= HOLDOUT_START).sum()
    샌것2 = (되읽음["known_at"] >= HOLDOUT_START).sum()
    print(f"  {'✅' if 샌것 == 0 else '🔴'} 봉인 구간 접수 {샌것:,}행")
    print(f"  {'✅' if 샌것2 == 0 else '🔴'} 봉인 구간 known_at {샌것2:,}행 "
          "(접수일이 경계 직전이면 known_at 이 넘어갈 수 있다)")
    if 샌것 or 샌것2:
        print("  🔴 중단 — 홀드아웃이 샌다.")
        return 1

    라벨 = pd.Series("neutral", index=되읽음.index)
    라벨[(되읽음["p_pos"] >= 되읽음["p_neg"])
        & (되읽음["p_pos"] >= 되읽음["p_neu"])] = "positive"
    라벨[(되읽음["p_neg"] > 되읽음["p_pos"])
        & (되읽음["p_neg"] >= 되읽음["p_neu"])] = "negative"
    분포 = {k: int(v) for k, v in 라벨.value_counts().items()}

    manifest = {
        "generated_at": datetime.now(_KST).isoformat(timespec="seconds"),
        "holdout_start": HOLDOUT_START,
        "holdout_start_authority": "evaluation/horizon.py HOLDOUT_START",
        "model": MODEL_ID,
        "revision": (df["model_rev"].dropna().iloc[0]
                     if df["model_rev"].notna().any() else None),
        "downloaded_on": "2026-09-03",
        "license_note": LICENSE_NOTE,
        "citation": CITATION,
        "known_rule": KNOWN_RULE,
        "label_distribution": 분포,
        "chi_square_note": (
            "감성 라벨 × 5일 방향 카이제곱 독립성 검정 (개발구간 1,265,118행): "
            "chi2=1698.3 · dof=4 · p<1e-300 · Cramer's V=0.026. "
            "표본이 커서 p 는 유의하지만 효과크기가 0.1(작음)의 1/4 이다. "
            "기준선 상승 30.54% 대비 부정 28.83% · 중립 31.06% · 긍정 29.20% — "
            "🔴 긍정이 중립보다 상승을 덜 맞힌다. **5일 방향 피처로 쓰지 말 것.**"
        ),
        "caveat": (
            "제목별로 보면 차이가 크지만 그건 감성이 아니라 제목에 박힌 연월이 "
            "시장 시점을 가리키는 것이다 ('사업보고서 (2019.12)' 상승 81% · "
            "'반기보고서 (2015.06)' 상승 4.6%). report_nm 을 그대로 피처로 쓰면 "
            "모델이 감성이 아니라 시점을 외운다."
        ),
        "files": [{
            "path": "text/text_signal_dev.parquet",
            "table": "dart_disclosure × text_signal",
            "time_column": "known_at",
            "rows": int(len(되읽음)),
            "columns": list(되읽음.columns),
            "range": [str(되읽음["rcept_dt"].min()), str(되읽음["rcept_dt"].max())],
            "bytes": len(raw),
            "size_mb": round(len(raw) / 1024 / 1024, 2),
            "sha256": hashlib.sha256(raw).hexdigest(),
            "note": ("공시 제목 감성 확률 3칸 — 접수번호마다 한 줄. "
                     "code + bas_dd ≥ known_at 으로 시세에 붙는다."),
        }],
    }
    mpath = root / "MANIFEST_text.json"
    mpath.write_text(json.dumps(manifest, ensure_ascii=False, indent=2),
                     encoding="utf-8")
    print(f"  ✅ {mpath.name}")
    print(f"\n  라벨 분포 {분포}")
    print(f"\n  올리기: python scripts/upload_to_hf.py --path {root} --no-card")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
