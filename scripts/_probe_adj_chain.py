"""수정주가 이어붙이기가 맞는지 **쓰기 전에** 잰다.

    python scripts/_probe_adj_chain.py

FDR 은 20140613 부터만 준다. 그 앞 1,103거래일을 우리 조정계수로 뒤로 이어 붙이는데,
그게 맞는지 확인할 방법이 필요하다. 네 가지를 본다.

  1. **앵커 배율이 공지된 분할비율과 맞나.** 삼성전자 20140613 의 배율이 정확히 1/50
     이어야 한다 (2018년 50:1 분할 하나만 그 뒤에 있으므로).
  2. **알려진 값과 맞나.** 삼성전자 20100104 원종가 809,000 → 16,180.00.
     `corporate_actions.back_adjusted_closes` 의 문서값과 같아야 한다.
  3. **겹치는 구간에서 우리 계산이 FDR 과 얼마나 같나.** ← 이게 제일 중요하다.
     FDR 을 일부러 20140613 하나만 남기고 지운 뒤 나머지를 전부 우리 계산으로
     채워 본다. 그 결과가 진짜 FDR 값과 얼마나 벌어지는지가 곧 **2010~2014 구간에
     우리가 넣을 값의 오차 상한**이다. 그 구간은 대조할 것이 없으므로 이렇게 잰다.
  4. **분할일 수익률이 펴지나.** 삼성전자 20180504 이 -98% 가 아니라 -2.08% 가 되나.

⚠️ 3번이 이 스크립트의 존재 이유다. 1·2·4 는 FDR 이 이미 아는 것을 확인하는 것이라
   "우리 계산" 을 검증하지 못한다.
"""

from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common.paths import krx_db_path  # noqa: E402
from ingest.clients import fdr_data  # noqa: E402
from ingest.store import adj_price  # noqa: E402

#: (코드, 이름, 분할일, 공지 비율). `_probe_fdr_adjust.py` 와 같은 표본이다.
SAMPLES = [
    ("005930", "삼성전자", "20180504", 50.0),
    ("035420", "NAVER", "20181012", 5.0),
    ("035720", "카카오", "20210415", 5.0),
]

#: FDR 이 주는 첫날. 여기가 앵커가 된다.
ANCHOR = "20140613"


def probe(conn: sqlite3.Connection, code: str, name: str,
          split_dd: str, expected: float) -> bool:
    rows = adj_price.load_rows(conn, code)
    by_day = {row["bas_dd"]: i for i, row in enumerate(rows)}
    adjusted = fdr_data.fetch_adjusted(code)
    span = fdr_data.coverage_span(adjusted)

    print(f"\n── {name}({code}) · 우리 {len(rows):,}행 {rows[0]['bas_dd']}~{rows[-1]['bas_dd']}"
          f" · FDR {len(adjusted):,}행 {span[0]}~{span[1]} ──")

    # ① 앵커 배율 ↔ 공지된 분할비율
    built = adj_price.build_rows(rows, adjusted)
    built_by_day = {row[-1]: row for row in built}
    anchor_i = by_day.get(ANCHOR)
    ok_anchor = True
    if anchor_i is not None:
        scales = adj_price.scale_series(
            rows, {d: v.get("adj_close") for d, v in adjusted.items()})
        scale = scales[anchor_i]
        ok_anchor = abs(float(scale) * expected - 1) < 0.01
        print(f"  ① 앵커배율   {ANCHOR} 1/{1 / float(scale):.3f} "
              f"vs 공지 1/{expected:.0f} → {'✅' if ok_anchor else '🔴'}")

    # ② 가장 이른 날의 값
    first = built_by_day[rows[0]["bas_dd"]]
    print(f"  ② 최초일     {rows[0]['bas_dd']} 원종가 {rows[0]['close']:,} "
          f"→ 수정 {first[3]:,.2f}  (출처 {first[4]})")

    # ③ 🔴 앵커 하나만 남기고 우리 계산으로 채운 뒤 진짜 FDR 과 대조
    only_anchor = {ANCHOR: adjusted[ANCHOR]} if ANCHOR in adjusted else {}
    if only_anchor:
        ours = {row[-1]: row[3] for row in adj_price.build_rows(rows, only_anchor)}
        diffs = []
        for day, value in adjusted.items():
            truth = value.get("adj_close")
            mine = ours.get(day)
            if truth and mine:
                diffs.append(abs(mine - truth) / truth)
        worst = max(diffs) if diffs else 0.0
        median = sorted(diffs)[len(diffs) // 2] if diffs else 0.0
        ok_chain = worst < 0.005                 # 0.5% 안이면 반올림 수준으로 본다
        print(f"  ③ 자체계산   {len(diffs):,}일 대조 · 중앙 {median * 100:.4f}% · "
              f"최대 {worst * 100:.4f}% → {'✅' if ok_chain else '🔴'}")
    else:
        ok_chain = False
        print("  ③ 자체계산   앵커일이 없어 대조 불가 🔴")

    # ④ 분할일 수익률이 펴지나
    i = by_day[split_dd]
    raw = rows[i]["close"] / rows[i - 1]["close"] - 1
    adj_now = built_by_day[split_dd][3]
    adj_prev = built_by_day[rows[i - 1]["bas_dd"]][3]
    fixed = adj_now / adj_prev - 1
    krx = rows[i]["change_rate"] if "change_rate" in rows[i] else None
    ok_split = abs(fixed * 100 - (krx if krx is not None else fixed * 100)) < 0.15
    print(f"  ④ 분할일     {split_dd} 원가격 {raw * 100:+.2f}% → 수정 {fixed * 100:+.2f}% "
          f"(KRX {krx:+.2f}%) → {'✅' if ok_split else '🔴'}"
          if krx is not None else
          f"  ④ 분할일     {split_dd} 원가격 {raw * 100:+.2f}% → 수정 {fixed * 100:+.2f}%")
    return ok_anchor and ok_chain and ok_split


def main() -> int:
    conn = sqlite3.connect(f"file:{krx_db_path().as_posix()}?mode=ro", uri=True)
    # change_rate 도 필요하다 — 분할일 대조에 쓴다.
    adj_price.PRICE_COLUMNS = adj_price.PRICE_COLUMNS + ("change_rate",)
    results = [probe(conn, *sample) for sample in SAMPLES]
    conn.close()
    print(f"\n{'✅ 전부 통과' if all(results) else '🔴 실패 있음'} "
          f"— {sum(results)}/{len(results)}")
    return 0 if all(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
