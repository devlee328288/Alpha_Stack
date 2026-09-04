"""FDR 이 조정하지 않은 자본변동 자리를 이어 붙인다.

    python scripts/fix_capital_reduction.py            # 무엇을 고칠지만 본다 (기본)
    python scripts/fix_capital_reduction.py --apply    # 실제로 고친다

## 무엇이 문제인가

`daily_price.adj_*` 는 두 출처를 섞는다 — FDR 이 닿는 구간은 외부 실측(`fdr`)을 그대로
쓰고, 그 앞은 우리 계수로 이어 붙인다(`chain`). 그런데 **FDR 이 감자를 조정하지 않는다.**

    017170 훈영  20110401 → 20110404
      주식수  53,406,538 → 2,670,326   (1/20 감자)
      close          105 → 2,100       (20배 — 이론대로 뛰었다)
      adj_close      105 → 2,100       🔴 똑같이 뛰었다 = 안 이어졌다

그 날 수익률이 **+1,900%** 로 읽히고 라벨이 상승으로 뒤집힌다.

FDR 의 공식 입장은 *"모든 가격 데이터는 수정 주가"* 지만(Issue #21) 실측이 다르다.
전량 검사에서 **17자리**가 나왔다.

## 어떻게 고치나 — 과거를 배율만큼 올린다

수정주가는 **뒤로 이어 붙인다**(back-adjust). 최근 가격을 그대로 두고 과거를 옮긴다.

주식수가 1/k 로 줄면(감자) 주가는 k 배가 된다. 그러니 그 날 **이전** 행 전부의
`adj_*` 에 k 를 곱하면 이어진다. 반대로 액면분할이면 k 가 1보다 작아 과거가 내려간다.

    scale[i] = scale[i+1] * factor[i+1]      ← adj_price.py 와 같은 방향

배율은 `Fraction` 으로 옮긴다 — 한 종목에 자리가 둘 이상일 수 있고(아센디오가 그렇다),
부동소수로 누적하면 반올림 잡음이 되돌아온다.

## 🔴 왜 원가격(`close`)은 안 건드리나

`close` 는 KRX 가 준 그대로다. **원자료는 고치지 않는다.** 그 날 실제로 2,100원에
거래됐고 그게 사실이다. 이어 붙이는 것은 파생값인 `adj_*` 의 몫이다.

## 되돌리기

`--apply` 는 고친 행을 `reports/fix_capital_reduction_<타임스탬프>.json` 에 **원래
값째로** 남긴다. 잘못됐으면 그 파일로 되돌릴 수 있다.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from datetime import datetime, timedelta, timezone
from fractions import Fraction
from pathlib import Path
from typing import Dict, List, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common.paths import krx_db_path  # noqa: E402

#: 상장주식수가 이만큼 변한 자리부터 본다. 10배로 두면 5:1·3:1 감자를 놓친다.
SHARE_RATIO_MIN = Fraction(2)

#: `close` 가 이론 점프를 따라갔다고 볼 허용 오차. 기준가는 호가단위로 반올림되므로
#: 정확히 일치하지 않는다.
TOLERANCE = 0.30

_KST = timezone(timedelta(hours=9))


def 미조정_자리(conn: sqlite3.Connection) -> List[Dict]:
    """수정주가가 못 이은 자리. `scripts/verify_base_info.py` 9절과 같은 판정이다."""
    후보 = conn.execute("""
        WITH t AS (
          SELECT code, isu_abbrv, market, kind_stkcert_tp_nm AS 종류, bas_dd,
                 CAST(list_shrs AS INTEGER) AS s,
                 LAG(CAST(list_shrs AS INTEGER)) OVER w AS p,
                 LAG(bas_dd) OVER w AS pd
            FROM stock_base_info
          WINDOW w AS (PARTITION BY code ORDER BY bas_dd))
        SELECT code, isu_abbrv, market, 종류, pd, bas_dd, p, s
          FROM t WHERE p > 0 AND s > 0
           AND (s * 1.0 / p >= ? OR p * 1.0 / s >= ?)
         ORDER BY code, bas_dd""",
        (float(SHARE_RATIO_MIN), float(SHARE_RATIO_MIN))).fetchall()

    필요 = set()
    for r in 후보:
        필요.add((r["code"], r["pd"]))
        필요.add((r["code"], r["bas_dd"]))
    시세 = {}
    for row in conn.execute("SELECT code, bas_dd, close, adj_close FROM daily_price"):
        키 = (row["code"], row["bas_dd"])
        if 키 in 필요:
            시세[키] = (row["close"], row["adj_close"])

    out = []
    for r in 후보:
        a, b = 시세.get((r["code"], r["pd"])), 시세.get((r["code"], r["bas_dd"]))
        if not a or not b or not all([a[0], a[1], b[0], b[1]]):
            continue
        이론 = Fraction(int(r["p"]), int(r["s"]))       # 주식수가 1/10 이면 주가는 10배
        if abs((b[0] / a[0]) / float(이론) - 1) > TOLERANCE:
            continue                                     # close 가 안 튀었다 — 조정 불필요
        if abs((b[1] / a[1]) / float(이론) - 1) >= TOLERANCE:
            continue                                     # adj 가 이미 이었다
        out.append({"code": r["code"], "name": r["isu_abbrv"], "market": r["market"],
                    "종류": r["종류"], "전일": r["pd"], "당일": r["bas_dd"],
                    "전주식수": r["p"], "후주식수": r["s"], "배율": 이론,
                    "adj배": round(b[1] / a[1], 3)})
    return out


def 종목별_보정(자리들: List[Dict]) -> Dict[str, List[Tuple[str, Fraction]]]:
    """`{종목: [(적용경계일, 배율), …]}`. 경계일 **미만**의 행에 배율을 곱한다."""
    묶음: Dict[str, List[Tuple[str, Fraction]]] = {}
    for x in sorted(자리들, key=lambda x: (x["code"], x["당일"])):
        묶음.setdefault(x["code"], []).append((x["당일"], x["배율"]))
    return 묶음


def 적용(conn: sqlite3.Connection, 묶음, *, dry: bool) -> Tuple[int, List[Dict]]:
    """행마다 곱할 누적 배율을 구해 한 번에 쓴다.

    같은 종목에 자리가 둘 이상이면 **더 뒤 자리의 배율이 그 앞 전부에 함께 걸린다.**
    그래서 날짜 내림차순으로 훑으며 배율을 누적한다.
    """
    바뀐행 = 0
    되돌리기: List[Dict] = []
    for code, 자리들 in 묶음.items():
        rows = conn.execute(
            "SELECT bas_dd, adj_open, adj_high, adj_low, adj_close, adj_source "
            "FROM daily_price WHERE code=? ORDER BY bas_dd", (code,)).fetchall()
        # 각 행에 걸리는 누적 배율 — 그 행보다 **뒤에 있는** 모든 자리의 배율을 곱한다
        누적: Dict[str, Fraction] = {}
        곱 = Fraction(1)
        자리맵 = dict(자리들)
        for r in reversed(rows):
            d = r["bas_dd"]
            if d in 자리맵:
                # 이 날의 조정은 **그 앞**에 걸린다. 당일과 그 뒤는 그대로 둔다 —
                # back-adjust 는 최근 가격을 기준으로 과거를 옮기는 것이다.
                곱 *= 자리맵[d]
            elif 곱 != 1:
                누적[d] = 곱
        for r in rows:
            d = r["bas_dd"]
            k = 누적.get(d)
            if k is None or k == 1:
                continue
            새값 = {}
            for 칸 in ("adj_open", "adj_high", "adj_low", "adj_close"):
                v = r[칸]
                새값[칸] = None if v is None else float(k) * v
            # 🔴 출처를 남긴다 — "이 값이 어디서 왔나" 를 나중에 되짚을 수 있어야
            #    한다. 그냥 고치면 FDR 이 준 값과 우리가 고친 값이 구별되지 않는다.
            새출처 = (r["adj_source"] or "") + "+ca_fix"
            되돌리기.append({"code": code, "bas_dd": d,
                             "before": {**{c: r[c] for c in 새값},
                                        "adj_source": r["adj_source"]},
                             "after": {**새값, "adj_source": 새출처},
                             "factor": float(k)})
            if not dry:
                conn.execute(
                    "UPDATE daily_price SET adj_open=?, adj_high=?, adj_low=?, "
                    "adj_close=?, adj_source=? WHERE code=? AND bas_dd=?",
                    (새값["adj_open"], 새값["adj_high"], 새값["adj_low"],
                     새값["adj_close"], 새출처, code, d))
            바뀐행 += 1
    if not dry:
        conn.commit()
    return 바뀐행, 되돌리기


def main() -> int:
    ap = argparse.ArgumentParser(description="FDR 이 놓친 자본변동을 이어 붙인다")
    ap.add_argument("--apply", action="store_true",
                    help="실제로 고친다 (기본은 무엇을 고칠지만 본다)")
    args = ap.parse_args()

    conn = sqlite3.connect(krx_db_path())
    conn.row_factory = sqlite3.Row

    자리들 = 미조정_자리(conn)
    print(f"── 수정주가가 못 이은 자리 {len(자리들)} ──\n")
    if not 자리들:
        print("  고칠 것이 없다.")
        return 0
    for x in 자리들:
        print(f"  {x['code']} {x['name']:<16} {x['market']:<7} {x['종류']:<6} "
              f"{x['전일']}→{x['당일']} 주식수 {x['전주식수']:>12,}→{x['후주식수']:>12,} "
              f"· 곱할 배율 {float(x['배율']):.4f}")

    묶음 = 종목별_보정(자리들)
    print(f"\n  종목 {len(묶음)}곳 · 자리 {len(자리들)}")
    바뀐행, 되돌리기 = 적용(conn, 묶음, dry=not args.apply)
    print(f"  고칠 행 {바뀐행:,}")

    if not args.apply:
        print("\n  (미리보기다. 실제로 고치려면 --apply)")
        return 0

    stamp = datetime.now(_KST).strftime("%Y%m%dT%H%M%S")
    out = Path("reports") / f"fix_capital_reduction_{stamp}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "자리": [{k: (float(v) if isinstance(v, Fraction) else v)
                  for k, v in x.items()} for x in 자리들],
        "행": 되돌리기,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n✅ {바뀐행:,}행 고침 · 되돌리기 기록 {out}")
    print("   확인: python scripts/verify_base_info.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
