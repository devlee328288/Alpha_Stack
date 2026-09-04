"""`stock_base_info` 를 **전량으로** 검사한다. 표본이 통과시킨 것을 여기서 잡는다.

    python scripts/verify_base_info.py

## 왜 전량인가

우선주 판별을 이름 추측에서 정본으로 바꿨다. 그 근거를 만들 때는 **세 날짜만**
봤다 — 20150102 · 20200102 · 20260901. 그때 나온 어긋남이 7종이었는데,
그건 "적어도 7종" 이지 "정확히 7종" 이 아니다. **전 구간을 보니 10종 · 10,190행이다.**

🔴 더 중요한 것: 20260831 유가 943종을 **전수로** 세도 어긋남이 **0건**이었다.
   하루만 보면 안 보인다. 이 검사는 **9,220,879행 전부**를 본다.

## 무엇을 재나

1. 이름 추측 vs 정본 — 전 구간에서 몇 종이 어긋나나
2. `daily_price` 와의 커버리지 — 못 이은 종목이 몇이나 되나
3. `known_at` — 규칙이 하나인가, 기준일보다 이른 행이 있나
4. 주권종류 — 빈 값이 있나 (있으면 그 종목은 통째로 유니버스에서 빠진다)
5. 상장일 — `corp_profile` 과 교차검증
6. 종목코드 — 앞 4자리가 숫자인가 (5·6번째 영문은 정상이다)
7. 🔴 자리표시자 — `00010101` · `99991231` · `0000000000000` 이 값인 척 들어와 있나
8. 상장주식수·액면가 — 0 이나 빈 값이 시가총액을 조용히 0 으로 만들지 않나
9. 상장주식수 급변 — 10배 넘게 뛴 자리가 액면가 변화로 설명되나

## 값이 없는 것과 값이 0 인 것은 다르다

`parval`(액면가)이 `'0'` 인 행이 7,616 있다. 결측처럼 보이지만 **주식예탁증권**이라
액면가라는 개념 자체가 없는 것이다(원주는 해외에 있고 우리가 거래하는 것은 증서다).
`'무액면'` 이라고 글자로 적힌 것도 73,713행 있다. 둘 다 정상이다.

그래서 이 검사는 액면가가 0 인지만 보지 않고 **증권종류를 함께 본다** — `주권` 인데
액면가가 0 이면 그때가 진짜 문제다(실측 0행).
"""

from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common.paths import krx_db_path  # noqa: E402
from ingest.clients.krx_data import COMMON_STOCK_KIND  # noqa: E402


#: 지금까지 쓰던 이름 추측. **이 함수를 고치지 않는다** — 무엇과 비교하는지가
#: 이 검사의 전부다. `scripts/verify_identity.py` 의 규칙과 같은 것을 쓴다.
def 이름추측_우선주(name: str) -> bool:
    n = (name or "").strip()
    return bool(n) and (n.endswith("우") or "우B" in n or "우(전환)" in n
                        or n.endswith("우C"))


def main() -> int:
    db = sqlite3.connect(krx_db_path())
    db.row_factory = sqlite3.Row
    문제 = []

    print(f"── stock_base_info 전량 검사 ── ({krx_db_path().name})\n")

    # ── 0. 규모 ───────────────────────────────────────────────
    r = db.execute(
        "SELECT COUNT(*), COUNT(DISTINCT bas_dd), COUNT(DISTINCT code), "
        "MIN(bas_dd), MAX(bas_dd) FROM stock_base_info").fetchone()
    print(f"  {r[0]:,}행 · {r[1]:,}거래일 · {r[2]:,}종 ({r[3]} ~ {r[4]})")

    달력 = db.execute(
        "SELECT COUNT(DISTINCT bas_dd) FROM trading_calendar").fetchone()[0]
    빠진날 = 달력 - r[1]
    print(f"  거래일 달력 {달력:,}일 중 {빠진날}일이 비어 있다")

    # "몇 일이 비었다" 로 끝내면 사람이 그 다음을 못 한다. **어느 날 어느 시장인지**
    # 이름을 대야 이어받을지 미룰지 판단할 수 있다.
    #
    # 시세의 마지막 거래일은 **다음 거래일을 몰라 known_at 을 못 낸다.** 없는 날을
    # 지어내지 않는 것이 옳으므로 이건 결함이 아니라 경계다. 시세가 하루 더 늘면
    # 저절로 채워진다. 그 하루만 비는 것은 통과시키고, 그보다 많으면 잡는다.
    마지막시세 = db.execute("SELECT MAX(bas_dd) FROM trading_calendar").fetchone()[0]
    안받은 = db.execute("""
        SELECT c.bas_dd, c.market FROM (
          SELECT DISTINCT bas_dd, market FROM trading_calendar
           WHERE market IN ('KOSPI', 'KOSDAQ')) c
        LEFT JOIN (SELECT DISTINCT bas_dd, market FROM stock_base_info) b
          ON b.bas_dd = c.bas_dd AND b.market = c.market
        WHERE b.bas_dd IS NULL ORDER BY c.bas_dd""").fetchall()
    경계 = [x for x in 안받은 if x["bas_dd"] == 마지막시세]
    진짜 = [x for x in 안받은 if x["bas_dd"] != 마지막시세]
    if 경계:
        쌍 = ", ".join(f"{x['bas_dd']} {x['market']}" for x in 경계)
        print(f"  ✅ 달력 경계라 미룬 {len(경계)}쌍: {쌍}")
        print(f"     ({마지막시세} 은 시세의 마지막 날이다 — 다음 거래일을 알아야 "
              "known_at 을 낸다. 시세가 하루 늘면 저절로 채워진다)")
    if 진짜:
        for x in 진짜[:10]:
            print(f"  🔴 안 받은 쌍: {x['bas_dd']} {x['market']}")
        if len(진짜) > 10:
            print(f"     … 외 {len(진짜) - 10}쌍")
        문제.append(f"안 받은 (날짜,시장) 쌍 {len(진짜)} — "
                    "python scripts/fetch_base_info.py 로 이어받는다")
    if not 안받은:
        print("  ✅ 안 받은 (날짜,시장) 쌍 0")

    # ── 1. 🔴 이름 추측 vs 정본 (이 파일의 존재 이유) ────────────
    print("\n── 1. 이름 추측 vs 정본 (전 구간) ──")
    어긋남 = db.execute(f"""
        SELECT code, isu_abbrv, kind_stkcert_tp_nm,
               MIN(bas_dd) AS 처음, MAX(bas_dd) AS 마지막, COUNT(*) AS 일수
          FROM stock_base_info
         WHERE (kind_stkcert_tp_nm = '{COMMON_STOCK_KIND}') = (
                   isu_abbrv LIKE '%우' OR isu_abbrv LIKE '%우B%'
                   OR isu_abbrv LIKE '%우(전환)%' OR isu_abbrv LIKE '%우C')
         GROUP BY code, isu_abbrv, kind_stkcert_tp_nm
         ORDER BY 일수 DESC""").fetchall()
    # SQL 의 LIKE 로 1차로 좁히고, 판정은 파이썬 함수로 다시 한다 — 두 규칙이
    # 갈라지면 이 검사 자체가 거짓말을 하므로 최종 판정은 한 곳에서만 한다.
    진짜어긋남 = [
        row for row in 어긋남
        if 이름추측_우선주(row["isu_abbrv"]) != (row["kind_stkcert_tp_nm"] != COMMON_STOCK_KIND)
    ]
    총일수 = sum(row["일수"] for row in 진짜어긋남)
    print(f"  어긋난 (종목,이름) 조합 : {len(진짜어긋남):,}")
    print(f"  그 조합이 걸린 종목-일수 : {총일수:,}행")
    for row in 진짜어긋남:
        방향 = "이름추측=우선주 · 정본=보통주" if row["kind_stkcert_tp_nm"] == COMMON_STOCK_KIND \
               else "이름추측=보통주 · 정본=우선주"
        print(f"    {row['code']} {row['isu_abbrv']:<16} {방향}"
              f"  {row['처음']}~{row['마지막']} ({row['일수']:,}일)")
    if 진짜어긋남:
        print(f"  → 이름 추측을 쓰면 이 {총일수:,}행이 잘못 분류된다. 정본을 쓴다.")

    # ── 2. daily_price 와의 커버리지 ──────────────────────────
    #
    # ⚠️ `LEFT JOIN` 으로 9,220,879행을 두 번 훑으면 10분이 넘는다(실측 2026-09-03).
    #    두 표 다 기본키가 `(bas_dd, code)` 라 이미 그 순서로 정렬돼 있으므로,
    #    `EXCEPT` 를 쓰면 인덱스를 **병합 스캔**해서 훨씬 빠르다.
    print("\n── 2. 시세와의 커버리지 ──")
    상한 = db.execute("SELECT MAX(bas_dd) FROM stock_base_info").fetchone()[0]
    for market in ("KOSPI", "KOSDAQ"):
        시세수 = db.execute(
            "SELECT COUNT(*) FROM daily_price WHERE market=? AND bas_dd<=?",
            (market, 상한)).fetchone()[0]
        못이은 = db.execute("""
            SELECT COUNT(*) FROM (
              SELECT bas_dd, code FROM daily_price WHERE market=? AND bas_dd<=?
              EXCEPT
              SELECT bas_dd, code FROM stock_base_info)""",
            (market, 상한)).fetchone()[0]
        비율 = 못이은 / 시세수 if 시세수 else 0
        표 = "✅" if 비율 < 0.001 else "🔴"
        print(f"  {표} {market:<7} 시세 {시세수:>9,}행 · 못 이은 행 {못이은:>7,} ({비율:.3%})")
        if 비율 >= 0.001:
            문제.append(f"{market} 에서 시세의 {비율:.2%}가 기본정보와 안 이어진다")

    # 반대 방향 — 기본정보에만 있고 시세에 없는 것 (거래정지 등이라 정상일 수 있다)
    한쪽 = db.execute("""
        SELECT COUNT(*) FROM (
          SELECT bas_dd, code FROM stock_base_info
          EXCEPT
          SELECT bas_dd, code FROM daily_price)""").fetchone()[0]
    print(f"     기본정보에만 있는 행 {한쪽:,} (상장은 됐으나 그날 시세가 없는 경우)")

    # ── 3. known_at ──────────────────────────────────────────
    print("\n── 3. known_at ──")
    규칙 = dict(db.execute(
        "SELECT known_rule, COUNT(*) FROM stock_base_info GROUP BY known_rule").fetchall())
    print(f"  규칙 분포 : {규칙}")
    if len(규칙) != 1:
        문제.append(f"known_at 규칙이 {len(규칙)}가지다 — 어느 행이 옛 규칙인지 갈라야 한다")
    else:
        print("  ✅ 규칙이 한 가지다 (섞여 있으면 어느 행이 옛 규칙인지 갈라야 한다)")

    이른행 = db.execute(
        "SELECT COUNT(*) FROM stock_base_info WHERE known_at <= bas_dd").fetchone()[0]
    print(f"  {'✅' if 이른행 == 0 else '🔴'} known_at 이 기준일보다 이르거나 같은 행 {이른행:,} "
          "(0이어야 한다 — 그 날 못 본 자료를 봤다는 뜻이다)")
    if 이른행:
        문제.append(f"known_at 이 기준일 이하인 행 {이른행:,} — 미래참조다")

    빈known = db.execute(
        "SELECT COUNT(*) FROM stock_base_info WHERE known_at IS NULL OR known_at = ''"
    ).fetchone()[0]
    if 빈known:
        문제.append(f"known_at 이 빈 행 {빈known:,} — as_of 가 영원히 못 거른다")

    # ── 4. 주권종류 ──────────────────────────────────────────
    print("\n── 4. 주권종류 ──")
    종류 = db.execute(
        "SELECT kind_stkcert_tp_nm, COUNT(DISTINCT code), COUNT(*) "
        "FROM stock_base_info GROUP BY kind_stkcert_tp_nm ORDER BY 2 DESC").fetchall()
    for k, 종수, 행수 in 종류:
        표시 = k if k else "🔴 (빈 값)"
        꼬리 = " ← 유니버스" if k == COMMON_STOCK_KIND else ""
        print(f"  {표시:<12} {종수:>6,}종 · {행수:>9,}행{꼬리}")
        if not k:
            문제.append(f"주권종류가 빈 행 {행수:,} — 그 종목은 통째로 유니버스에서 빠진다")

    # 한 종목의 주권종류가 도중에 바뀌었나 (바뀌었다면 시점별로 달리 판정해야 한다)
    바뀐것 = db.execute(
        "SELECT code, COUNT(DISTINCT kind_stkcert_tp_nm) c FROM stock_base_info "
        "GROUP BY code HAVING c > 1").fetchall()
    print(f"  {'✅' if not 바뀐것 else '⚠️'} 주권종류가 도중에 바뀐 종목 {len(바뀐것):,}")
    for row in 바뀐것[:5]:
        이력 = db.execute(
            "SELECT kind_stkcert_tp_nm, MIN(bas_dd), MAX(bas_dd) FROM stock_base_info "
            "WHERE code=? GROUP BY kind_stkcert_tp_nm", (row["code"],)).fetchall()
        print(f"     {row['code']}: " + " → ".join(f"{k}({a}~{b})" for k, a, b in 이력))

    # ── 5. 상장일 교차검증 ────────────────────────────────────
    print("\n── 5. 상장일 — corp_profile 과 교차검증 ──")
    교차 = db.execute("""
        SELECT COUNT(*) AS 맞대본,
               SUM(CASE WHEN b.list_dd <> p.xchg_lstg_dt THEN 1 ELSE 0 END) AS 다름
          FROM (SELECT DISTINCT code, list_dd FROM stock_base_info
                 WHERE market='KOSPI' AND list_dd IS NOT NULL AND list_dd <> '') b
          JOIN stock_identity i ON i.code = b.code
          JOIN corp_profile p ON p.crno = i.crno
         WHERE p.xchg_lstg_dt IS NOT NULL AND p.xchg_lstg_dt <> ''""").fetchone()
    if 교차["맞대본"]:
        비율 = (교차["다름"] or 0) / 교차["맞대본"]
        print(f"  맞대 본 쌍 {교차['맞대본']:,} · 다른 것 {교차['다름'] or 0:,} ({비율:.2%})")
        print("  ⚠️ 두 출처가 다른 것 자체는 오류가 아니다 — 재상장·이전상장에서 갈린다.")
    else:
        print("  맞대 볼 쌍이 없다 (stock_identity·corp_profile 을 먼저 받는다)")

    # ── 6. 코드 형식 ─────────────────────────────────────────
    print("\n── 6. 종목코드 ──")
    비숫자 = db.execute(
        "SELECT COUNT(DISTINCT code) FROM stock_base_info "
        "WHERE code GLOB '*[^0-9]*'").fetchone()[0]
    print(f"  숫자가 아닌 글자가 섞인 코드 {비숫자:,}종 "
          "(있는 게 정상이다 — 5·6번째 자리에 영문이 오는 신형우선주)")
    앞자리 = db.execute(
        "SELECT COUNT(*) FROM stock_base_info "
        "WHERE substr(code,1,4) GLOB '*[^0-9]*'").fetchone()[0]
    print(f"  {'✅' if 앞자리 == 0 else '🔴'} 앞 4자리가 숫자가 아닌 행 {앞자리:,}")
    if 앞자리:
        문제.append(f"앞 4자리가 숫자가 아닌 행 {앞자리:,} — 접두사가 안 떨어졌을 수 있다")

    # ── 7. 🔴 자리표시자 ─────────────────────────────────────
    #
    # 자리표시자는 **모양이 멀쩡하다.** `00010101` 은 여덟 자리 숫자고, `99991231` 은
    # 달·일까지 말이 된다. 형식 검사를 전부 통과하고 값인 척 앉아 있다가, 정렬하면
    # 맨 앞이나 맨 뒤로 튀어 "가장 오래된 상장" 이나 "가장 늦은 폐지" 가 된다.
    #
    # 공공데이터포털 쪽에서 실제로 겪었다 — `00010101`("해당 없음")이 서기 1년으로
    # 들어와 있었다. KRX 쪽에는 지금 0건이지만, **0건인 것과 안 본 것은 다르다.**
    print("\n── 7. 자리표시자 ──")
    자리표시자 = ("00010101", "99991231", "0000000000000", "1111111111111",
                  "00000000", "19000101", "99999999")
    날짜꼴 = ("bas_dd", "list_dd", "known_at")
    걸린것 = []
    for 칸 in 날짜꼴:
        구멍 = ",".join("?" * len(자리표시자))
        for 값, 수 in db.execute(
                f"SELECT {칸}, COUNT(*) FROM stock_base_info WHERE {칸} IN ({구멍}) "
                f"GROUP BY {칸}", 자리표시자).fetchall():
            걸린것.append((칸, 값, 수))
    if 걸린것:
        for 칸, 값, 수 in 걸린것:
            print(f"  🔴 {칸} = {값!r} 인 행 {수:,}")
            문제.append(f"{칸} 에 자리표시자 {값!r} 가 {수:,}행 — 값인 척 들어와 있다")
    else:
        print(f"  ✅ 날짜꼴 {len(날짜꼴)}칸에서 자리표시자 {len(자리표시자)}종 모두 0건")
        print(f"     (본 것: {', '.join(날짜꼴)})")

    # 미래 날짜 — 자리표시자가 아니어도 "아직 안 온 날" 은 값일 수 없다
    미래 = db.execute(
        "SELECT COUNT(*) FROM stock_base_info WHERE list_dd > bas_dd").fetchone()[0]
    print(f"  {'✅' if 미래 == 0 else '🔴'} 상장일이 기준일보다 늦은 행 {미래:,} "
          "(그 날 아직 상장 안 된 종목이 시세에 있다는 뜻이다)")
    if 미래:
        문제.append(f"상장일 > 기준일 인 행 {미래:,} — 미래참조이거나 상장일이 틀렸다")

    # ── 8. 상장주식수 · 액면가 ───────────────────────────────
    #
    # 시가총액은 `close × list_shrs` 로 낸다. `list_shrs` 가 0 이면 시가총액이 0 이 되고,
    # 그 종목은 "시총 하위" 로 조용히 밀려 유니버스에서 빠진다. 에러가 안 난다.
    print("\n── 8. 상장주식수 · 액면가 ──")
    수량 = db.execute("""
        SELECT SUM(list_shrs IS NULL OR list_shrs = '') AS 빈,
               SUM(CAST(list_shrs AS INTEGER) <= 0) AS 영이하,
               MIN(CAST(list_shrs AS INTEGER)) AS 최소,
               MAX(CAST(list_shrs AS INTEGER)) AS 최대
          FROM stock_base_info""").fetchone()
    print(f"  {'✅' if not (수량['빈'] or 수량['영이하']) else '🔴'} "
          f"상장주식수 빈 값 {수량['빈']:,} · 0 이하 {수량['영이하']:,} "
          f"(범위 {수량['최소']:,} ~ {수량['최대']:,})")
    if 수량["빈"] or 수량["영이하"]:
        문제.append(f"상장주식수가 비었거나 0 이하인 행 {수량['빈'] + 수량['영이하']:,} "
                    "— 시가총액이 0 이 되어 유니버스에서 조용히 빠진다")

    # 액면가 0 은 결측이 아니다 — 증권종류를 함께 봐야 판정할 수 있다
    무액면 = dict(db.execute("""
        SELECT secugrp_nm, COUNT(*) FROM stock_base_info
         WHERE parval IN ('0', '무액면') OR parval IS NULL OR parval = ''
         GROUP BY secugrp_nm""").fetchall())
    print(f"  액면가가 0·무액면·빈 값인 행의 증권종류 : {무액면}")
    주권0 = db.execute("""
        SELECT COUNT(*) FROM stock_base_info
         WHERE secugrp_nm = '주권'
           AND (parval = '0' OR parval IS NULL OR parval = '')""").fetchone()[0]
    print(f"  {'✅' if 주권0 == 0 else '🔴'} 주권인데 액면가가 0·빈 값인 행 {주권0:,} "
          "(예탁증권·투자회사는 액면가 개념이 없어 정상이다)")
    if 주권0:
        문제.append(f"주권인데 액면가가 0·빈 값인 행 {주권0:,} — 결측을 0 으로 받았을 수 있다")

    # ── 9. 🔴 자본변동을 수정주가가 이었나 ───────────────────
    #
    # 상장주식수가 x 배가 되면 이론상 주가는 1/x 배가 된다 — 10:1 감자면 주가 10배,
    # 1:2 액면분할이면 주가 절반. 수정주가는 이 점프를 **이어 붙여 없애야** 한다.
    #
    # 🔴 처음에는 "액면가가 같이 바뀌었나" 로 판정했는데 그게 틀렸다. **감자는 액면가가
    #    그대로**여서(주식수만 줄고 액면가는 유지) 219건이 전부 '설명 안 됨' 으로
    #    쏟아졌고, 그 안에 진짜 결함 17건이 묻혔다. 액면가는 분할·병합만 가려낸다.
    #
    # 그래서 판정을 바꿨다 — `close` 가 이론 점프를 따라갔는데 `adj_close` 도 같이
    # 따라갔다면, 그 자리는 **조정이 안 된 것**이다.
    #
    # ⚠️ 임계값을 10배로 두면 5:1·3:1 감자를 통째로 놓친다. 2배로 내린다.
    print("\n── 9. 자본변동을 수정주가가 이었나 ──")
    후보 = db.execute("""
        WITH t AS (
          SELECT code, isu_abbrv, market, kind_stkcert_tp_nm AS 종류, bas_dd,
                 CAST(list_shrs AS INTEGER) AS s,
                 LAG(CAST(list_shrs AS INTEGER)) OVER w AS p,
                 LAG(bas_dd) OVER w AS pd
            FROM stock_base_info
          WINDOW w AS (PARTITION BY code ORDER BY bas_dd))
        SELECT code, isu_abbrv, market, 종류, pd, bas_dd, p, s
          FROM t WHERE p > 0 AND s > 0 AND (s * 1.0 / p >= 2 OR p * 1.0 / s >= 2)
         ORDER BY bas_dd""").fetchall()

    # 건마다 질의하면 2,000번을 왕복한다. 필요한 (종목,날짜)만 모아 한 번에 끌어온다.
    필요 = set()
    for r in 후보:
        필요.add((r["code"], r["pd"]))
        필요.add((r["code"], r["bas_dd"]))
    시세 = {}
    for row in db.execute("SELECT code, bas_dd, close, adj_close FROM daily_price"):
        키 = (row["code"], row["bas_dd"])
        if 키 in 필요:
            시세[키] = (row["close"], row["adj_close"])

    미조정, 이어짐, 무관, 값없음 = [], 0, 0, 0
    for r in 후보:
        a, b = 시세.get((r["code"], r["pd"])), 시세.get((r["code"], r["bas_dd"]))
        if not a or not b or not all([a[0], a[1], b[0], b[1]]):
            값없음 += 1
            continue
        이론 = r["p"] / r["s"]          # 주식수가 1/10 이면 주가는 10배
        if abs((b[0] / a[0]) / 이론 - 1) > 0.30:
            무관 += 1                   # close 가 안 튀었다 — 증자·전환 등이라 정상
            continue
        if abs((b[1] / a[1]) / 이론 - 1) < 0.30:
            미조정.append((r, b[1] / a[1]))
        else:
            이어짐 += 1

    print(f"  주식수가 2배 이상 변한 자리 {len(후보):,} · 시세 없음 {값없음:,}")
    print(f"    가격이 안 튀었다(증자·전환 등 조정이 필요 없다) {무관:,}")
    print(f"    가격이 튀었고 수정주가가 이었다              {이어짐:,}")
    if not 미조정:
        print("  ✅ 수정주가가 못 이은 자리 0")
    else:
        # 유니버스는 보통주만 담는다. 우선주가 섞여 있으면 영향 범위가 달라지므로 가른다.
        보통주 = [x for x in 미조정 if x[0]["종류"] == COMMON_STOCK_KIND]
        print(f"  🔴 수정주가가 못 이은 자리 {len(미조정):,} "
              f"(보통주 {len(보통주):,} · 우선주 등 {len(미조정) - len(보통주):,})")
        for r, 실제 in 미조정[:20]:
            print(f"     {r['code']} {r['isu_abbrv']:<16} {r['market']:<7} "
                  f"{r['pd']}→{r['bas_dd']} 주식수 {r['p']:>12,}→{r['s']:>12,} "
                  f"· adj 가 {실제:.2f}배 뛰었다 (이어졌다면 1 근처)")
        if len(미조정) > 20:
            print(f"     … 외 {len(미조정) - 20}자리")
        문제.append(
            f"자본변동을 수정주가가 못 이은 자리 {len(미조정):,} "
            "— 그 날 수익률이 몇 배로 읽힌다 (FDR 이 감자를 조정하지 않는다)")

    # ── 판정 ─────────────────────────────────────────────────
    print("\n── 판정 ──", end=" ")
    if 문제:
        print("🔴 손볼 것이 있다")
        for m in 문제:
            print(f"  · {m}")
        return 1
    print("✅ 이상 없음")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
