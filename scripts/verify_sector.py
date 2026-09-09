"""업종 스냅샷이 옳게 들어왔는지 **값으로** 확인한다.

    python scripts/verify_sector.py

행 수만 세는 검사로는 되짚은 기준일이 하루 어긋난 것도, 같은 날을 두 번 들여 덮어쓴 것도
못 잡는다. 그래서 스냅샷의 **모든 행**을 `daily_price` 와 맞댄다 — 종가와 시가총액이
그날 것과 같으면 기준일이 맞은 것이고, 다르면 되짚기부터 다시 본다.

보는 것
  1. 스냅샷 날짜·행 수 — 가이드의 18개와 같은가
  2. 🔴 종가·시가총액 전량 대조 — 기준일이 진짜 그날인가
  3. 업종 체계가 바뀐 지점 — 새 업종이 한 번 나타나면 계속 있는가
  4. `index_price` 지수명과 조인 — 이름 대조표 셋으로 다 붙는가
  5. 모델 파트 2안 시연 — 업종 상위 10 → 업종별 시총 상위 5 → 50종이 실제로 나오는가

## 실측 2026-09-03 에서 배운 것

**2024-07-01 은 지수가 생긴 날이지 종목 분류가 바뀐 날이 아니었다.** `IT 서비스`·
`부동산`·`오락·문화` **지수**는 `index_price` 에 2024-07-01 부터 있는데, 그날 스냅샷의
종목 분류는 옛 체계 그대로였다(광업만 사라져 23종). NAVER 가 `일반서비스` 에서
`IT 서비스` 로, KT&G 가 `기타제조` 에서 `음식료·담배` 로 옮긴 것은 2025-01-02 스냅샷에서
처음 보인다. 바뀐 날은 2024-07-02 ~ 2025-01-02 사이 어딘가다. 그래서 이 검사는
개편일을 못 박지 않고 **처음 나타난 스냅샷을 보고**하며, 한 번 나타난 업종이 다시
사라지면(뒤섞인 파일) 그때만 문제로 잡는다.
"""

from __future__ import annotations

import json
import sqlite3
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common.paths import krx_db_path  # noqa: E402
from supply import sector, universe  # noqa: E402

#: 가이드가 정한 스냅샷 날짜. 달력에서 뽑은 매년 첫 거래일 17개 + 2024-07-01(지수 신설일).
EXPECTED_DAYS = [
    "20100104", "20110103", "20120102", "20130102", "20140102", "20150102", "20160104",
    "20170102", "20180102", "20190102", "20200102", "20210104", "20220103", "20230102",
    "20240102", "20240701", "20250102", "20260102",
]
#: 2024 개편으로 새로 생긴 KOSPI 업종. 언제부터 종목에 붙는지는 스냅샷이 말해 준다.
NEW_SECTORS_2024 = {"IT 서비스", "부동산", "오락·문화"}
#: 업종지수가 없는 업종 — 이 둘 말고 조인이 안 되는 이름이 있으면 대조표를 넓혀야 한다.
NO_INDEX_SECTORS = {"농업, 임업 및 어업", "광업"}
#: 2안 시연에 쓰는 날. 개발구간 안이고 개편 전이라 옛 체계 그대로다.
DEMO_DAY, DEMO_AS_OF = "20240102", "2024-01-04"


def main() -> int:  # noqa: PLR0915 — 검사 다섯 절을 한 화면에 순서대로 보인다
    문제 = 0
    conn = sqlite3.connect(f"file:{krx_db_path().as_posix()}?mode=ro", uri=True)

    # ── 1. 날짜·행 수 ────────────────────────────────────────────────
    snaps = sector.snapshots()
    days = sorted(snaps["bas_dd"].unique())
    print(f"── 1. 스냅샷 ── {len(days)}장 · {len(snaps):,}행")
    빠짐 = sorted(set(EXPECTED_DAYS) - set(days))
    더있음 = sorted(set(days) - set(EXPECTED_DAYS))
    if 빠짐 or 더있음:
        문제 += 1
        print(f"  🔴 가이드와 다르다 — 빠짐 {빠짐} · 더 있음 {더있음}")
    else:
        print("  ✅ 가이드의 18개 날짜와 같다")
    행수 = snaps.groupby("bas_dd").size()
    print(f"  행 수 최소 {행수.min()}({행수.idxmin()}) · 최대 {행수.max()}({행수.idxmax()})")
    # 🔴 KRX 화면은 한 종목을 업종 둘에 싣는 때가 있다 (실측: KOSDAQ 17쌍 · KOSPI 0).
    #    원자료가 그런 것이라 격리하지 않고, supply.sector 가 결정적 규칙으로 하나를 고른다.
    #    여기서 볼 것은 "겹쳤나" 가 아니라 **"골라낸 뒤에도 남았나"** 다 — 남았다면 규칙이
    #    닿지 않은 새로운 모양이라는 뜻이고, 그때는 조인이 조용히 부푼다.
    겹침 = int(snaps.duplicated(["bas_dd", "code"]).sum())
    해소 = sector.resolve_duplicates(snaps)
    남은중복 = int(해소.duplicated(["bas_dd", "code"]).sum())
    if 겹침:
        겹친행 = snaps[snaps.duplicated(["bas_dd", "code"], keep=False)]
        조합 = sorted({" + ".join(sorted(g["sector_nm"]))
                      for _, g in 겹친행.groupby(["bas_dd", "code"])})
        print(f"  ℹ️ 원본에 (날짜,종목) 겹침 {겹침}쌍 — 시장 {sorted(겹친행['market'].unique())} · "
              f"종목 {겹친행['code'].nunique()}종 · 조합 {조합}")
        고른것 = sorted(해소.loc[해소["ambiguous"], "sector_nm"].unique())
        print(f"     → 규칙(같은 스냅샷 종목 수 많은 쪽 → 동수면 사전순)이 고른 업종: {고른것}")
    print(f"  {'✅' if 남은중복 == 0 else '🔴'} 고른 뒤 남은 (날짜, 종목) 중복 {남은중복}행")
    문제 += int(남은중복 > 0)

    # ── 2. 🔴 전량 값 대조 ───────────────────────────────────────────
    print("\n── 2. 종가·시가총액 전량 대조 (daily_price) ──")
    acc = pd.read_sql_query("SELECT payload FROM inbox_accepted WHERE kind = 'sector'", conn)
    rows = pd.DataFrame([json.loads(p) for p in acc["payload"]])
    rows["close"] = rows["close"].astype("Int64")
    rows["market_cap"] = rows["market_cap"].astype("Int64")
    price = pd.read_sql_query(
        "SELECT bas_dd, code, close, market_cap, market FROM daily_price "
        f"WHERE bas_dd IN ({','.join('?' * len(days))})", conn, params=days)
    m = rows.merge(price, on=["bas_dd", "code"], how="left", suffixes=("", "_db"))
    있음 = m["close_db"].notna()
    없음 = int((~있음).sum())
    종가다름 = int(((m["close"] != m["close_db"]) & 있음).sum())
    시총다름 = int(((m["market_cap"] != m["market_cap_db"]) & 있음).sum())
    시장다름 = int(((m["market"] != m["market_db"]) & 있음).sum())
    print(f"  스냅샷 {len(m):,}행 중 daily_price 에 없는 (날짜,종목) {없음:,}행")
    print(f"  {'✅' if 종가다름 == 0 else '🔴'} 종가가 다른 행 {종가다름:,}")
    print(f"  {'✅' if 시총다름 == 0 else '⚠️'} 시가총액이 다른 행 {시총다름:,}")
    print(f"  {'✅' if 시장다름 == 0 else '🔴'} 시장이 다른 행 {시장다름:,}")
    문제 += int(종가다름 > 0) + int(시장다름 > 0)
    if 없음:
        예 = m.loc[~있음, ["bas_dd", "code", "name"]].head(5).to_dict("records")
        print(f"     없는 예: {예}")
        print("     (그날 시세에 없는 종목 — 거래정지·정리매매일 수 있다. 격리하지 않는다)")
    if 시총다름:
        칸 = ["bas_dd", "code", "market_cap", "market_cap_db"]
        예 = m.loc[(m["market_cap"] != m["market_cap_db"]) & 있음, 칸].head(3)
        print(f"     시총 다른 예: {예.to_dict('records')}")

    # ── 3. 체계가 바뀐 지점 ───────────────────────────────────────────
    print("\n── 3. 업종 체계 — 새 업종이 처음 나타난 스냅샷 ──")
    # 🔴 두 시장을 합쳐서 세면 안 된다. KOSPI 와 KOSDAQ 은 **업종 체계가 다르고**
    #    (KOSDAQ 은 2024 개편에서 32종→24종) 이름도 일부만 겹친다. 합치면 "KOSPI 에서
    #    사라진 업종" 이 KOSDAQ 에 남아 있어 안 사라진 것처럼 보인다 — 실제로 `부동산` 이
    #    그렇게 보였다. 아래 개편 검사는 그래서 KOSPI 만 본다.
    시장들 = sorted(snaps["market"].unique())
    이름들_시장 = {
        m: {d: set(snaps.loc[(snaps["bas_dd"] == d) & (snaps["market"] == m), "sector_nm"])
            for d in days}
        for m in 시장들
    }
    for m in 시장들:
        print(f"  [{m}] 업종 수 흐름: " + " ".join(f"{d}:{len(이름들_시장[m][d])}" for d in days))
    이름들 = 이름들_시장.get("KOSPI", {d: set() for d in days})
    print("  (아래 2024 개편 검사는 KOSPI 체계 기준)")
    for 업종 in sorted(NEW_SECTORS_2024):
        있는날 = [d for d in days if 업종 in 이름들[d]]
        if not 있는날:
            print(f"  ⚠️ {업종}: 어느 스냅샷에도 없다")
            continue
        처음 = 있는날[0]
        # 한 번 나타난 뒤 사라지면 파일이 뒤섞인 것이다
        뒤 = [d for d in days if d > 처음]
        빠진 = [d for d in 뒤 if 업종 not in 이름들[d]]
        표시 = "✅" if not 빠진 else "🔴"
        문제 += int(bool(빠진))
        print(f"  {표시} {업종}: {처음} 부터" + (f" — 그 뒤 {빠진} 에서 빠짐" if 빠진 else ""))
    for m in 시장들:
        이름 = 이름들_시장[m]
        사라진 = [(d0, d1, sorted(이름[d0] - 이름[d1]))
                  for d0, d1 in zip(days, days[1:], strict=False) if 이름[d0] - 이름[d1]]
        for d0, d1, s in 사라진:
            print(f"  ℹ️ [{m}] {d0}→{d1} 사라진 업종 {s}")

    # ── 4. 지수명 조인 ───────────────────────────────────────────────
    print("\n── 4. index_price 업종지수와 조인 ──")
    조인문제 = 0
    for m in 시장들:
        idx = pd.read_sql_query(
            "SELECT DISTINCT bas_dd, index_name FROM index_price WHERE index_class = ?",
            conn, params=(m,))
        if idx.empty:
            # 🔴 지수를 안 받았을 뿐 스냅샷이 틀린 게 아니다. 문제로 세지 않고 할 일을 적는다.
            #    다만 **받는 명령을 여기 적지 않는다** — 그 명령이 2026-09-07 에 KOSPI
            #    업종지수 40,324행을 덮어썼다. 기본키 (bas_dd, index_name) 에 시장이 없어
            #    두 시장이 같은 이름(건설·금속·화학…)을 두고 서로를 지운다.
            print(f"  ⚠️ [{m}] index_price 에 이 시장 지수가 0종이라 조인을 건너뛴다")
            print(f"     🔴 지금은 받으면 안 된다 — {m} 을 받으면 같은 이름의 KOSPI 업종지수를")
            print("        덮어쓴다. 기본키에 index_class 를 넣는 마이그레이션(v13)이 먼저다")
            print("        (ingest/store/krx_index.py 의 _시장가드 가 실제로 막고 있다)")
            continue
        if m != "KOSPI":
            # `index_name_for` 대조표는 KOSPI 업종지수를 보고 만든 것이다. 다른 시장은
            # 아직 대조표가 없어 이름이 안 맞는 것이 정상이다 — 알리되 문제로 세지 않는다.
            안붙은날 = [d for d in days
                       if {n for n in 이름들_시장[m][d]
                           if sector.index_name_for(n)
                           not in set(idx.loc[idx["bas_dd"] == d, "index_name"])}]
            print(f"  ⚠️ [{m}] 대조표가 아직 없다 — "
                  f"{len(안붙은날)}/{len(days)}장에서 안 붙는 이름이 있다")
            continue
        for day in days:
            지수들 = set(idx.loc[idx["bas_dd"] == day, "index_name"])
            안붙음 = {n for n in 이름들_시장[m][day]
                    if sector.index_name_for(n) not in 지수들} - NO_INDEX_SECTORS
            if 안붙음:
                조인문제 += 1
                print(f"  🔴 [{m}] {day} 지수 없는 업종: {sorted(안붙음)}")
        if not 조인문제:
            print(f"  ✅ [{m}] {len(days)}장 전부 — 농업·광업 말고는 지수명이 다 붙는다")
    문제 += int(조인문제 > 0)

    # ── 5. 2안 시연 ──────────────────────────────────────────────────
    print(f"\n── 5. 모델 파트 2안 ({DEMO_DAY}) — 업종 상위 10 → 업종별 시총 상위 5 ──")
    caps = pd.read_sql_query(
        "SELECT index_name, market_cap FROM index_price "
        "WHERE bas_dd = ? AND index_class = 'KOSPI'", conn, params=(DEMO_DAY,))
    업종지수 = caps[~caps["index_name"].str.startswith("코스피")].dropna()
    # 🔴 제조·금융은 여러 업종의 합이라 넣으면 같은 종목을 두 번 센다 (supply.sector 참조)
    후보지수 = 업종지수[~업종지수["index_name"].isin(sector.UMBRELLA_INDICES)]
    상위10 = 후보지수.nlargest(10, "market_cap")
    보통주 = universe.common_stocks(DEMO_DAY, as_of=DEMO_AS_OF)
    붙인 = sector.attach_industry(보통주, as_of=DEMO_AS_OF)
    붙인["index_name"] = 붙인["industry"].map(sector.index_name_for)
    뽑힌 = []
    for _, r in 상위10.iterrows():
        후보 = 붙인[붙인["index_name"] == r["index_name"]].nlargest(5, "market_cap")
        뽑힌.append(후보)
        print(f"  {r['index_name']:<10} 지수시총 {r['market_cap'] / 1e12:6.1f}조 → "
              f"{', '.join(후보['name'].head(5))}")
    총 = pd.concat(뽑힌)
    없는보통주 = int(붙인["industry"].isna().sum())
    print(f"  → {len(총)}종 (묶음 지수 {sorted(sector.UMBRELLA_INDICES)} 제외) · "
          f"industry 없는 보통주 {없는보통주}/{len(붙인)}")
    if 없는보통주:
        print(f"     예: {붙인.loc[붙인['industry'].isna(), 'name'].head(5).tolist()}")

    conn.close()
    print(f"\n── 판정 ── {'✅ 이상 없음' if 문제 == 0 else f'🔴 문제 {문제}건'}")
    return 1 if 문제 else 0


if __name__ == "__main__":
    raise SystemExit(main())
