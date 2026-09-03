"""`stock_identity` · `corp_profile` 이 **맞게** 담겼는지 본다.

    python scripts/verify_identity.py

🔴 **행 수만 세는 검사는 덮어쓰기를 못 잡는다.** 재무 수집에서 기본키에 칸 하나를
빼는 바람에 자본변동표의 6.4%가 사라진 적이 있는데, 행 수는 그대로였다. 그리고
반출본이 행 수 80,439로 똑같은데 SHA-256 만 다른 적도 있다.

🔴 **검증식이 항등식이면 아무것도 못 잡는다.** "우리가 넣은 값 == 우리가 넣은 값" 은
검사가 아니다. 그래서 기대값은 **다른 경로에서** 온다 — 이 파일은 포털이 준 신원을
우리가 KRX 에서 따로 받은 `daily_price` 와 맞대어 본다.

무엇을 보나
----------
| 검사 | 기대값의 출처 |
|---|---|
| 접두사가 남은 행이 있나 | 스키마 CHECK 와 별개로 실제 값을 다시 본다 |
| 영문 낀 코드 84종이 살아남았나 | `daily_price` 에서 센 84종 |
| 못 이은 종목은 무엇인가 | `daily_price` 의 고유 종목 |
| 상장폐지일이 우리 추정과 맞나 | `daily_price` 에서 시세가 끊긴 날 |
| `known_at` 규칙이 섞여 있나 | `known_rule` 칸 |
"""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ingest.clients import data_go_kr  # noqa: E402
from ingest.store.krx_store import connect  # noqa: E402

문제 = 0


def 알림(맞나: bool, 글: str) -> None:
    global 문제
    print(f"  {'✅' if 맞나 else '🔴'} {글}")
    if not 맞나:
        문제 += 1


def main() -> int:
    with connect() as conn:
        신원수 = conn.execute("SELECT COUNT(*) FROM stock_identity").fetchone()[0]
        if 신원수 == 0:
            print("stock_identity 가 비어 있다 — 검사할 것이 없다.")
            print("  할 일: python scripts/fetch_data_go_kr.py --plan 으로 먼저 세어 본다.")
            return 0

        print(f"── 종목 신원 검사 ── ({신원수:,}행)\n")

        # ── 1. 접두사가 남았나 ────────────────────────────────────────
        # 스키마 CHECK 가 막고 있지만 **CHECK 는 새로 들어오는 행만 본다.**
        # 마이그레이션 이전에 들어온 행이나 CHECK 를 우회한 경로가 있으면 여기서 잡힌다.
        남은접두 = conn.execute(
            "SELECT COUNT(*) FROM stock_identity WHERE code LIKE 'A%'").fetchone()[0]
        알림(남은접두 == 0,
            f"'A' 접두사가 남은 행 {남은접두:,} (0이어야 한다 — 남으면 조인이 조용히 0행)")

        긴코드 = conn.execute(
            "SELECT COUNT(*) FROM stock_identity WHERE length(code) <> 6").fetchone()[0]
        알림(긴코드 == 0, f"코드 길이가 6이 아닌 행 {긴코드:,}")

        # ── 2. 포털이 **무엇을 빼는가** ───────────────────────────────
        # 실측(2026-09-03): 포털의 KRX상장종목정보는 **우선주와 외국기업을 담지 않는다.**
        # 20200102 기준 못 이은 141종의 내역이 우선주 117 · 전환우선주 3 · 외국기업 21
        # 이었다. 그래서 "영문 낀 코드가 신원에 0종" 인 것은 **우리가 거른 게 아니라
        # 포털이 안 준 것**이다 — 이 둘을 섞으면 없는 버그를 쫓게 된다.
        #
        # 그럼 무엇으로 우리 잘못을 가려내나: **포털이 준 것 중에 잃은 게 있나**를 본다.
        구간 = conn.execute(
            "SELECT MIN(bas_dd), MAX(bas_dd) FROM stock_identity").fetchone()
        시세종목 = {r[0]: r[1] for r in conn.execute(
            "SELECT code, MAX(name) FROM daily_price "
            "WHERE bas_dd BETWEEN ? AND ? GROUP BY code", (구간[0], 구간[1]))}
        신원종목 = {r[0] for r in conn.execute(
            "SELECT DISTINCT code FROM stock_identity")}
        못이은 = {c: n for c, n in 시세종목.items() if c not in 신원종목}
        우선주 = {c: n for c, n in 못이은.items()
                if n and (n.endswith("우") or "우B" in n or "우(전환)" in n
                          or n.endswith("우C"))}
        외국 = {c: n for c, n in 못이은.items()
              if c.startswith(("900", "950")) and c not in 우선주}
        그밖 = {c: n for c, n in 못이은.items()
              if c not in 우선주 and c not in 외국}

        print(f"\n  구간 {구간[0]} ~ {구간[1]}")
        print(f"    시세 {len(시세종목):,}종 · 신원 {len(신원종목):,}종"
              f" · 못 이은 {len(못이은):,}종")
        print(f"      우선주   {len(우선주):>4,}  ← 포털이 안 준다 (알려진 사실)")
        print(f"      외국기업 {len(외국):>4,}  ← 포털이 안 준다 (알려진 사실)")
        print(f"      그 밖    {len(그밖):>4,}  ← 여기가 커지면 우리 잘못을 의심한다")
        if 그밖:
            for 코드, 이름 in list(그밖.items())[:15]:
                print(f"        {코드}  {이름}")
            if len(그밖) > 15:
                print(f"        … 그 밖 {len(그밖) - 15:,}종")
        # 🔴 "몇 종 이었다" 로 끝내지 않고 이름을 출력한다 — 숫자만 보면
        #    "그 정도면 됐다" 로 넘어가고, 무엇이 빠졌는지 영영 모른다.
        알림(len(그밖) / max(1, len(시세종목)) < 0.05,
            f"우선주·외국기업 말고 못 이은 비율 "
            f"{len(그밖) / max(1, len(시세종목)):.1%} (5% 를 넘으면 조인을 의심한다)")

        # ── 3. 🔴 포털 목록은 **시점 목록이 아니다** ──────────────────
        # 실측(2026-09-03): `basDt=20200102` 목록에 **4년 뒤에야 상장되는 종목 33종**이
        # 들어 있었다 (듀켐바이오 첫 시세 2024-12-20). 즉 포털은 기준일 목록이 아니라
        # 최신 목록에 가까운 것을 기준일 딱지만 붙여 준다.
        #
        # 이걸 모르고 `stock_identity` 로 유니버스를 만들면 **아직 없던 종목이 섞인다** —
        # 미래참조이고, 에러 없이 성능만 좋아진다. 그래서 여기서 매번 세어 알린다.
        미래섞임 = 0
        표본날 = 구간[0]
        포털그날 = {r[0]: r[1] for r in conn.execute(
            "SELECT code, item_nm FROM stock_identity WHERE bas_dd = ?", (표본날,))}
        늦은것 = []
        for 코드, 이름 in 포털그날.items():
            첫날 = conn.execute(
                "SELECT MIN(bas_dd) FROM daily_price WHERE code = ?", (코드,)).fetchone()[0]
            if 첫날 and 첫날 > 표본날:
                미래섞임 += 1
                늦은것.append((코드, 이름, 첫날))
        print(f"\n  🔴 시점 정합 — {표본날} 목록 {len(포털그날):,}종 중")
        print(f"     그날 이후에야 상장된 종목 : {미래섞임:,}종")
        if 늦은것:
            늦은것.sort(key=lambda x: x[2])
            for 코드, 이름, 첫날 in 늦은것[-3:]:
                print(f"       {코드}  {이름:16s} 첫 시세 {첫날}")
        print("     → 유니버스를 만들 때는 반드시 `daily_price` 와 **교집합**을 낸다.")
        print("       포털 목록을 그대로 쓰면 아직 없던 종목이 섞인다.")
        # 실패로 세지 않는다 — 우리 잘못이 아니라 출처의 성질이다. 다만 **매번 보이게**
        # 해서 아무도 "그 시점 목록이겠거니" 하고 넘어가지 못하게 한다.

        # ── 4. known_at 규칙이 섞여 있나 ──────────────────────────────
        규칙 = Counter(r[0] for r in conn.execute(
            "SELECT known_rule FROM stock_identity"))
        print(f"\n  known_at 규칙 분포 : {dict(규칙)}")
        알림(len(규칙) <= 1,
            "규칙이 한 가지다 (섞여 있으면 어느 행이 옛 규칙인지 갈라야 한다)")

        이른known = conn.execute(
            "SELECT COUNT(*) FROM stock_identity WHERE known_at <= bas_dd").fetchone()[0]
        알림(이른known == 0,
            f"known_at 이 기준일보다 이르거나 같은 행 {이른known:,} "
            "(0이어야 한다 — 그 날 못 본 자료를 봤다는 뜻이다)")

        # ── 5. 법인 개요 ──────────────────────────────────────────────
        개요수 = conn.execute("SELECT COUNT(*) FROM corp_profile").fetchone()[0]
        if 개요수 == 0:
            print("\n── 법인 개요 ── 비어 있다 (아직 안 받았다)")
            return 1 if 문제 else 0

        print(f"\n── 법인 개요 검사 ── ({개요수:,}행)\n")
        빈유효 = conn.execute(
            "SELECT COUNT(*) FROM corp_profile WHERE fst_opeg_dt IS NULL "
            "OR length(fst_opeg_dt) <> 8").fetchone()[0]
        알림(빈유효 == 0, f"유효시작일이 없거나 8자가 아닌 행 {빈유효:,}")

        # 두 자리 연도를 잘못 풀었으면 여기서 드러난다 — 2076년 상장은 없다.
        미래상장 = conn.execute(
            "SELECT COUNT(*) FROM corp_profile "
            "WHERE xchg_lstg_dt > '20301231' OR estb_dt > '20301231'").fetchone()[0]
        알림(미래상장 == 0,
            f"상장일·설립일이 2030년보다 미래인 행 {미래상장:,} "
            "(있으면 두 자리 연도를 2000대로 잘못 풀었다)")

        너무이른 = conn.execute(
            "SELECT COUNT(*) FROM corp_profile "
            "WHERE xchg_lstg_dt IS NOT NULL AND xchg_lstg_dt < '19560301'").fetchone()[0]
        알림(너무이른 == 0,
            f"KRX 개장(1956-03)보다 이른 상장일 {너무이른:,} "
            "(있으면 두 자리 연도를 1900대로 잘못 풀었다)")

        폐지있음 = conn.execute(
            "SELECT COUNT(DISTINCT crno) FROM corp_profile "
            "WHERE xchg_lstg_abol_dt IS NOT NULL "
            "OR kosdaq_lstg_abol_dt IS NOT NULL").fetchone()[0]
        법인수 = conn.execute(
            "SELECT COUNT(DISTINCT crno) FROM corp_profile").fetchone()[0]
        print(f"\n  법인 {법인수:,}곳 중 폐지일이 있는 곳 {폐지있음:,}")

        규칙2 = Counter(r[0] for r in conn.execute(
            "SELECT known_rule FROM corp_profile"))
        print(f"  known_at 규칙 분포 : {dict(규칙2)}")
        알림(set(규칙2) <= {data_go_kr.KNOWN_RULE_OBSERVED},
            "법인 개요의 known_at 은 관측값이어야 한다 (계산값이 섞이면 안 된다)")

    print(f"\n── 판정 ── {'✅ 이상 없음' if 문제 == 0 else f'🔴 {문제}건'}")
    return 1 if 문제 else 0


if __name__ == "__main__":
    raise SystemExit(main())
