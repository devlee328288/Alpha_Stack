"""판정을 사람이 읽을 수 있는 글로 옮긴다 — `reports/inbox/` 에 남는 것.

**왜 JSON 만으로는 모자란가.** 판정 결과를 받을 사람은 팀원이고, 그 사람이 알고 싶은 것은
`{"rule": "high_ge_low", "rows": 1}` 이 아니라 *"내 파일 8행 중 3행이 안 들어갔고, 그중 하나는
고가가 저가보다 낮아서"* 다. 기계가 읽을 것과 사람이 읽을 것을 **둘 다** 낸다.

    reports/inbox/
      2026-09-01/
        ohlcv_stock_stock_dirty.json    ← 기계용 (전량 집계 + 표본 20건)
        ohlcv_stock_stock_dirty.md      ← 사람용 (무엇이 왜 안 들어갔나)
        요약.md                          ← 그날 들인 것 전부의 한 장 요약

🔴 **보고서에 자료를 싣지 않는다.** 이 저장소는 PUBLIC 이고, 시세는 KRX 이용약관 제11조 ②가
제3자 제공을 금지하며 뉴스 본문은 언론사 저작물이다. 그래서 보고서가 담는 것은 **판정과
표본뿐**이고, 표본조차 값이 실제로 바뀐 자리의 before/after 20건까지다. 원본 행은 `data/inbox/`
와 DB 에만 있고 둘 다 커밋되지 않는다.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from common.trading_calendar import KST

#: 보고서가 쌓이는 곳. 날짜 폴더로 나눠 그날 무엇을 들였는지 한눈에 본다.
REPORT_ROOT = Path("reports/inbox")


def report_dir(when: Optional[datetime] = None, root: Optional[Path] = None) -> Path:
    """그날의 보고서 폴더. 없으면 만든다."""
    stamp = (when or datetime.now(KST)).strftime("%Y-%m-%d")
    path = (root or REPORT_ROOT) / stamp
    path.mkdir(parents=True, exist_ok=True)
    return path


def _slug(source: str) -> str:
    """파일 이름을 보고서 이름으로. 경로 구분자와 공백만 눕힌다."""
    stem = Path(source).stem
    return "".join(ch if ch.isalnum() or ch in "-_가-힣" else "_" for ch in stem)[:60]


def _percent(part: int, whole: int) -> str:
    return f"{part / whole * 100:.2f}%" if whole else "—"


def _with_particle(word: str, with_batchim: str, without_batchim: str) -> str:
    """받침에 맞는 조사를 붙인다 — `종가는` · `시장구분이`.

    규격의 `title` 이 칸마다 달라서 조사를 하나로 못 박으면 *"종가 은"* 처럼 어색해진다.
    팀원이 읽는 글이라 이 정도는 맞춰 준다. 한글이 아니면 조사를 붙이지 않는다.
    """
    if not word:
        return word
    last = word[-1]
    if not ("가" <= last <= "힣"):
        return f"{word} {without_batchim}"
    has_batchim = (ord(last) - 0xAC00) % 28 != 0
    return word + (with_batchim if has_batchim else without_batchim)


# ==================================================
# 사람이 읽는 글
# ==================================================
def render_markdown(result, *, batch_id: Optional[str] = None,
                    contributor: Optional[str] = None) -> str:
    """판정 하나를 글로 옮긴다."""
    report = result.report
    lines: List[str] = []
    source_name = Path(result.source).name

    lines.append(f"# 반입 판정 — {source_name}")
    lines.append("")
    lines.append(f"- **종류**: `{result.kind}` (규격 v{report.get('schema_version')})")
    if contributor:
        lines.append(f"- **보낸 사람**: {contributor}")
    if batch_id:
        lines.append(f"- **묶음**: `{batch_id}`")
    lines.append(f"- **검사 시각**: {datetime.now(KST).strftime('%Y-%m-%d %H:%M KST')}")
    lines.append("")

    # ── 파일째 거부 ───────────────────────────────
    if result.rejected:
        lines.append("## ❌ 파일째 되돌렸습니다")
        lines.append("")
        lines.append("```")
        lines.append(result.rejected)
        lines.append("```")
        lines.append("")
        lines.append("행을 골라 담을 수 없는 종류의 문제라 파일 전체를 돌려보냅니다. "
                     "위 내용을 고쳐 다시 보내 주세요.")
        return "\n".join(lines) + "\n"

    total = result.rows_total
    accepted = len(result.accepted)
    quarantined = len(result.quarantined)

    # ── 한 줄 결론 ───────────────────────────────
    lines.append("## 결과")
    lines.append("")
    if quarantined == 0:
        lines.append(f"✅ **{total:,}행이 모두 들어왔습니다.**")
    elif accepted == 0:
        lines.append(f"❌ **{total:,}행이 모두 격리됐습니다.** 아래 사유를 봐 주세요.")
    else:
        lines.append(f"⚠️ **{total:,}행 중 {accepted:,}행이 들어오고 "
                     f"{quarantined:,}행({_percent(quarantined, total)})이 격리됐습니다.**")
    lines.append("")
    lines.append("| | 행 수 | 비율 |")
    lines.append("|---|---:|---:|")
    lines.append(f"| 들어옴 | {accepted:,} | {_percent(accepted, total)} |")
    lines.append(f"| 격리 | {quarantined:,} | {_percent(quarantined, total)} |")
    lines.append(f"| 합계 | {total:,} | 100% |")
    lines.append("")

    # ── 격리 사유 ────────────────────────────────
    reasons = report.get("quarantine_reasons") or []
    if reasons:
        lines.append("## 왜 격리됐나")
        lines.append("")
        lines.append("| 사유 | 행 수 | 무슨 뜻인가 |")
        lines.append("|---|---:|---|")
        notes = _rule_notes(result)
        for item in reasons:
            note = notes.get(item["rule"], "")
            note = note.replace("\n", " ").replace("|", "·")
            if len(note) > 120:
                note = note[:117] + "..."
            lines.append(f"| `{item['rule']}` | {item['rows']:,} | {note} |")
        lines.append("")

    # ── 우리가 고친 것 ───────────────────────────
    cleaners = report.get("cleaners") or []
    if cleaners:
        lines.append("## 우리가 손댄 것")
        lines.append("")
        lines.append("보낸 파일을 그대로 담지 않고 규격에 맞춰 다듬었습니다. "
                     "**무엇을 무엇으로 바꿨는지 전부 적습니다** — 값이 이상해 보일 때 "
                     "출처가 그런 것인지 우리가 그런 것인지 알 수 있어야 하니까요.")
        lines.append("")
        lines.append("| 칸 | 손질 | 바뀐 행 | 못 읽음 | 예 (앞 3건) |")
        lines.append("|---|---|---:|---:|---|")
        for entry in cleaners:
            samples = " · ".join(
                f"`{s['from']}` → `{s['to']}`" for s in entry["samples"][:3]
            ) or "—"
            lines.append(
                f"| `{entry['column']}` | `{entry['cleaner']}` | {entry['changed']:,} | "
                f"{entry['failed']:,} | {samples} |"
            )
        lines.append("")
        failed_entries = [e for e in cleaners if e["failed"]]
        if failed_entries:
            lines.append("### 못 읽은 값")
            lines.append("")
            lines.append("값이 **있었는데** 규격이 정한 형으로 읽지 못한 것들입니다. "
                         "빈 칸과는 다르게 다뤄 그 행을 격리했습니다.")
            lines.append("")
            for entry in failed_entries:
                shown = " · ".join(f"`{v}`" for v in entry["failed_samples"][:10])
                lines.append(f"- `{entry['column']}` ({entry['cleaner']}): {shown}")
            lines.append("")

    # ── 우리가 채운 것 ───────────────────────────
    derive = report.get("derive") or []
    filled = [d for d in derive if d["filled"] or d["too_early"] or d["undecidable"]]
    if filled:
        lines.append("## 우리가 채우고 검사한 것 — 시점")
        lines.append("")
        lines.append("*\"이 자료를 언제부터 쓸 수 있었나\"* 는 비어 있으면 채우고, "
                     "채워져 있으면 규칙과 맞는지 검사합니다. "
                     "**규칙보다 이른 값은 미래참조라 격리합니다.**")
        lines.append("")
        lines.append("| 칸 | 채움 | 확인됨 | 너무 이름 | 판정 불가 |")
        lines.append("|---|---:|---:|---:|---:|")
        for item in derive:
            lines.append(
                f"| `{item['column']}` | {item['filled']:,} | {item['verified']:,} | "
                f"{item['too_early']:,} | {item['undecidable']:,} |"
            )
        lines.append("")

    # ── 경고 ─────────────────────────────────────
    warns = [t for t in (report.get("row_rules") or [])
             if t["severity"] != "error" and t["violations"]]
    if warns:
        lines.append("## 들이긴 했지만 알아 두세요")
        lines.append("")
        lines.append("규격이 `warn` 으로 둔 것들입니다. 오류가 아니라 **알아 둘 사실**이라 "
                     "행은 들였습니다.")
        lines.append("")
        for item in warns:
            lines.append(f"- `{item['rule']}` — {item['violations']:,}행")
        lines.append("")

    # ── 없는 칸을 결측으로 보고 잰 규칙 ──────────
    # 예전에는 이 자리가 "재지 못한 검사" 였다. 이제는 건너뛰지 않고 **채워서 재므로**,
    # 남길 것은 "못 쟀다" 가 아니라 "무엇을 채워서 쟀나" 다. 이걸 안 적으면 보고서를
    # 읽는 사람이 이 검사가 실제 값으로 통과했다고 오해한다.
    filled = [t for t in (report.get("row_rules") or []) if t.get("filled_columns")]
    if filled:
        lines.append("## 없는 칸을 결측으로 보고 잰 검사")
        lines.append("")
        lines.append("파일에 그 칸이 없어 **전부 비어 있는 것으로 보고** 검사했습니다. "
                     "`X is null or …` 꼴은 그대로 통과하고, "
                     "`X is not null or Y is not null` 꼴은 남은 칸으로 판정됩니다.")
        lines.append("")
        for item in filled:
            상태 = "통과" if not item["violations"] else f"위반 {item['violations']:,}행"
            lines.append(f"- `{item['rule']}` — 채운 칸 "
                         f"`{'`, `'.join(item['filled_columns'])}` · {상태}")
        lines.append("")

    # ── 사람에게 묻는 것 ─────────────────────────
    if result.questions:
        lines.append("## 물어볼 것")
        lines.append("")
        lines.append("이름만 보고는 어느 칸인지 정할 수 없어 **추측하지 않고 남겨 뒀습니다.** "
                     "값을 봐서 정하면 절반은 틀립니다.")
        lines.append("")
        for question in result.questions:
            candidates = " 또는 ".join(f"`{c}`" for c in question["candidates"]) or "—"
            note = (question.get("note") or "").replace("\n", " ")
            if len(note) > 200:
                note = note[:197] + "..."
            lines.append(f"- **`{question['column']}`** → {candidates}")
            if note:
                lines.append(f"  - {note}")
        lines.append("")

    # ── 규격 밖 칸 ───────────────────────────────
    extras = (report.get("columns") or {}).get("extras") or []
    if extras:
        lines.append("## 규격에 없는 칸")
        lines.append("")
        lines.append("**버리지 않고 `extras` 로 함께 담았습니다.** 지금은 쓰지 않지만 "
                     "나중에 쓸모가 생겼을 때 파일을 다시 받는 것보다 쌉니다.")
        lines.append("")
        lines.append("- " + ", ".join(f"`{c}`" for c in extras))
        lines.append("")

    return "\n".join(lines) + "\n"


def _rule_notes(result) -> dict:
    """규칙 id → 규격이 적어 둔 설명. 사람이 읽을 표에 붙인다."""
    from ingest.inbox.engine import load_spec

    try:
        spec = load_spec(result.kind)
    except Exception:                                   # noqa: BLE001 — 설명은 없어도 된다
        return {}
    notes = {rule.get("id"): rule.get("note", "")
             for rule in (spec.get("x-alphastack") or {}).get("rowRules") or []}
    # 칸 제약과 정제 실패는 규격의 rowRules 가 아니라 우리가 붙인 이름이다.
    notes.setdefault("cleaner.failed", "값이 있었는데 규격이 정한 형으로 읽지 못했다")
    notes.setdefault("lookahead", "언제부터 쓸 수 있는 자료인지가 규칙과 어긋난다")
    for spec_field in spec["fields"]:
        name = spec_field["name"]
        title = spec_field.get("title", name)
        topic = _with_particle(title, "은", "는")
        subject = _with_particle(title, "이", "가")
        notes.setdefault(f"{name}.required", f"{topic} 반드시 있어야 한다")
        notes.setdefault(f"{name}.pattern", f"{title} 의 모양이 규격과 다르다")
        notes.setdefault(f"{name}.enum", f"{subject} 규격이 정한 값이 아니다")
        notes.setdefault(f"{name}.minimum", f"{subject} 규격의 최솟값보다 작다")
        notes.setdefault(f"{name}.maximum", f"{subject} 규격의 최댓값보다 크다")
    return notes


# ==================================================
# 쓰기
# ==================================================
def write_report(result, *, batch_id: Optional[str] = None,
                 contributor: Optional[str] = None,
                 when: Optional[datetime] = None,
                 root: Optional[Path] = None) -> dict:
    """판정 하나를 JSON + 마크다운으로 남기고 두 경로를 돌려준다.

    🔴 **파일명에 `batch_id` 를 넣는다.** 폴더는 날짜뿐이라 이름이 종류+파일명이면
    - 팀원 둘이 같은 날 `시세.csv` 를 올릴 때 뒤엣것이 앞엣것 보고서를 덮고,
    - `--force` 재검사가 DB 에는 새 `batch_id` 로 남는데 파일에서는 덮인다.

    격리된 행을 다시 보려고 보고서를 찾았을 때 그 자리에 다른 파일의 판정이 있으면
    조사가 거기서 끊긴다. `batch_id` 가 없을 때(예비 검사)는 붙이지 않는다 —
    그때는 DB 에 남는 것도 없어 덮여도 잃을 것이 없다.
    """
    folder = report_dir(when, root)
    name = f"{result.kind}_{_slug(result.source)}"
    if batch_id:
        name = f"{name}__{_slug(str(batch_id))}"

    payload = dict(result.report)
    payload["batch_id"] = batch_id
    payload["contributor"] = contributor
    payload["rejected"] = result.rejected

    json_path = folder / f"{name}.json"
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    md_path = folder / f"{name}.md"
    md_path.write_text(render_markdown(result, batch_id=batch_id, contributor=contributor),
                       encoding="utf-8")

    return {"json": str(json_path), "markdown": str(md_path)}


def write_summary(entries: List[dict], *, when: Optional[datetime] = None,
                  root: Optional[Path] = None) -> str:
    """그날 들인 것 전부를 한 장으로 묶는다 — 세션마다 이걸 먼저 본다."""
    folder = report_dir(when, root)
    lines: List[str] = []
    stamp = (when or datetime.now(KST)).strftime("%Y-%m-%d")

    lines.append(f"# 반입 요약 — {stamp}")
    lines.append("")

    if not entries:
        lines.append("들어온 파일이 없습니다.")
        (folder / "요약.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
        return str(folder / "요약.md")

    total = sum(e.get("rows_total", 0) for e in entries)
    accepted = sum(e.get("rows_accepted", 0) for e in entries)
    quarantined = sum(e.get("rows_quarantined", 0) for e in entries)
    rejected_files = [e for e in entries if e.get("rejected")]

    lines.append(f"파일 **{len(entries)}개** · 행 **{total:,}** "
                 f"→ 들어옴 **{accepted:,}** · 격리 **{quarantined:,}**")
    if rejected_files:
        lines.append("")
        lines.append(f"⚠️ 그중 **{len(rejected_files)}개 파일은 통째로 되돌렸습니다.**")
    lines.append("")

    lines.append("| 파일 | 종류 | 보낸 사람 | 전체 | 들어옴 | 격리 | 판정 |")
    lines.append("|---|---|---|---:|---:|---:|---|")
    for entry in entries:
        verdict = "❌ 파일째 거부" if entry.get("rejected") else (
            "✅ 전량" if entry.get("rows_quarantined", 0) == 0 else "⚠️ 일부 격리"
        )
        lines.append(
            f"| `{Path(entry['source']).name}` | {entry['kind']} | "
            f"{entry.get('contributor') or '—'} | {entry.get('rows_total', 0):,} | "
            f"{entry.get('rows_accepted', 0):,} | {entry.get('rows_quarantined', 0):,} | "
            f"{verdict} |"
        )
    lines.append("")
    lines.append("자세한 판정은 같은 폴더의 파일별 `.md` 를 보세요.")
    lines.append("")

    path = folder / "요약.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return str(path)


__all__ = ["REPORT_ROOT", "report_dir", "render_markdown", "write_report", "write_summary"]
