"""DART 사업보고서 원문 읽기 — 부문별 매출 · 시장점유율 · 생산능력 (M4)

왜 이 파일이 생겼나
    GIC 15장 양식(§6.1)의 slot 4(사업부·제품·지역) · slot 5(가치사슬·운영 KPI) ·
    slot 7(경쟁구조·시장 위치)은 **정형 API 에 없다.** DART 정기보고서 주요정보 30종을
    다 뒤져도 부문별 매출은 없고, 재무제표 전체계정(213개)에도 세그먼트가 없다.

    그런데 `공시서류원본파일 API`(document.xml)로 **사업보고서 본문 XML** 을 받으면
    그 표들이 그대로 들어 있다. 키는 이미 쓰던 DART 키 하나면 된다.

실측 (2025 사업보고서 · 2026-08-01)

    회사        내려받기   본문    표 수   부문표  점유율표  생산능력표
    삼성전자     0.78MB   9.0MB   2,704     32      5        6
    SK하이닉스   0.69MB   9.3MB   2,479     11      1        2
    카카오       1.13MB  16.2MB   3,373     13      0        0
    클래시스     0.50MB   5.5MB   2,188      3      0        2

    **회사마다 다르다.** 점유율은 제조업 위주로 있고 카카오에는 없다. 없는 것을 만들지 않고
    `found=False` 로 돌려주면 상위 단계가 Gap 을 발행한다 (GIC 불변원칙 §2-1·2-3).

주의
    - 본문이 최대 16MB 다. 서버리스에서 매번 받으면 느리므로 `/tmp` 에 24시간 캐시한다.
    - 표 안의 태그를 걷어내고 셀 단위로 자른다. 서식이 회사마다 달라
      **표 제목·머리행으로 고른다** (위치로 고르면 다른 회사에서 엉뚱한 표가 잡힌다).
"""
from __future__ import annotations

import io
import re
import zipfile
from typing import Dict, List, Optional
from urllib.request import urlopen

from ingest.store import tmp_cache

from . import dart_data

DOCUMENT_URL = "https://opendart.fss.or.kr/api/document.xml"
REQUEST_TIMEOUT = 20                        # 서버리스 상한이 60초라 여기서 오래 붙들면 안 된다
CACHE_TTL = 24 * 3600                       # 사업보고서는 1년에 한 번 나온다. 하루 캐시면 넉넉하다

# 표를 고르는 열쇠말. 회사마다 표 제목이 달라서 **여러 개를 OR 로** 본다.
SEGMENT_KEYS = ("부문", "사업부문", "영업부문")
SEGMENT_VALUE_KEYS = ("매출액", "매출", "영업이익")
SHARE_KEYS = ("점유율",)
CAPACITY_KEYS = ("생산능력", "생산실적", "가동률")     # 생산능력·생산실적은 늘 짝으로 나온다
RND_KEYS = ("연구개발비", "연구개발")


def _strip_tags(fragment: str) -> List[str]:
    """표 조각에서 셀 값만 순서대로 뽑는다."""
    cells = re.findall(r"<T[DH][^>]*>(.*?)</T[DH]>", fragment, re.S | re.I)
    out = []
    for cell in cells:
        text = re.sub(r"<[^>]+>", " ", cell)
        text = text.replace("&nbsp;", " ").replace("&amp;", "&")
        out.append(re.sub(r"\s+", " ", text).strip())
    return out


def _rows_of(fragment: str) -> List[List[str]]:
    """표 조각을 행 단위 2차원 배열로 만든다."""
    rows = []
    for row in re.findall(r"<TR[^>]*>(.*?)</TR>", fragment, re.S | re.I):
        cells = _strip_tags(row)
        if cells:
            rows.append(cells)
    return rows


def _number(text: str) -> Optional[float]:
    """'1,879,673' · '△301,146' · '56.3%' · '97,146,675(100%)' → 숫자. 못 읽으면 None.

    - `△`·`▲`·칸 전체를 감싼 `(…)` 는 회계의 음수 표기다. 부호를 잃으면 부문 합계가 안 맞는다.
    - 칸에 숫자가 둘 이상이면(`97,146,675(100%)` 처럼 금액과 비율을 한 칸에 쓴 경우)
      **앞의 것**만 쓴다. 이어 붙이면 971억이 9,714억이 된다 (SK하이닉스 실측에서 겪었다).
    """
    if not text:
        return None
    raw = text.strip()
    negative = raw.startswith("△") or raw.startswith("▲") or (raw.startswith("(") and raw.endswith(")"))
    match = re.search(r"\d[\d,]*(?:\.\d+)?", raw)
    if not match:
        return None
    try:
        value = float(match.group(0).replace(",", ""))
    except ValueError:
        return None
    return -value if negative and value > 0 else value


def _fetch_document(rcept_no: str) -> str:
    """접수번호로 공시 원문(ZIP 안의 XML 전부)을 이어 붙여 돌려준다."""
    cached = tmp_cache.read("dart_doc", rcept_no, CACHE_TTL)
    if cached and isinstance(cached, dict) and cached.get("body"):
        return cached["body"]

    key = dart_data._require_key()
    url = f"{DOCUMENT_URL}?crtfc_key={key}&rcept_no={rcept_no}"
    with urlopen(url, timeout=REQUEST_TIMEOUT) as response:
        raw = response.read()

    try:
        archive = zipfile.ZipFile(io.BytesIO(raw))
    except zipfile.BadZipFile as error:
        # 키 오류·없는 접수번호면 ZIP 이 아니라 XML 오류 응답이 온다
        message = raw.decode("utf-8", "replace")[:200]
        raise dart_data.DartError(f"공시 원문을 받지 못했습니다 — {message}") from error

    body = "\n".join(archive.read(item).decode("utf-8", "replace") for item in archive.infolist())
    tmp_cache.write("dart_doc", rcept_no, {"body": body})
    return body


def find_business_report(code: str, months: int = 15,
                         rows: Optional[List[Dict]] = None) -> Optional[Dict]:
    """최근 사업보고서 한 건을 찾는다 (없으면 None).

    이미 받아 둔 공시 목록이 있으면 `rows` 로 넘긴다 — **다시 부르지 않기 위해서**다.
    배포본에서 DART 한 번이 9초라(실측) 같은 목록을 두 번 받으면 그만큼 그냥 버린다.

    ⚠️ `"사업보고서" in report_name` 으로 거르면 안 된다. 신한지주 실측에서
    `해외증권거래소등에신고한사업보고서등의국내신고` 가 먼저 잡혀 빈 문서를 받았다.
    **이름이 `사업보고서 (` 로 시작하는 것**만 고른다.
    """
    if rows is None:
        rows = dart_data.fetch_disclosures(code, months=months, limit=100).get("rows", [])
    for row in rows:
        name = (row.get("report_name") or "").strip()
        if name.startswith("사업보고서 (") or name == "사업보고서":
            return row
    return None


CONTEXT_WINDOW = 600                        # 표 앞뒤로 이만큼을 '그 표가 놓인 문맥' 으로 본다


def _tables_with_context(body: str) -> List[Dict]:
    """표를 **놓인 문맥과 함께** 뽑는다.

    열쇠말이 표 안에만 있다고 볼 수 없다. 실측(삼성전자)에서 시장점유율 숫자는
    `제품 / 2025년 / 2024년` 표에 들어 있고, `점유율` 이라는 낱말은 **바로 뒤 각주 표**와
    앞쪽 소제목에만 있었다. 표만 보면 영영 못 찾는다.
    """
    out = []
    for match in re.finditer(r"<TABLE.*?</TABLE>", body, re.S | re.I):
        before = body[max(0, match.start() - CONTEXT_WINDOW): match.start()]
        after = body[match.end(): match.end() + CONTEXT_WINDOW]
        out.append({
            "table": match.group(0),
            "context": re.sub(r"<[^>]+>", " ", before + " " + after),
        })
    return out


def _is_clean_header(header: List[str]) -> bool:
    """진짜 표 머리행인지 본다.

    회계정책 주석은 문단 전체가 한 칸에 들어간 '표' 로 돼 있어서, 낱말만 맞춰 고르면
    그게 잡힌다 (SK하이닉스 실측). 머리행 칸이 길면 표가 아니라 문단이다.
    """
    if not header or len(header) < 2:
        return False
    return all(len(cell) <= 40 for cell in header)


def _numeric_rows(rows: List[List[str]], label_max: int = 30) -> List[Dict]:
    """머리행을 뺀 나머지에서 '이름 + 숫자들' 행만 남긴다."""
    parsed = []
    for row in rows:
        if len(row) < 2:
            continue
        label = row[0].strip()
        if not label or len(label) > label_max:
            continue
        numbers = [n for n in (_number(cell) for cell in row[1:]) if n is not None]
        if numbers:
            parsed.append({"label": label, "values": numbers, "cells": row})
    return parsed


def _parse_segments(tables: List[Dict]) -> Dict:
    """부문별 매출 표를 찾아 {부문, 값들} 목록으로 만든다.

    고르는 규칙 — 머리행이 짧고(문단이 아니고), 머리행에 금액 칸이 있고,
    이름+숫자 행이 둘 이상. 후보가 여럿이면 첫 칸이 '부문' 인 표를 우대한다.
    """
    best: Optional[Dict] = None
    best_score = 0
    for entry in tables:
        table = entry["table"]
        if not any(key in table for key in SEGMENT_KEYS):
            continue
        rows = _rows_of(table)
        if len(rows) < 2 or not _is_clean_header(rows[0]):
            continue
        header = rows[0]
        if not any(any(key in cell for key in SEGMENT_VALUE_KEYS) for cell in header):
            continue

        parsed = _numeric_rows(rows[1:])
        if len(parsed) < 2:
            continue
        score = len(parsed) + (5 if any(key in header[0] for key in SEGMENT_KEYS) else 0)
        if score > best_score:
            best_score = score
            best = {"rows": parsed, "header": header}

    if not best:
        return {"found": False, "rows": [], "reason": "부문별 금액 표를 찾지 못했습니다"}
    return {
        "found": True,
        "rows": [{"segment": r["label"], "values": r["values"]} for r in best["rows"]],
        "header": best["header"],
        "reason": f"부문 {len(best['rows'])}개 · 머리행 {' / '.join(best['header'][:6])}",
    }


SHARE_LABEL_MAX = 20            # 이보다 긴 칸은 머리표가 아니라 문장이다


def _share_labelled(rows: List[List[str]]) -> bool:
    """머리행에 **'점유율' 이라고 이름 붙은 칸**이 있는지 본다 (M9 · N96).

    두 가지를 걸러야 한다 — 둘 다 실측에서 실제로 잡혔다.

    | 거르는 것 | 왜 | 실측 |
    |---|---|---|
    | **긴 칸 안에 섞인 '점유율'** | 문장이지 머리표가 아니다 | HLB펩 — `시장상황` 산문 칸에 "시장점유율 1위" 가 있어서 **시장규모 표**(세계 25,015억)가 점유율로 잡혔다 |
    | **'매출액 점유율'** | 회사가 그렇게 쓰지만 뜻은 **매출 비중**이다 | 시그네틱스(해외 72%/국내 28%) · 상아프론테크(부문별 50.6%) |

    매출 비중을 slot 7(경쟁구조·시장 위치)에 싣는 것은 시장에서의 위치를 잘못 말하는 것이다.
    """
    for row in rows[:2]:
        for cell in row:
            text = cell.strip()
            if "점유율" not in text or len(text) > SHARE_LABEL_MAX:
                continue
            if "매출" in text:                 # 매출액 점유율 = 매출 비중
                continue
            return True
    return False


def _percent_column(rows: List[List[str]]) -> bool:
    """데이터 행 중 **한 칸줄이 통째로 백분율**인 것이 있는지 본다 (M9 · N96).

    2단 머리행 표를 살리려고 만든 검사다. `금액 / 점유율` 을 번갈아 싣는 표는
    칸의 절반이 금액이라 '전체 칸의 60% 가 %' 라는 문턱을 못 넘는다
    (NH투자증권 실측 44%·39% — **진짜 점유율 표인데 떨어졌다**).

    자리(column)로 보면 점유율 칸줄은 100% 가 % 다. 그것만 확인한다.
    """
    if not rows:
        return False
    width = max(len(row) for row in rows)
    for column in range(1, width):
        cells = [row[column] for row in rows if column < len(row) and row[column].strip()]
        if len(cells) >= 2 and sum(1 for c in cells if "%" in c) >= len(cells) * 0.6:
            return True
    return False


def _parse_share(tables: List[Dict]) -> Dict:
    """시장점유율 표.

    문맥에 '점유율' 이 있다는 것만으로는 부족했다 — 옆에 있던 매출 비중표·광고시장 규모표가
    딸려 왔다(SK하이닉스·카카오 실측). 그래서 **값이 실제로 백분율 표기(%)** 인 것까지 본다.
    점유율은 반드시 % 로 적히고, 매출액 표는 그렇지 않다.

    ── M9 에서 넓힌 곳 (N96) ────────────────────────────────────
    전종목 검출률이 **7.3%** 였다. 표본 30곳을 관문별로 세어 어디서 떨어지는지 봤다.

    | 관문 | 표본 30곳 | 고칠 수 있나 |
    |---|---:|---|
    | ① 문맥에 '점유율' 이라는 낱말 자체가 없다 | 16곳 (53%) | **없다** — 회사가 점유율을 안 쓴다 |
    | ② 머리행 검사(칸 40자)에서 전멸 | 3곳 (10%) | 문단형 '표' 라 맞다 |
    | ③ % 비율 60% 문턱에서 전멸 | **8곳 (27%)** | **일부만** |
    | 통과 | 3곳 (10%) | |

    ③ 을 눈으로 봤더니 **두 부류**였다.

    - **진짜 점유율 표인데 떨어진 것** — `금액 / 점유율` 을 번갈아 싣는 **2단 머리행** 표.
      칸의 절반이 금액이라 문턱을 못 넘는다 (NH투자증권 44%·39%).
    - **떨어지는 게 맞는 것** — 매출 '비율' 표(성우테크론) · 판매경로표(HL홀딩스) ·
      연혁표(가비아) · 수주잔고표(케이씨티). 문맥 600자 안에 '점유율' 이 있었을 뿐이다.

    그래서 문턱을 **낮추지 않고**, 아래 둘을 **동시에** 만족할 때만 예외로 받는다.

      ① 머리행(최대 두 줄)에 **'점유율' 이라고 이름 붙은 칸이 있다**
      ② 데이터 행에 **통째로 백분율인 칸줄이 있다** (`_percent_column`)

    ①이 매출 '비율' 표를 막고(머리행이 `비율` 이지 `점유율` 이 아니다),
    ②가 문맥만 스친 잡표를 막는다. 둘 중 하나만으로는 못 막는다.
    """
    picked: List[Dict] = []
    for entry in tables:
        table = entry["table"]
        near = table + " " + entry["context"]
        if "점유율" not in near:
            continue
        rows = _rows_of(table)
        if len(rows) < 2 or not _is_clean_header(rows[0]):
            continue

        # 값 칸의 절반 이상이 % 표기여야 점유율 표로 인정한다
        value_cells = [cell for row in rows[1:] for cell in row[1:] if cell.strip()]
        if not value_cells:
            continue
        percent = sum(1 for cell in value_cells if "%" in cell)
        if percent < len(value_cells) * 0.6:
            # 2단 머리행 예외 — 머리행이 '점유율' 이라 **이름 붙였고**
            # 실제로 백분율만 담은 칸줄이 있을 때만 받는다 (N96)
            # 부머리행(숫자가 하나도 없는 줄)은 빼고 자리별로 본다
            data_rows = [row for row in rows[1:]
                         if any(_number(cell) is not None for cell in row[1:])]
            if not (_share_labelled(rows) and _percent_column(data_rows)):
                continue

        parsed = _numeric_rows(rows[1:])
        if parsed:
            picked.append({"header": rows[0],
                           "rows": [{"label": r["label"], "values": r["values"]} for r in parsed]})

    if not picked:
        return {"found": False, "tables": [],
                "reason": "점유율 표가 없습니다 (제조업이 아니면 흔한 일이다)"}
    return {"found": True, "tables": picked[:6],
            "reason": f"시장점유율 표 {len(picked)}개 · 행 {sum(len(t['rows']) for t in picked)}개"}


def _parse_labelled(tables: List[Dict], keys: tuple, label: str) -> Dict:
    """머리행이나 첫 칸에 **열쇠말이 직접 박힌** 표만 모은다 (생산능력 · 연구개발).

    문맥으로 고르면 엉뚱한 것이 딸려 온다 — 실측에서 '연구개발' 문맥에
    `유동파생상품부채`(카카오) · `시차출퇴근제 사용자 수`(신한지주) · `기대주가변동성`(SK하이닉스)이
    잡혔다. 틀린 사실을 싣느니 없다고 하는 편이 낫다 (불변원칙 §2-1).
    """
    picked: List[Dict] = []
    for entry in tables:
        rows = _rows_of(entry["table"])
        if len(rows) < 2 or not _is_clean_header(rows[0]):
            continue
        header_hit = any(any(key in cell for key in keys) for cell in rows[0])
        # 첫 칸으로 고를 때는 **그 칸이 항목 이름이어야** 한다 — 짧고, 열쇠말로 시작한다.
        #
        # 문장 안에 열쇠말이 섞여 있는 것까지 세면 소송 목록의
        # `…연구개발…` 한 줄에 표 전체가 딸려 온다 (신한지주 실측).
        # 반대로 두 줄 이상을 요구하면 `생산능력` 한 줄짜리 진짜 표를 놓친다 (삼성전자 실측).
        label_hits = sum(1 for row in rows[1:]
                         if row and len(row[0].strip()) <= 20
                         and any(row[0].strip().startswith(key) for key in keys))
        if not (header_hit or label_hits >= 1):
            continue
        parsed = _numeric_rows(rows[1:])
        if parsed:
            picked.append({"header": rows[0],
                           "rows": [{"label": r["label"], "values": r["values"]} for r in parsed]})

    if not picked:
        return {"found": False, "tables": [], "reason": f"{label} 표가 없습니다"}
    return {"found": True, "tables": picked[:6],
            "reason": f"{label} 표 {len(picked)}개 · 행 {sum(len(t['rows']) for t in picked)}개"}


def _parse_business_summary(body: str) -> str:
    """'주요 사업의 내용' 문단을 한 덩어리 뽑는다 (slot 3 회사 개요용)."""
    match = re.search(r"주요\s*사업의\s*내용(.{0,3000})", body, re.S)
    if not match:
        return ""
    text = re.sub(r"<[^>]+>", " ", match.group(1))
    text = re.sub(r"\s+", " ", text).strip()
    return text[:800]


def fetch_report_facts(code: str, rows: Optional[List[Dict]] = None) -> Dict:
    """사업보고서 원문에서 slot 3·4·5·7 재료를 뽑아 한 덩어리로 돌려준다.

    **없는 것은 없다고 말한다.** 회사마다 서식이 달라 다 나오지 않는 것이 정상이고,
    그 사실 자체가 리포트의 Gap 이 된다.

    `rows` 에 이미 받아 둔 공시 목록을 넘기면 DART 를 한 번 덜 부른다.
    """
    report = find_business_report(code, rows=rows)
    if not report:
        return {"available": False, "reason": "최근 15개월 안에 사업보고서가 없습니다",
                "code": code}

    rcept_no = report.get("rcept_no", "")
    try:
        body = _fetch_document(rcept_no)
    except Exception as error:              # 원문을 못 받아도 리서치는 계속돼야 한다
        return {"available": False, "reason": f"원문을 받지 못했습니다 — {error}",
                "code": code, "rcept_no": rcept_no}

    # 표와 문맥은 **한 번만** 뽑아 네 파서가 나눠 쓴다.
    # 파서마다 다시 훑으면 9MB 짜리 본문을 네 번 정규식으로 지나간다.
    tables = _tables_with_context(body)

    return {
        "available": True,
        "code": code,
        "table_count": len(tables),
        "corp_name": report.get("corp_name", ""),
        "report_name": report.get("report_name", ""),
        "rcept_no": rcept_no,
        "rcept_date": report.get("date", ""),
        "url": report.get("url", ""),
        "size_mb": round(len(body) / 1e6, 1),
        "business_summary": _parse_business_summary(body),
        "segments": _parse_segments(tables),
        "market_share": _parse_share(tables),
        "capacity": _parse_labelled(tables, CAPACITY_KEYS, "생산능력·가동률"),
        "rnd": _parse_labelled(tables, RND_KEYS, "연구개발"),
    }
