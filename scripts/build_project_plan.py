"""프로젝트 계획서 docx 생성 — 강사님 양식(구분·내용·비고 3열)에 맞춘다

**왜 스크립트로 만드나.** 계획서는 한 번 쓰고 끝나는 문서가 아니다. 킥오프에서
미결이 닫히고 실측이 갱신될 때마다 고쳐야 하는데, Word 에서 손으로 고치면
**무엇이 왜 바뀌었는지 git 이 못 본다.** 내용을 이 파일에 두면 diff 로 읽힌다.

## 양식은 강사님 것을 그대로 따른다

강사님 예시(`docs/프로젝트계획서양식_AI퀀트예시.docx`, 저작권 때문에 커밋하지
않는다)를 뜯어보면 구조가 이렇다:

    제목 문단 (가운데, 굵게, 15pt)
    표 1개 · 9행 × 3열 · [구분 | 내용 | 비고]
      팀명 / 팀원(역할) / 기간 / 프로젝트 명 /
      주제 선정 이유 / 프로젝트 목표 / 분석 방법 / 예상 산출물

⚠️ **강사님 예시는 한 장짜리다.** 목표 3개·분석방법 4개로 짧다. 우리는 할 말이
더 많지만 **본표를 부풀리지 않는다.** 본표는 스캔되는 곳이고 세부는 부록으로 뺀다.

## 스타일 기반

`python-docx` 의 기본 템플릿은 한글 폰트(eastAsia)가 잡혀 있지 않아 Word 에서
글꼴이 어긋난다. 그래서 **기존 계획서 docx 를 열어 본문만 비우고 다시 채운다.**

실행:
    python scripts/build_project_plan.py
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Sequence, Tuple

from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml.ns import qn
from docx.shared import Emu, Pt, RGBColor

#: 본표 열 너비(EMU). 강사님 양식은 (985520 / 3789680 / 552450) 인데
#: **비고를 넓혔다** — 예시는 비고가 비어 있어 좁아도 됐지만 우리는 글이 들어가서
#: 552450 으로는 한 줄에 한 낱말씩 끊긴다(실측). 합은 인쇄 가능 너비 5400675.
COL_WIDTHS = (Emu(1_010_000), Emu(3_440_000), Emu(950_675))

#: 부록 표별 열 너비. 합이 5400675 와 같아야 한다.
W_SCOPE = (Emu(560_000), Emu(2_240_000), Emu(940_000), Emu(1_660_675))
W_ROLE = (Emu(1_760_000), Emu(680_000), Emu(680_000), Emu(2_280_675))
W_SCHEDULE = (Emu(660_000), Emu(940_000), Emu(880_000), Emu(940_000),
              Emu(900_000), Emu(1_080_675))
W_ASSET = (Emu(1_700_000), Emu(560_000), Emu(3_140_675))
W_MEASURED = (Emu(1_060_000), Emu(1_440_000), Emu(940_000), Emu(1_960_675))
W_ROADMAP = (Emu(700_000), Emu(2_180_000), Emu(2_520_675))

#: 본문 한글 글꼴. Word 가 eastAsia 를 따로 보기 때문에 둘 다 지정해야 한다.
FONT_KO = "맑은 고딕"

#: 표 머리행 배경.
HEADER_FILL = "D9E2F3"

#: 다이어그램. 없으면 건너뛴다 — mmdc 가 없어도 문서는 만들 수 있어야 한다.
DIAGRAM_DIR = Path("docs/아키텍처/version1.1")

BASE_DOCX = Path("docs/프로젝트계획서_적층_AlphaStack.docx")
OUT_DOCX = Path("docs/프로젝트계획서_Qurious_v1.docx")


# ── docx 원시 조작 ─────────────────────────────────────────────────────────

def _ko(run) -> None:
    """한글 글꼴을 run 에 박는다.

    ⚠️ `run.font.name` 만 넣으면 **라틴 문자에만** 적용되고 한글은 Word 기본값
       으로 떨어진다. `w:eastAsia` 를 따로 넣어야 한다. 빠뜨리면 글꼴이 뒤섞이는데
       **파일 생성은 성공한다.**
    """
    run.font.name = FONT_KO
    rPr = run._element.get_or_add_rPr()
    rFonts = rPr.find(qn("w:rFonts"))
    if rFonts is None:
        rFonts = rPr.makeelement(qn("w:rFonts"), {})
        rPr.append(rFonts)
    rFonts.set(qn("w:eastAsia"), FONT_KO)
    rFonts.set(qn("w:ascii"), FONT_KO)
    rFonts.set(qn("w:hAnsi"), FONT_KO)


def _shade(cell, hex_fill: str) -> None:
    """셀 배경색. python-docx 에 API 가 없어 XML 로 넣는다."""
    tcPr = cell._tc.get_or_add_tcPr()
    shd = tcPr.makeelement(qn("w:shd"), {})
    shd.set(qn("w:val"), "clear")
    shd.set(qn("w:color"), "auto")
    shd.set(qn("w:fill"), hex_fill)
    tcPr.append(shd)


def _borders(table) -> None:
    """표 테두리를 XML 로 직접 넣는다.

    ⚠️ `table.style = "Table Grid"` 를 쓰지 않는다. 한글 Word 로 만든 문서에는
       그 이름의 스타일이 없어서 `KeyError` 로 죽는다(실측). 스타일 이름은 Word
       언어판마다 다르므로 테두리는 이름에 기대지 않고 직접 그린다.
    """
    tblPr = table._tbl.tblPr
    borders = tblPr.makeelement(qn("w:tblBorders"), {})
    for edge in ("top", "left", "bottom", "right", "insideH", "insideV"):
        el = borders.makeelement(qn("w:" + edge), {})
        el.set(qn("w:val"), "single")
        el.set(qn("w:sz"), "4")
        el.set(qn("w:color"), "999999")
        borders.append(el)
    tblPr.append(borders)


def _tight(p) -> None:
    """줄이 어색하게 벌어지는 것을 막는다. 둘을 끈다:

    ① **양쪽 정렬** — 기반 문서의 기본값이라 짧은 줄에서 낱말 사이가 벌어진다.
       실제로 "⑦ 모델 4 종 동일 조건 비교" 가 한 줄에 늘어져 인쇄됐다.
    ② **한글·라틴 자동 간격** — Word 가 한글과 숫자 사이에 공백을 끼워 넣어
       "2010 년부터 16 년" 처럼 보인다. 우리가 쓴 문자열과 달라진다.
    """
    p.alignment = WD_ALIGN_PARAGRAPH.LEFT
    pPr = p._p.get_or_add_pPr()
    for tag in ("w:autoSpaceDE", "w:autoSpaceDN"):
        el = pPr.makeelement(qn(tag), {})
        el.set(qn("w:val"), "0")
        pPr.append(el)


def _fixed_layout(table) -> None:
    """열 너비를 고정한다. 없으면 Word 가 내용에 맞춰 제멋대로 늘린다."""
    tblPr = table._tbl.tblPr
    layout = tblPr.makeelement(qn("w:tblLayout"), {})
    layout.set(qn("w:type"), "fixed")
    tblPr.append(layout)


def _clear_body(doc: Document) -> None:
    """본문을 비운다. `sectPr`(용지·여백)만 남긴다."""
    body = doc.element.body
    for child in list(body):
        if child.tag != qn("w:sectPr"):
            body.remove(child)


def _para(doc: Document, text: str = "", *, size: float = 10.5,
          bold: bool = False, align=None, color: str | None = None):
    p = doc.add_paragraph()
    if align is not None:
        p.alignment = align
    if text:
        r = p.add_run(text)
        r.bold = bold
        r.font.size = Pt(size)
        if color:
            r.font.color.rgb = RGBColor.from_string(color)
        _ko(r)
    return p


_BULLETS = ("·", "①", "②", "③", "④", "⑤", "⑥", "⑦", "⑧", "⑨")


def _fill_cell(cell, lines: Sequence[str], *, size: float = 9.5,
               bold: bool = False) -> None:
    """셀에 여러 줄을 넣는다. 첫 줄은 기존 빈 문단을 재사용한다.

    글머리 기호로 시작하면 내어쓰기를 준다. 표 안에서 Word 목록 스타일을 쓰면
    셀 여백이 들쭉날쭉해져서 기호는 **문자로 직접** 넣는다.
    """
    cell.text = ""
    for i, line in enumerate(lines):
        p = cell.paragraphs[0] if i == 0 else cell.add_paragraph()
        p.paragraph_format.space_after = Pt(1)
        p.paragraph_format.space_before = Pt(0)
        _tight(p)
        if line.startswith(_BULLETS):
            p.paragraph_format.left_indent = Pt(10)
            p.paragraph_format.first_line_indent = Pt(-10)
        r = p.add_run(line)
        r.font.size = Pt(size)
        r.bold = bold
        _ko(r)


def _table(doc: Document, rows: Sequence[Sequence[str]],
           widths: Sequence[Emu] | None = None, *, size: float = 9.0):
    """머리행이 있는 표. `rows[0]` 이 머리행이다."""
    t = doc.add_table(rows=len(rows), cols=len(rows[0]))
    _borders(t)
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    t.autofit = widths is None
    for ri, row in enumerate(rows):
        for ci, text in enumerate(row):
            cell = t.cell(ri, ci)
            _fill_cell(cell, str(text).split("\n"), size=size, bold=(ri == 0))
            if ri == 0:
                _shade(cell, HEADER_FILL)
    if widths:
        _fixed_layout(t)
        for ci, w in enumerate(widths):
            for row in t.rows:
                row.cells[ci].width = w
    return t


def _heading(doc: Document, text: str, *, size: float = 12.0) -> None:
    _para(doc)
    _para(doc, text, size=size, bold=True)


def _picture(doc: Document, name: str, caption: str,
             width: int = 5_300_000) -> None:
    """다이어그램을 넣는다. 없으면 건너뛰되 **무엇을 해야 하는지** 알려준다.

    ⚠️ 예외를 던지지 않는다. mmdc 가 없는 사람도 문서는 만들 수 있어야 한다.
    """
    path = DIAGRAM_DIR / (name + ".png")
    if not path.exists():
        print("   ⚠️ " + str(path) + " 가 없어 건너뜁니다.")
        print("      만들려면: cd " + str(DIAGRAM_DIR) + " && mmdc -i "
              + name + ".mmd -o " + name + ".png -b white -s 2")
        return
    doc.add_picture(str(path), width=Emu(width))
    doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
    _para(doc, caption, size=8.5, align=WD_ALIGN_PARAGRAPH.CENTER,
          color="666666")


# ── 본표 내용 ──────────────────────────────────────────────────────────────

TITLE = "프로젝트 계획서 (주가지수 데이터 활용 머신러닝·딥러닝)"

MEMBERS = [
    "이동원(팀장) – 데이터 수집·정제·전처리·크롤링 / 문서·일정 관리",
    "오준영 – AI 모델 개발 및 학습",
    "강민석 – 백테스팅, 성과지표, 대시보드 제작",
    "신장환 – 피처 엔지니어링·기술적 지표·데이터 품질 (크롤링 분담)",
]

REASON = [
    "① 개별 기술적 지표(RSI·MACD·이동평균)를 따로 쓰기보다 머신러닝으로 "
    "종합했을 때 예측력이 실제로 향상되는지 검증한다.",
    "",
    "② 그런데 주가 예측은 “몇 % 맞혔나”보다 “어떻게 검증했나”가 결론을 "
    "좌우한다. 우리 자료로 직접 잰 숫자로 말하면 — 아무 모델 없이 “항상 상승”만 "
    "해도 52.64%를 맞히고, 검증구간 1,217거래일에서 통계적으로 유의하려면 "
    "54.99%가 필요하며, 국내외 문헌의 현실적 상한은 55%대다. 이길 수 있는 폭이 "
    "2%p 남짓이라는 뜻이다.",
    "",
    "그 좁은 폭은 실수 한 줄로 사라진다. 레이블을 시가(t)→시가(t+5)로 하루 "
    "어긋나게 정렬해 봤더니 상관이 정확히 10.0배로 부풀었다(+0.1709 대 "
    "+0.0171). 에러는 나지 않는다. 그래서 예측 모델과 성과 검증 엔진을 같은 "
    "비중으로 만든다.",
    "",
    "③ 6주 3개 프로젝트가 모두 “개발 및 성과 검증”으로 끝난다. 매 차수 새로 "
    "만드는 것은 앞쪽이고 검증하는 방법은 같다. 1차를 세 프로젝트가 공유할 "
    "바닥층으로 설계한다.",
]

GOALS = [
    "① 룩어헤드 없는 성과 검증 엔진 — 워크포워드 gap 잠금·거래비용·"
    "기준선 3종·사전등록",
    "② 지수 트랙 — KOSPI200 의 5거래일 뒤 등락을 상승/보합/하락 3분류로 예측",
    "③ 종목 트랙 — 예측이 통하는 종목을 먼저 선별하고, 그 안에서 랭킹을 매긴다",
    "④ 요일별 5트랙 중첩 운용 — 같은 모델을 매 거래일 돌려 진입 시점을 분산한다",
    "⑤ 유동 임계값 — 상승/보합/하락 경계를 고정하지 않고 변동성에 맞춰 조정한다",
    "⑥ 신호→포지션 규칙 — 보유/무보유 × 상승/보합/하락 행동과 시드 배분을 정한다",
    "⑦ 모델 4종 동일 조건 비교 — LogisticRegression·RandomForest·"
    "XGBoost·LightGBM",
    "⑧ 단일 명령 재현 + Streamlit 시연 화면",
]

METHODS = [
    "① 수집 — KRX OpenAPI 로 2010년부터 16년 전수. 상장폐지 910종목을 포함해 "
    "생존편향을 막는다",
    "② 시점 정합 — T일 종가로 판단, T+1 시가에 체결, T+6 시가에 평가. "
    "미래 자료 접근을 코드가 차단한다",
    "③ 품질 게이트 — 거래일 캘린더 대조·결측·이상치 리포트. 기준 미달이면 "
    "파이프라인이 멈춘다",
    "④ 피처 — 이동평균·RSI·MACD·볼린저·거래량 파생·변동성을 numpy 로 직접 "
    "구현하고 지표마다 손계산 테스트를 붙인다",
    "⑤ 레이블 — 3분류. 밴드는 실측으로 정했다(지수 ±1.0% / 종목 ±2.0%)",
    "⑥ 종목 선별 — 예측가능성 통계로 1차 선별. 선별은 개발구간에서만 하고 "
    "검증구간에서 고정한다",
    "⑦ 모델 — 4종을 같은 자료·같은 분할·같은 시드로 비교하고, 시도 횟수를 "
    "장부에 기록한다",
    "⑧ 검증 — 워크포워드(gap=5·12폴드)·기준선 3종·ARIMA 동반·거래비용 4수준",
    "⑨ 통계 검정 — 수업에서 배운 기법으로(ADF·Ljung-Box·카이제곱·대응표본 t·"
    "ANOVA+Tukey), 주 검정은 부트스트랩 단측 p<0.05",
]

OUTPUTS = [
    "① Streamlit 시연 화면 — 신호·성과·수집 현황을 한 화면에서 (발표 시연)",
    "② 모델 성능 비교표 — 4종 × 기준선 3종, 거래비용 반영 전후",
    "③ 재현 파이프라인 — 명령 하나로 수집부터 성과까지. 품질 리포트 포함",
    "④ 발표자료·README·주요 결정 기록(ADR)",
    "⑤ 2·3차가 그대로 가져다 쓰는 성과 검증 엔진",
]

PROJECT_NAME = [
    "1안) AlphaStack (알파스택) — 신호·검증·화면을 층으로 쌓는다",
    "2안) Qurious (큐리어스) — 팀명과 통일",
]

PERIOD = ["2026.09.01 ~ 2026.09.15 (발표장소 : koreaIT노원 B강의실)"]

MAIN_ROWS: List[Tuple[str, Sequence[str], str]] = [
    ("팀명", ["Qurious (큐리어스)"], "Quant + Curious"),
    ("팀원(역할)", MEMBERS, "주담당일 뿐\n교차 검수 (부록 B)"),
    ("기간", PERIOD, "6주 3개 중 1차"),
    ("프로젝트 명", PROJECT_NAME, "킥오프에서 확정"),
    ("주제 선정 이유", REASON, "숫자는 모두 실측"),
    ("프로젝트 목표", GOALS, "④⑤⑥ 팀원 제안\n세부는 부록 A"),
    ("분석 방법", METHODS, "세부는 부록 A"),
    ("예상 산출물", OUTPUTS, "①② 가 발표 시연"),
]


# ── 부록 내용 ──────────────────────────────────────────────────────────────

SCOPE_ROWS = [
    ["구분", "범위", "담당", "판단 근거"],
    ["필수", "① 성과 검증 엔진 (워크포워드·거래비용·기준선 3종)",
     "강민석", "이 프로젝트의 논지 자체"],
    ["필수", "② 지수 트랙 — KOSPI200 3분류",
     "오준영", "기준선·유의임계를 이미 실측"],
    ["필수", "③ 종목 트랙 — 선별 후 랭킹",
     "오준영·신장환", "손익분기 52.56% 로 지수보다 유리"],
    ["필수", "④ 요일별 5트랙 중첩 운용",
     "강민석", "팀원 제안 · 진입 시점 분산"],
    ["필수", "⑤ 유동 임계값 (변동성 스케일)",
     "신장환", "팀원 제안 · 고정 밴드와 병행"],
    ["필수", "⑥ 신호→포지션 규칙과 시드 배분",
     "신장환·강민석", "팀원 제안 · 조합을 사전 고정"],
    ["필수", "⑦ 모델 4종 동일 조건 비교",
     "오준영", "수업에서 배운 sklearn 전 영역"],
    ["필수", "⑧ 수집·품질 게이트·공급 계층",
     "이동원", "이미 920만 행 적재 완료"],
    ["필수", "⑨ Streamlit 시연 화면", "강민석", "발표 시연"],
    ["선택", "⑩ 시장 균열 스코어 (변동성 예측용)",
     "신장환", "방향 예측 증분은 실측상 0에 가까웠다"],
    ["선택", "⑪ 뉴스·공시 수집과 텍스트 피처",
     "이동원", "약관 검토가 먼저 (부록 F)"],
    ["선택", "⑫ Keras LSTM/GRU 비교",
     "오준영", "수업 진도가 9월 중 여기까지 나간다"],
    ["선택", "⑬ 표에 없는 기능",
     "제안자", "주제 안에 있고 설명할 수 있으면 넣는다"],
]

ROLE_ROWS = [
    ["영역", "주담당", "교차 검수", "산출물"],
    ["데이터 수집·정제·전처리·크롤링", "이동원", "신장환",
     "적재 파이프라인·품질 리포트·공급 계층"],
    ["피처 엔지니어링·기술적 지표·품질", "신장환", "오준영",
     "지표 모듈·손계산 테스트·선별 통계"],
    ["AI 모델 개발·학습", "오준영", "강민석", "모델 4종·성능 비교표"],
    ["백테스팅·성과지표·대시보드", "강민석", "이동원",
     "검증 엔진·Streamlit 화면"],
]

ROLE_RULES = [
    "① 모든 코드는 Pull Request 로 병합하고 교차 검수자 1인의 승인을 받는다.",
    "② 자신이 만든 기능은 팀 회의에서 5분 안에 설명할 수 있어야 한다. "
    "설명하지 못하는 코드는 병합하지 않는다 — 도구를 쓰는 것은 자유이나 "
    "결과를 아는 것은 책임이다.",
    "③ 매일 진행 상황을 짧게 공유한다. 막히면 하루를 넘기지 않고 말한다.",
    "④ 주요 결정은 한 장짜리 기록으로 남긴다 — 무엇을, 왜, 어떤 대안을 버렸는지.",
    "⑤ 발표자는 신장환 또는 강민석 중에서 킥오프에 정한다.",
]

SCHEDULE_ROWS = [
    ["날짜", "이동원 · 데이터", "오준영 · 모델", "강민석 · 검증·화면",
     "신장환 · 피처·품질", "그날의 완료 기준"],
    ["9/1(화)\n킥오프", "미결 8건 · 담당 경계 · 환경 통일", "〃", "〃", "〃",
     "전원 로컬 테스트 통과\nKRX 키 이용신청"],
    ["9/2(수)", "품질 게이트", "베이스라인 골격", "워크포워드 그룹 인식",
     "지표 6종 골격", "품질 리포트가 나온다"],
    ["9/3(목)", "시세 공급 함수", "학습 루프", "5트랙 중첩 설계",
     "지표 완성·shift 검사", "팀원이 공급 API 로 시세를 꺼낸다"],
    ["9/4(금)", "섹터 매핑", "레이블 생성기", "비용 모델 4수준",
     "유동 밴드 부 실험", "데이터셋 스키마 확정"],
    ["9/5(토)", "유니버스·유동성 필터", "데이터셋 생성", "기준선 3종 + ARIMA",
     "예측가능성 통계", "학습용 자료가 나온다"],
    ["9/6(일)", "예비일 · 문서", "LogReg 기준선", "폴드 커버리지 실측",
     "종목 1차 선별", "베이스라인이 기준선 대비 수치를 낸다"],
    ["9/7(월)\n★인수", "데이터 인수 지점", "모델 2종", "성과지표 완성",
     "선별 결과 고정", "아무도 저장소를 직접 부르지 않는다"],
    ["9/8(화)\n★점검", "중간 점검", "모델 4종", "시도 횟수 계측",
     "포지션 규칙 3안", "선택 범위를 한 번만 결정한다"],
    ["9/9(수)", "뉴스·공시 수집(선택)", "하이퍼파라미터", "5트랙 백테스트",
     "과적합 점검", "모델 비교표 초안"],
    ["9/10(목)", "수집 현황 화면", "모델 확정", "자산곡선·MDD",
     "피처 중요도", "비교표 완성"],
    ["9/11(금)", "문서 정리", "지수·종목 통합", "Streamlit 착수",
     "지표 기여도", "검증 엔진이 닫힌다"],
    ["9/12(토)", "리포트 원고", "예비일", "Streamlit 완성",
     "예비일", "화면에서 신호와 성과를 함께 본다"],
    ["9/13(일)\n★개봉", "—", "—", "검증구간 1회 개봉 · 주 검정",
     "—", "개봉 기록이 정확히 1행"],
    ["9/14(월)", "발표자료", "리허설", "리허설", "리허설", "리허설 1회 완주"],
    ["9/15(화)", "발표", "", "", "", "—"],
]

ASSET_ROWS = [
    ["계층", "줄 수", "상태"],
    ["ingest/ — 수집·적재", "7,229",
     "동작 중. KRX 920만 행이 이 경로로 들어왔다"],
    ["evaluation/ — 성과 검증", "732",
     "동작 중. 모델 없이 단독으로 돈다. 테스트 52개"],
    ["supply/ — 시점정합 공급", "228",
     "동작 중. 미래 자료 접근을 테스트가 막는다"],
    ["common/ — 설정·경로·예산·robots", "2,348",
     "절반 동작. 나머지는 정리 대상"],
    ["timeseries/ — ARIMA·ADF·ACF", "2,063",
     "코드는 있으나 아직 호출처가 없다"],
    ["scripts/ — 실행 CLI", "1,831", "동작 중 (수집·품질·실측 9종)"],
    ["tests/", "2,800", "216개 · 8초"],
    ["features/ — 피처", "35", "비어 있다. 2주에 만들 곳"],
    ["models/ — 모델", "34", "비어 있다. 2주에 만들 곳"],
    ["pipelines/ — 오케스트레이션", "0", "비어 있다. 2주에 만들 곳"],
]

MEASURED_ROWS = [
    ["무엇", "값", "구간", "왜 중요한가"],
    ["데이터 규모", "9,209,812행 · 3,677종목\n4,097거래일",
     "2010-01-04 ~\n2026-08-25", "16년 전수. 부분 표본이 아니다"],
    ["중도 소멸 종목", "910종목", "전 구간",
     "상장폐지가 자료에 남아 있다 → 생존편향 방어"],
    ["기준선 (항상 상승)", "52.64%", "개발구간 2,880일",
     "모델이 이겨야 할 하한"],
    ["유의 임계", "54.99%", "검증구간 1,217일",
     "이보다 낮으면 우연과 구분되지 않는다"],
    ["지수 E|5일수익|", "1.753%", "개발구간", "손익분기 계산의 분모"],
    ["지수 손익분기", "51.43% (ETF 0.05%)\n56.56% (개별주 0.23%)", "개발구간",
     "지수는 ETF 로 거래해야 이길 수 있다"],
    ["종목 E|5일수익|", "4.489% (KOSPI)\n5.159% (전 시장)", "개발구간",
     "지수의 2.6~2.9배. 비용을 흡수한다"],
    ["종목 손익분기", "52.56% (KOSPI)\n52.23% (전 시장)", "개발구간",
     "비용이 5배인데 손익분기는 지수보다 낮다"],
    ["지수 3분류 ±1.0%", "34.03 / 38.66 / 27.31", "개발구간",
     "세 클래스가 15~45% 안"],
    ["종목 3분류 ±2.0%", "30.12 / 37.90 / 31.98", "개발구간 KOSPI",
     "±1.0% 면 중립이 20.7% 로 얇아진다"],
    ["레이블 오정렬 시", "상관이 정확히 10.0배\n(+0.1709 대 +0.0171)",
     "개발구간", "하루 어긋나면 이렇게 된다. 에러는 안 난다"],
]

ROADMAP_ROWS = [
    ["차수", "무엇을 새로 만드나", "1차에서 그대로 가져가는 것"],
    ["1차\n9/1~9/15", "지표 기반 등락 예측 + 성과 검증 엔진", "—"],
    ["2차", "한 종목 신호를 여러 종목 배분으로 넓힌다",
     "검증 엔진·수집 계층·시점정합 규칙"],
    ["3차", "기성 지표(RSI·MACD) 자리를 우리가 만든 지표로 바꾼다",
     "검증 엔진·피처 계약·실험 기록 규약"],
]

OPEN_ITEMS = [
    "① 프로젝트명 — 1안 AlphaStack / 2안 Qurious",
    "② 발표자 — 신장환 또는 강민석",
    "③ 산업 분류 기준 — KRX 업종 / GICS / 직접 매핑 "
    "(수업 자료에 KRX 업종분류·WICS 수집 코드가 이미 있다)",
    "④ 뉴스 제목·요약을 로컬 DB 에 저장해도 되는가 — 네이버 약관이 "
    "“저장(캐시 포함)”을 금지한다. 법적 판단이라 팀에서 함께 결정한다",
    "⑤ 동영상 수집 대상 채널 목록",
]


def build_main_table(doc: Document) -> None:
    rows = [["구분", "내용", "비고"]]
    for label, body, note in MAIN_ROWS:
        rows.append([label, "\n".join(body), note])
    _table(doc, rows, COL_WIDTHS, size=9.0)


def build_appendices(doc: Document) -> None:
    _para(doc)
    _para(doc, "부록", size=14, bold=True)

    _heading(doc, "부록 A. 필수 범위와 선택 범위")
    _para(doc, "필수 범위는 2주 안에 반드시 끝낸다. 선택 범위는 9/8 중간 "
               "점검에서 한 번만 결정하고, 그 뒤로는 늘리지 않는다.", size=9.5)
    _table(doc, SCOPE_ROWS, W_SCOPE, size=8.5)

    _heading(doc, "부록 B. 역할과 교차 검수 규약")
    _para(doc, "아래 담당은 “무엇을 했는지 말할 수 있게” 나눈 것이지 벽을 세운 "
               "것이 아니다. 각자 자기 영역을 끌고 가되 옆 담당의 산출물을 "
               "한 번 더 확인한다.", size=9.5)
    _table(doc, ROLE_ROWS, W_ROLE, size=8.5)
    _para(doc)
    for rule in ROLE_RULES:
        _para(doc, rule, size=9.0)

    _heading(doc, "부록 C. 2주 일정")
    _table(doc, SCHEDULE_ROWS, W_SCHEDULE, size=7.5)
    _para(doc)
    _para(doc, "★ 세 지점이 축이다 — 9/7 데이터 인수, 9/8 선택 범위 결정, "
               "9/13 검증구간 개봉.", size=9.0)
    _para(doc, "※ 검증구간은 9/13 에 단 한 번 연다. 그 전에 열면 사전에 정해 "
               "둔 검정 절차가 무의미해지고, 성능을 보고 설계를 고쳤다는 의심을 "
               "벗을 수 없다.", size=9.0)
    _para(doc, "※ 주말(9/5·6, 9/12·13)을 일정에 포함했다. 실제 배분은 "
               "킥오프에서 조정한다.", size=9.0)
    _para(doc)
    _picture(doc, "일정간트", "[그림 1] 2주 일정")

    _heading(doc, "부록 D. 착수 자산")
    _para(doc, "팀장이 사전에 개발해 둔 부분을 팀 저장소로 이관해 두었다. "
               "데이터와 검증 엔진은 서 있고, 둘을 잇는 피처·모델·파이프라인이 "
               "비어 있다. 2주에 만들 곳이 바로 거기다.", size=9.5)
    _table(doc, ASSET_ROWS, W_ASSET, size=8.5)
    _para(doc)
    _para(doc, "설치·확인 완료 — scikit-learn 1.9.0 · LightGBM 4.7.0 · "
               "XGBoost 3.4.1 · scipy 1.18.1 · pandas 3.0.5 · numpy 2.5.1",
          size=9.0)
    _para(doc, "설치 예정 — streamlit · matplotlib "
               "(대시보드 담당과 킥오프에서 확정)", size=9.0)

    _heading(doc, "부록 E. 실측 근거")
    _para(doc, "아래는 모두 우리 자료로 직접 잰 값이다. 인용이 아니다. "
               "재현: python scripts/measure_horizon.py · "
               "measure_stock_horizon.py", size=9.5)
    _table(doc, MEASURED_ROWS, W_MEASURED, size=8.0)
    _para(doc)
    _para(doc, "※ 손익분기는 “방향 적중 여부와 수익 크기가 서로 무관하다”는 "
               "가정 위에 있다. 큰 변동일이 예측하기 더 어렵다면 실제로 필요한 "
               "정확도는 더 높다.", size=9.0)

    _heading(doc, "부록 F. 킥오프에서 정할 것")
    for item in OPEN_ITEMS:
        _para(doc, item, size=9.0)

    _heading(doc, "부록 G. 3개 프로젝트 확장 로드맵")
    _para(doc, "세 차수의 주제가 모두 “개발 및 성과 검증”으로 끝난다. 매번 "
               "새로 만드는 것은 앞쪽이고 검증하는 방법은 같다. 1차에서 검증 "
               "엔진을 재사용 가능하게 분리하는 것이 6주 전체의 효율을 "
               "좌우한다.", size=9.5)
    _table(doc, ROADMAP_ROWS, W_ROADMAP, size=8.5)

    _heading(doc, "부록 H. 아키텍처")
    _para(doc, "자료는 반드시 시점정합 공급 계층을 지나야 한다. 이 문을 지나지 "
               "않는 조회는 테스트가 막는다. “그때 알 수 있었던 것”만 모델에 "
               "들어가게 하는 구조적 장치다.", size=9.5)
    _picture(doc, "계층아키텍처", "[그림 2] 계층 구조 — 시점정합 정문",
             4_600_000)
    _para(doc)
    _para(doc, "역할별로 파이프라인을 나눠 4명이 병렬로 작업한다. 1차에는 LLM "
               "을 쓰지 않으므로 역할별 워커와 오케스트레이션이며, RAG·에이전트"
               "로의 승격은 2·3차에서 다룬다.", size=9.5)
    _picture(doc, "역할파이프라인", "[그림 3] 역할별 파이프라인 분리")


def main() -> int:
    ap = argparse.ArgumentParser(description="프로젝트 계획서 docx 생성")
    ap.add_argument("--base", type=Path, default=BASE_DOCX,
                    help="스타일을 가져올 기존 docx (한글 글꼴·용지 설정)")
    ap.add_argument("--out", type=Path, default=OUT_DOCX)
    args = ap.parse_args()

    if not args.base.exists():
        print("🔴 스타일 기반 파일이 없습니다: " + str(args.base))
        print("   기존 계획서 docx 가 있어야 한글 글꼴을 물려받습니다.")
        return 1

    doc = Document(str(args.base))
    _clear_body(doc)

    _para(doc, TITLE, size=15, bold=True, align=WD_ALIGN_PARAGRAPH.CENTER)
    _para(doc)
    build_main_table(doc)
    build_appendices(doc)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    doc.save(str(args.out))
    print("✅ " + str(args.out))
    print("   표 " + str(len(doc.tables)) + "개 · 문단 "
          + str(len(doc.paragraphs)) + "개")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
