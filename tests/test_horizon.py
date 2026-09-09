"""예측 지평 계산 테스트

**왜 이 테스트가 필요한가.** 여기서 틀리면 **에러가 안 난다** — 기준선이 조금 낮게
나오거나, 라벨이 하루 앞당겨지거나, 손익분기가 낮게 잡혀도 파이프라인은 그냥 돈다.
나중에 결론만 조용히 뒤집힌다.

특히 **기준선을 반올림된 등락률로 세는 실수**는 실제로 이 저장소에서 일어났다.
오차는 0.17%p 로 작았지만 **방향이 전부 한쪽**이었다 — 우리가 이기기 쉬워지는 쪽.
그래서 그 구분을 테스트로 못 박는다.
"""

from __future__ import annotations

import math

from evaluation import horizon


def _행(bas_dd: str, *, open_: float = 100.0, close: float = 100.0,
        change: float = 0.0) -> dict:
    return {"bas_dd": bas_dd, "open": open_, "close": close, "change": change}


# ── 기준선 ──────────────────────────────────────────────────────────────────

def test_기준선은_원값_change_로_센다():
    """`change_rate`(반올림)로 세면 실제 오른 날이 보합으로 빠진다.

    아래 셋째 행이 그 경우다 — 원값은 +0.004 로 **오른 날**인데, KRX 가 소수 2자리로
    반올림해 주는 등락률로는 `0.00` 이라 보합으로 세어진다. 실제로 전구간에서 15일이
    이렇고 **그중 7일이 상승일**이다.
    """
    rows = [_행("20200102", change=1.5),
            _행("20200103", change=-2.0),
            _행("20200106", change=0.004)]

    비율, 상승, 하락, 보합 = horizon.daily_baseline(rows)

    assert (상승, 하락, 보합) == (2, 1, 0)
    assert 비율 == 2 / 3


def test_change_가_없는_행도_보합으로_센다():
    """KRX 가 안 주는 날이 있다. 빼 버리면 분모가 줄어 기준선이 부푼다."""
    rows = [_행("20200102", change=1.0), {"bas_dd": "20200103", "change": None}]

    비율, 상승, 하락, 보합 = horizon.daily_baseline(rows)

    assert (상승, 하락, 보합) == (1, 0, 1)
    assert 비율 == 0.5


def test_유의_임계는_알려진_값과_맞는다():
    """기준선 52.64% · 표본 1,217 → 54.99%. 우리 홀드아웃의 실제 크기다."""
    임계 = horizon.significance_threshold(0.5264, 1217)

    assert abs(임계 - 0.5499) < 0.0001


def test_표본이_커질수록_임계가_내려간다():
    """검정력이 표본에 달렸다는 것 자체를 고정한다."""
    작음 = horizon.significance_threshold(0.5264, 250)
    큼 = horizon.significance_threshold(0.5264, 2500)

    assert 큼 < 작음


# ── 레이블 ──────────────────────────────────────────────────────────────────

def test_5일_레이블은_다음날_시가에서_시작한다():
    """🔴 시가(t)에서 시작하면 **t일 장중 수익률이 라벨에 들어간다.**

    조사에서 그 오염이 상관을 10배 부풀리는 것을 확인했다(+0.1709 대 +0.0171).
    그리고 **에러는 나지 않는다.** 그래서 인덱스를 테스트로 잠근다.
    """
    rows = [_행(f"2020010{i}", open_=100.0 + i) for i in range(7)]

    수익 = horizon.returns_5d_open(rows)

    assert len(수익) == 1
    # rows[6].open / rows[1].open - 1 = 106/101 - 1 (rows[0] 은 쓰이지 않는다)
    assert abs(수익[0] - (106.0 / 101.0 - 1.0)) < 1e-12


def test_1일_시가_레이블도_다음날에서_시작한다():
    rows = [_행("20200101", open_=100.0), _행("20200102", open_=110.0),
            _행("20200103", open_=121.0)]

    수익 = horizon.returns_1d_open(rows)

    assert len(수익) == 1
    assert abs(수익[0] - 0.1) < 1e-12


def test_종가_레이블은_실행_불가라_따로_둔다():
    """종가→종가는 비교용이다. 체결 가정에 쓰면 실행할 수 없는 거래가 된다."""
    rows = [_행("20200101", close=100.0), _행("20200102", close=105.0)]

    수익 = horizon.returns_1d_close(rows)

    assert abs(수익[0] - 0.05) < 1e-12


# ── 3분류 ───────────────────────────────────────────────────────────────────

def test_중립_밴드_경계는_중립이다():
    """정확히 ±1.0% 인 날이 상승/하락으로 새면 클래스 비율이 흔들린다."""
    분포 = horizon.classify_3([0.01, -0.01, 0.0101, -0.0101, 0.0])

    assert 분포 == {"상승": 1, "중립": 3, "하락": 1}


def test_클래스_균형_판정():
    """세 클래스가 15~45% 를 벗어나면 임계값을 조정해야 하고 그것도 시도 횟수다."""
    assert horizon.class_balance_ok({"상승": 34, "중립": 39, "하락": 27}) is True
    assert horizon.class_balance_ok({"상승": 50, "중립": 30, "하락": 20}) is False
    assert horizon.class_balance_ok({"상승": 0, "중립": 0, "하락": 0}) is False


# ── 비용 ────────────────────────────────────────────────────────────────────

def test_손익분기_공식():
    """0.5 + 왕복비용 / (2 × E|수익|). 개발구간 실측으로 대조한다."""
    # 1일 · 왕복 0.23% · E|일수익| 0.765% → 65.03%
    assert abs(horizon.breakeven_accuracy(0.0023, 0.00765) - 0.6503) < 0.0001
    # 5일 · 왕복 0.05% · E|5일수익| 1.753% → 51.43%
    assert abs(horizon.breakeven_accuracy(0.0005, 0.01753) - 0.5143) < 0.0001


def test_기대수익이_0이면_손익분기를_숫자로_주지_않는다():
    """0 으로 나눠 inf 를 돌려주면 '아주 어렵다' 로 읽혀 그대로 표에 실린다."""
    assert math.isnan(horizon.breakeven_accuracy(0.0005, 0.0))


def test_회전비용은_보유기간에_반비례한다():
    """1일마다 갈아타면 5일마다의 5배다 — 이게 5일을 고른 이유의 절반이다."""
    assert abs(horizon.annual_turnover_cost(1) - 0.1225) < 1e-9
    assert abs(horizon.annual_turnover_cost(5) - 0.0245) < 1e-9


# ── 봉인 ────────────────────────────────────────────────────────────────────

def test_기본은_개발구간이고_홀드아웃은_명시해야_열린다():
    """🔴 봉인 구간을 실수로 보는 일이 실제로 있었다.

    부르는 쪽이 아무것도 안 정하면 **안전한 쪽**이 나와야 한다.
    """
    rows = [_행("20240831"), _행("20240901"), _행("20260825")]

    assert [r["bas_dd"] for r in horizon.split_dev(rows)] == ["20240831"]
    assert [r["bas_dd"] for r in horizon.split_holdout(rows)] == ["20240901", "20260825"]


def test_두_구간은_겹치지_않고_합치면_전체다():
    """경계 하루가 양쪽에 들어가거나 빠지면 표본 수가 조용히 어긋난다."""
    rows = [_행(f"2024083{i}") for i in range(1, 2)] + [_행("20240901"), _행("20241001")]

    개발 = horizon.split_dev(rows)
    홀드 = horizon.split_holdout(rows)

    assert len(개발) + len(홀드) == len(rows)
    assert not ({r["bas_dd"] for r in 개발} & {r["bas_dd"] for r in 홀드})


# ── 종목 구멍 처리 ──────────────────────────────────────────────────────────

def test_거래정지_구간을_5일수익률로_세지_않는다():
    """🔴 이 테스트가 이 파일에서 가장 중요하다.

    종목은 **거래정지로 행이 비는데**, 행 번호로 "5거래일 뒤" 를 세면 정지 구간을
    통째로 건너뛴 값이 5일 수익률로 둔갑한다. 아래는 그 상황이다 — 종목이
    `0107` 다음 한 달을 쉬고 `0203` 에 돌아오는데, 그 사이 주가가 2배가 됐다.

    행 번호로 세면 +100% 가 "5일 수익률" 로 들어간다. 달력으로 세면 **버려진다.**
    """
    # 시장은 1월 내내 열려 있었다. 멈춘 것은 이 종목이다.
    달력 = horizon.trading_day_index(
        [f"202001{d:02d}" for d in range(2, 32)]
        + [f"202002{d:02d}" for d in range(3, 8)]
    )
    종목 = [_행("20200102", open_=100), _행("20200103", open_=100),
            _행("20200106", open_=100), _행("20200107", open_=100),
            _행("20200203", open_=200), _행("20200204", open_=200),
            _행("20200205", open_=200), _행("20200206", open_=200),
            _행("20200207", open_=200)]

    구멍무시 = horizon.returns_5d_open(종목)
    구멍인식 = horizon.returns_5d_open_gapless(종목, 달력)

    # 행 번호로 세면 정지 구간을 건너뛴 +100% 가 섞인다.
    assert any(r > 0.9 for r in 구멍무시)
    # 달력으로 세면 진입·청산이 정확히 5거래일 떨어진 쌍만 남는다.
    assert 구멍인식 == []


def test_구멍이_없으면_두_계산이_같다():
    """정지가 없는 종목에서는 기존 계산과 결과가 같아야 한다.

    다르면 새 함수가 멀쩡한 행까지 버리고 있다는 뜻이다.
    """
    날짜 = [f"202001{d:02d}" for d in range(2, 20)]
    달력 = horizon.trading_day_index(날짜)
    종목 = [_행(d, open_=100 + i) for i, d in enumerate(날짜)]

    assert horizon.returns_5d_open_gapless(종목, 달력) == horizon.returns_5d_open(종목)


def test_시가가_없는_날은_진입하지_않는다():
    """시가 0 은 체결 가정이 불가능하다. 나누면 조용히 폭발한다."""
    날짜 = [f"202001{d:02d}" for d in range(2, 12)]
    달력 = horizon.trading_day_index(날짜)
    종목 = [_행(d, open_=100) for d in 날짜]
    종목[1]["open"] = 0          # 첫 진입일의 시가가 없다

    수익 = horizon.returns_5d_open_gapless(종목, 달력)

    # 진입 후보 4개 중 시가가 0 인 첫 날이 빠져 3개가 남는다.
    assert len(수익) == 3
    assert all(math.isfinite(r) for r in 수익)


# ── 중첩 레이블 보정 ────────────────────────────────────────────────────────

def test_중첩_분산팽창계수는_알려진_값과_맞는다():
    """h=5 의 이론값은 3.78 이다.

    rho_k = (2/pi)·arcsin(1 - k/5) = 0.590 / 0.410 / 0.262 / 0.128
    VIF = 1 + 2·(합) = 3.78
    """
    assert abs(horizon.overlap_vif(5) - 3.78) < 0.01
    assert horizon.overlap_vif(1) == 1.0      # 중첩이 없으면 보정도 없다


def test_지평이_길수록_실효표본이_더_준다():
    assert horizon.overlap_vif(2) < horizon.overlap_vif(5) < horizon.overlap_vif(10)


def test_중첩_보정이_유의임계를_실제로_올린다():
    """🔴 이 테스트가 핵심이다.

    우리 문서가 적어 온 54.99% 는 iid 가정에서 나온 값이다. 레이블이 5일 중첩인데
    그 보정을 빠뜨리면 **유의하지 않은 것을 유의하다고 말하게 된다.**
    """
    iid = horizon.significance_threshold(0.5264, 1217)
    보정 = horizon.significance_threshold_overlapping(0.5264, 1217, horizon=5)

    assert abs(iid - 0.5499) < 0.0005        # 문서에 적힌 값이 재현된다
    assert 보정 > iid                          # 보정은 반드시 임계를 올린다
    assert abs(보정 - 0.5722) < 0.0010        # 실제로 2.2%p 위다


def test_실측_분산팽창계수는_자기상관을_잡는다():
    """완전 자기상관 계열은 VIF 가 커야 하고, 번갈아 뒤집히면 1 로 잘려야 한다."""
    뭉친것 = [0.0] * 50 + [1.0] * 50          # 강한 양의 자기상관
    번갈아 = [float(i % 2) for i in range(100)]  # 음의 자기상관

    assert horizon.empirical_vif(뭉친것, 5) > 3.0
    # 음의 자기상관으로 표준오차를 줄이지는 않는다 — 우리에게 유리한 보정은 안 한다
    assert horizon.empirical_vif(번갈아, 5) == 1.0


def test_보정은_주_검정에_쓰는_것이_아니다():
    """vif=1 을 명시하면 iid 판과 같아야 한다 — 주 검정 d_t 쪽 경로다."""
    a = horizon.significance_threshold(0.5264, 1217)
    b = horizon.significance_threshold_overlapping(0.5264, 1217, horizon=5, vif=1.0)
    assert abs(a - b) < 1e-12
