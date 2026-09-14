# dashboard/services/experiment_config.py
"""
Backtest / Cost Sens 공용 실험 조건 (SSOT · Single Source of Truth).

- Backtest  : 이 config 로 1회 실행
- Cost Sens : 이 config 에서 trade_cost 만 여러 값으로 변경
- 캐시 키는 config 전체 해시. 하나라도 다르면 다른 결과.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Optional


@dataclass(frozen=True)
class ExperimentConfig:
    # ── 데이터 ──────────────────────────────────
    scope: str  # "MARKET" | "STOCK" | "UNIVERSE"
    ticker: str  # "KOSPI200" | "005930" | ...
    start: str  # "YYYY-MM-DD"
    end: str  # "YYYY-MM-DD"
    source: str = "stocks30"

    # ── 모델 ────────────────────────────────────
    predictor: str = "Random"  # "Random" | "RandomForest" | ...
    model_revision: str = "v1"
    baseline_kind: str = "fwd_return"  # "fwd_return" | "adaptive"

    # ── 회계 ────────────────────────────────────
    initial_cash: float = 100.0
    trade_cost: float = 0.001

    # ── 재현성 ──────────────────────────────────
    seed: Optional[int] = None

    # ── 직렬화 ──────────────────────────────────
    def to_dict(self) -> dict:
        return asdict(self)

    def with_cost(self, trade_cost: float) -> "ExperimentConfig":
        d = self.to_dict()
        d["trade_cost"] = float(trade_cost)
        return ExperimentConfig(**d)

    # ── 캐시 키 ─────────────────────────────────
    def cache_key(self) -> tuple:
        """완전 설정 해시 — 하나라도 다르면 다른 결과."""
        return (
            self.scope,
            self.ticker,
            self.start,
            self.end,
            self.source,
            self.predictor,
            self.model_revision,
            self.baseline_kind,
            float(self.initial_cash),
            float(self.trade_cost),
            self.seed,
        )

    def signal_key(self) -> tuple:
        """예측·신호 재사용 판단용 (비용·현금 제외)."""
        return (
            self.scope,
            self.ticker,
            self.start,
            self.end,
            self.source,
            self.predictor,
            self.model_revision,
            self.baseline_kind,
            self.seed,
        )

    # ── 요약 표시용 ─────────────────────────────
    def summary_rows(self) -> list[dict]:
        return [
            {"KEY": "SCOPE / TICKER", "VALUE": f"{self.scope} · {self.ticker}"},
            {"KEY": "PERIOD", "VALUE": f"{self.start} ~ {self.end}"},
            {"KEY": "SOURCE", "VALUE": self.source},
            {"KEY": "PREDICTOR", "VALUE": f"{self.predictor} ({self.model_revision})"},
            {"KEY": "BASELINE", "VALUE": self.baseline_kind},
            {"KEY": "INITIAL CASH", "VALUE": f"{self.initial_cash:,.2f}"},
            {"KEY": "TRADE COST", "VALUE": f"{self.trade_cost:.4f}"},
        ]
