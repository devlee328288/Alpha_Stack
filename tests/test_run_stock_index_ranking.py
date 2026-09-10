from pathlib import Path

from scripts import run_stock_index_ranking as runner


def test_결합실행은_long_only_최종선정보고서를_읽는다():
    assert runner.INDEX_REPORT_PATH == (
        Path(runner.ROOT) / "reports" / "index_long_only_selection.json"
    )
