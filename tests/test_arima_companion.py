import numpy as np

from evaluation.arima_companion import evaluate_arima_companion


def test_arima는_지도학습과같은학습끝에서_t1_t6을비교한다():
    seen_history = []

    def fake_fit(values):
        seen_history.append(list(values))
        return {"model": {"last": values[-1]}, "order": {"p": 1, "d": 1, "q": 0}}

    def fake_forecast(_model, steps):
        return 100.0 + np.arange(1, steps + 1, dtype=float)

    result = evaluate_arima_companion(
        opens=np.full(20, 100.0),
        signal_positions=np.arange(10),
        labels=np.array([0, 0, 0, 0, 0, 0, 1, 1, 1, 1]),
        splits=[(np.arange(5), np.array([6, 7]))],
        neutral_band=0.01,
        fit_function=fake_fit,
        forecast_function=fake_forecast,
    )

    assert len(seen_history[0]) == 5
    assert result.predictions["row_index"].tolist() == [6, 7]
    assert result.predictions["predicted"].tolist() == [1, 1]
    assert result.fold_results.loc[0, "p"] == 1
    assert result.trial_results["phase"].tolist() == ["outer_evaluation"]
