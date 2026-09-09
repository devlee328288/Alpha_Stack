from scripts.build_stock_model_notebooks import _prepare_best_result_directory


def test_best_result에는_현재_순위_폴더만_남긴다(tmp_path):
    root = tmp_path / "best"
    for name in ("⭐조합A", "❤️조합A", "♡조합A", "조합A"):
        directory = root / name
        directory.mkdir(parents=True)
        (directory / "old.txt").write_text("old", encoding="utf-8")

    target = _prepare_best_result_directory(root, "A", "❤️")

    assert target == root / "❤️조합A"
    assert [path.name for path in root.iterdir()] == ["❤️조합A"]
