from scripts.build_model_notebooks import MODELS, notebook_targets


def test_기존_24개와_합집합_G_한개_실험폴더를_만든다():
    targets = notebook_targets()

    assert len(targets) == 25
    assert len(targets) * len(MODELS) == 100
    assert len({path for _, _, path in targets}) == 25
    g_targets = [
        (combination, variant)
        for combination, variant, _ in targets
        if combination == "G"
    ]
    assert g_targets == [("G", "base")]
