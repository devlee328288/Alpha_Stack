from scripts.build_model_notebooks import MODELS, notebook_targets


def test_모델_실험은_24개_폴더와_96개_모델로_고정된다():
    targets = notebook_targets()

    assert len(targets) == 24
    assert len(targets) * len(MODELS) == 96
    assert len({path for _, _, path in targets}) == 24
