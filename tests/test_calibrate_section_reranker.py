import numpy as np

from scripts.calibrate_section_reranker import fit_logistic, predict, validation_song


def test_group_split_is_deterministic() -> None:
    assert validation_song("42") == validation_song("42")


def test_logistic_ranks_separable_examples() -> None:
    x = np.asarray([[0.0], [0.2], [0.8], [1.0]])
    y = np.asarray([0.0, 0.0, 1.0, 1.0])
    model = fit_logistic(x, y, steps=500)
    scores = predict(model, x)
    assert scores[-1] > scores[0]
