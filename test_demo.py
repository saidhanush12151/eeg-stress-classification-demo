"""Smoke tests: python -m pytest -q  (or python test_demo.py)"""
import numpy as np
from eeg_demo import band_features, generate_dataset, run


def test_shapes():
    d = generate_dataset(n_subjects=2, trials_per_class=2)
    assert d.X.shape == (8, 8, 512)
    assert band_features(d.X[0]).shape == (80,)


def test_subject_split_and_metrics(tmp_path="/tmp/eeg_test_results"):
    _, m = run(out_dir=tmp_path)
    assert m["train_subjects"] + m["test_subjects"] == m["n_subjects"]
    assert 0.5 < m["accuracy"] <= 1.0


if __name__ == "__main__":
    test_shapes(); test_subject_split_and_metrics(); print("all tests passed")
