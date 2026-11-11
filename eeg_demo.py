"""EEG Stress Classification Demo (synthetic data, educational use only).

Pipeline: synthetic multi-channel EEG -> frequency-band features -> baseline
classifier -> held-out evaluation with a subject-wise split.

Run:  python eeg_demo.py            # trains, evaluates, writes results/
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass

import numpy as np
from scipy import signal
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

FS = 128            # sampling rate (Hz)
DURATION = 4        # seconds per trial
N_CHANNELS = 8
CHANNEL_NAMES = ["F3", "F4", "C3", "C4", "P3", "P4", "O1", "O2"]
BANDS = {"delta": (1, 4), "theta": (4, 8), "alpha": (8, 13), "beta": (13, 30), "gamma": (30, 45)}
CLASSES = ["relaxed", "stressed"]

# Mean band amplitudes per class. Stress: higher beta/gamma, lower alpha.
# The two profiles overlap on purpose so the task is not trivially separable.
PROFILES = {
    0: {"delta": 10, "theta": 7, "alpha": 11, "beta": 5.5, "gamma": 2.2},   # relaxed
    1: {"delta": 10, "theta": 7, "alpha": 9.8, "beta": 6.2, "gamma": 2.5},  # stressed
}


@dataclass
class Dataset:
    X: np.ndarray        # (n_trials, n_channels, n_samples)
    y: np.ndarray        # (n_trials,)
    subject: np.ndarray  # (n_trials,) simulated subject id


def simulate_trial(label: int, subject_gain: np.ndarray, subject_shift: float,
                   rng: np.random.Generator) -> np.ndarray:
    """One multi-channel trial: sum of band oscillations + pink-ish noise."""
    n = FS * DURATION
    t = np.arange(n) / FS
    eeg = np.zeros((N_CHANNELS, n))
    for ch in range(N_CHANNELS):
        for band, (lo, hi) in BANDS.items():
            amp = PROFILES[label][band] * subject_gain[ch] * rng.lognormal(0, 0.35)
            for _ in range(3):  # a few components per band
                f = np.clip(rng.uniform(lo, hi) + subject_shift, lo, hi)
                eeg[ch] += (amp / 3) * np.sin(2 * np.pi * f * t + rng.uniform(0, 2 * np.pi))
        white = rng.normal(0, 4.0, n)
        eeg[ch] += signal.lfilter([1], [1, -0.9], white) * 0.5  # 1/f-like noise
    return eeg


def generate_dataset(n_subjects: int = 40, trials_per_class: int = 15, seed: int = 42) -> Dataset:
    rng = np.random.default_rng(seed)
    X, y, subj = [], [], []
    for s in range(n_subjects):
        gain = rng.lognormal(0, 0.2, N_CHANNELS)   # subject-specific channel gains
        shift = rng.normal(0, 0.7)                 # subject-specific frequency shift
        for label in (0, 1):
            for _ in range(trials_per_class):
                X.append(simulate_trial(label, gain, shift, rng))
                y.append(label)
                subj.append(s)
    return Dataset(np.asarray(X), np.asarray(y), np.asarray(subj))


def band_features(eeg: np.ndarray) -> np.ndarray:
    """Per channel: log absolute band power (5) + relative band power (5)."""
    freqs, psd = signal.welch(eeg, fs=FS, nperseg=FS * 2, axis=-1)
    feats = []
    for ch in range(eeg.shape[0]):
        abs_p = np.array([np.trapezoid(psd[ch, (freqs >= lo) & (freqs < hi)],
                                       freqs[(freqs >= lo) & (freqs < hi)])
                          for lo, hi in BANDS.values()])
        feats.extend(np.log(abs_p + 1e-12))
        feats.extend(abs_p / abs_p.sum())
    return np.asarray(feats)


def feature_names() -> list[str]:
    names = []
    for ch in CHANNEL_NAMES:
        names += [f"{ch}_log_{b}" for b in BANDS] + [f"{ch}_rel_{b}" for b in BANDS]
    return names


def build_model():
    return make_pipeline(StandardScaler(),
                         RandomForestClassifier(n_estimators=300, random_state=0, n_jobs=-1))


def run(out_dir: str = "results", seed: int = 42):
    data = generate_dataset(seed=seed)
    F = np.stack([band_features(x) for x in data.X])

    # Subject-wise split: every trial of a simulated subject lands in exactly one split.
    gss = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=seed)
    tr, te = next(gss.split(F, data.y, groups=data.subject))
    assert set(data.subject[tr]).isdisjoint(data.subject[te])

    model = build_model().fit(F[tr], data.y[tr])
    pred = model.predict(F[te])
    acc = accuracy_score(data.y[te], pred)
    f1 = f1_score(data.y[te], pred, average="macro")
    cm = confusion_matrix(data.y[te], pred, labels=[0, 1])

    rf = model.named_steps["randomforestclassifier"]
    top = sorted(zip(feature_names(), rf.feature_importances_), key=lambda p: -p[1])[:8]

    metrics = {
        "data": "synthetic EEG (educational demo)",
        "n_trials": int(len(data.y)), "n_subjects": int(len(np.unique(data.subject))),
        "n_channels": N_CHANNELS, "fs_hz": FS, "trial_seconds": DURATION,
        "n_features": int(F.shape[1]),
        "train_trials": int(len(tr)), "test_trials": int(len(te)),
        "train_subjects": int(len(np.unique(data.subject[tr]))),
        "test_subjects": int(len(np.unique(data.subject[te]))),
        "model": "StandardScaler + RandomForestClassifier(n_estimators=300)",
        "accuracy": round(float(acc), 4), "macro_f1": round(float(f1), 4),
        "confusion_matrix": {"labels": CLASSES, "rows_true_cols_pred": cm.tolist()},
        "top_features": [[n, round(float(v), 4)] for n, v in top],
        "seed": seed,
    }
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "metrics.json"), "w") as fh:
        json.dump(metrics, fh, indent=2)
    report = classification_report(data.y[te], pred, target_names=CLASSES, digits=3)
    with open(os.path.join(out_dir, "metrics.md"), "w") as fh:
        fh.write("# Results (synthetic data, held-out subjects)\n\n")
        fh.write(f"Generated by `python eeg_demo.py` (seed={seed}).\n\n")
        fh.write(f"- Trials: {metrics['n_trials']} from {metrics['n_subjects']} simulated subjects\n")
        fh.write(f"- Train/test: {metrics['train_trials']} / {metrics['test_trials']} trials "
                 f"({metrics['train_subjects']} / {metrics['test_subjects']} subjects, no overlap)\n")
        fh.write(f"- **Accuracy: {acc:.4f}**\n- **Macro F1: {f1:.4f}**\n\n")
        fh.write("## Confusion matrix (rows = true, columns = predicted)\n\n")
        fh.write("| | pred relaxed | pred stressed |\n|---|---|---|\n")
        fh.write(f"| **true relaxed** | {cm[0,0]} | {cm[0,1]} |\n| **true stressed** | {cm[1,0]} | {cm[1,1]} |\n\n")
        fh.write("## Classification report\n\n```\n" + report + "```\n\n")
        fh.write("## Top features (random forest importance)\n\n| Feature | Importance |\n|---|---|\n")
        for n, v in top:
            fh.write(f"| {n} | {v:.4f} |\n")
    return model, metrics


if __name__ == "__main__":
    _, m = run()
    print(json.dumps({k: m[k] for k in ("accuracy", "macro_f1", "confusion_matrix",
                                         "train_subjects", "test_subjects")}, indent=2))
