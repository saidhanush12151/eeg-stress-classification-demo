"""Gradio interface: generate a synthetic EEG trial, plot it, and predict its class.

Educational demo on synthetic data only, not a medical or diagnostic tool.
Run:  python app.py   then open the printed local URL.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from eeg_demo import CHANNEL_NAMES, CLASSES, DURATION, FS, band_features, run, simulate_trial

MODEL, METRICS = run()  # trains the baseline at start-up (~5 s)


def make_sample(true_class: str, seed: int):
    rng = np.random.default_rng(int(seed))
    label = CLASSES.index(true_class)
    gain = rng.lognormal(0, 0.2, len(CHANNEL_NAMES))  # a new, unseen simulated subject
    return simulate_trial(label, gain, rng.normal(0, 0.7), rng), label


def plot_signal(eeg: np.ndarray):
    t = np.arange(eeg.shape[1]) / FS
    fig, ax = plt.subplots(figsize=(9, 5))
    offset = 6 * eeg.std()
    for i, name in enumerate(CHANNEL_NAMES):
        ax.plot(t, eeg[i] - i * offset, lw=0.6)
    ax.set_yticks([-i * offset for i in range(len(CHANNEL_NAMES))], CHANNEL_NAMES)
    ax.set_xlabel("Time (s)")
    ax.set_title(f"Synthetic {len(CHANNEL_NAMES)}-channel EEG ({DURATION} s @ {FS} Hz)")
    fig.tight_layout()
    return fig


def predict(true_class: str, seed: int):
    eeg, label = make_sample(true_class, seed)
    proba = MODEL.predict_proba(band_features(eeg)[None, :])[0]
    pred = CLASSES[int(np.argmax(proba))]
    verdict = "correct" if pred == CLASSES[label] else "incorrect"
    text = (f"Predicted: **{pred}** ({proba.max():.0%} confidence). "
            f"Simulated true class: {CLASSES[label]} ({verdict}).\n\n"
            f"Baseline test accuracy {METRICS['accuracy']:.1%}, macro F1 {METRICS['macro_f1']:.4f} "
            f"on {METRICS['test_subjects']} held-out simulated subjects.")
    return plot_signal(eeg), {c: float(p) for c, p in zip(CLASSES, proba)}, text


def build_ui():
    import gradio as gr
    with gr.Blocks(title="EEG Stress Classification Demo") as demo:
        gr.Markdown("# EEG Stress Classification Demo\n"
                    "Synthetic data, for educational use only. Not a medical tool.")
        with gr.Row():
            cls = gr.Radio(CLASSES, value="stressed", label="Simulate a trial that is")
            seed = gr.Number(value=7, precision=0, label="Random seed")
            btn = gr.Button("Generate and predict", variant="primary")
        plot = gr.Plot(label="Sample signal")
        with gr.Row():
            probs = gr.Label(label="Predicted class probabilities")
            info = gr.Markdown()
        btn.click(predict, [cls, seed], [plot, probs, info])
        demo.load(predict, [cls, seed], [plot, probs, info])
    return demo


if __name__ == "__main__":
    build_ui().launch()
