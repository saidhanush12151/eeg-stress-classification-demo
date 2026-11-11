# EEG Stress Classification Demo

> **Synthetic data, educational purposes only.** Every EEG signal in this repository is simulated. Nothing here is real brain data, and the model is not a medical or diagnostic tool.

This is a fresh, small portfolio demo by **Murthy Sai Dhanush**. It is based on the ideas from my B.Tech capstone, a team project on EEG-based stress prediction. The capstone itself is a separate, larger codebase that used a hybrid deep learning model followed by XGBoost. This demo is not that code. It is a deliberately simple, fully reproducible baseline that walks through the core steps end to end.

## What it does

1. **Generates synthetic multi-channel EEG.** There are 8 channels (F3, F4, C3, C4, P3, P4, O1, O2) sampled at 128 Hz, with 4-second trials. Each signal is a sum of delta, theta, alpha, beta, and gamma oscillations plus 1/f-like noise. The "stressed" trials get slightly more beta and gamma and less alpha than the "relaxed" trials. The two classes overlap on purpose, and each simulated subject has its own channel gains and frequency shift.
2. **Extracts frequency-band features.** Using Welch's PSD, it computes the log absolute power and the relative power of each band for every channel, giving **80 features per trial**.
3. **Trains a baseline classifier.** The model is `StandardScaler` followed by `RandomForestClassifier(n_estimators=300)`.
4. **Evaluates on held-out subjects.** `GroupShuffleSplit` keeps every trial from a simulated subject in the same split, either train or test. The code asserts that the two sets of subjects do not overlap. This prevents the subject-level leakage that inflates EEG results when trials are split at random.
5. **Reports** accuracy, macro F1, a confusion matrix, a classification report, and the top features, written to `results/`.
6. **Provides a Gradio app** that simulates a trial from a new, unseen subject, plots the signal, and shows the predicted class with its probabilities.

## Results (from an actual run, `seed=42`)

| Item | Value |
|---|---|
| Trials | 1,200 from 40 simulated subjects (balanced) |
| Train / test | 900 / 300 trials (30 / 10 subjects, no overlap) |
| **Accuracy** | **0.7733** |
| **Macro F1** | **0.7725** |

Confusion matrix on the held-out test set (rows = true class, columns = predicted class):

| | pred relaxed | pred stressed |
|---|---|---|
| **true relaxed** | 125 | 25 |
| **true stressed** | 43 | 107 |

**What this shows:** the band-power features carry a real but noisy signal. The classifier is better at recognising relaxed trials (recall 0.83) than stressed ones (recall 0.71). Relative gamma and beta power are the most important features, which matches how the data was simulated. The task was made deliberately hard. The score reflects only how the simulation was designed, and it says nothing about real-world stress detection. Full details are in [`results/metrics.md`](results/metrics.md) and [`results/metrics.json`](results/metrics.json).

## Setup and run

```bash
git clone <this-repo-url>
cd eeg-stress-classification-demo
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt

python eeg_demo.py      # train + evaluate, rewrites results/ (about 5 s)
python test_demo.py     # smoke tests
python app.py           # Gradio UI, then open the printed http://127.0.0.1:7860 URL
```

In the app, choose whether to simulate a "stressed" or "relaxed" trial and pick a random seed. Then click **Generate and predict**. You will see the 8-channel signal, the class probabilities, and whether the prediction matched the simulated class.

## Files

```
eeg_demo.py        data simulation, band features, subject-wise split, training, metrics
app.py             Gradio interface (sample signal + predicted class)
test_demo.py       smoke tests (shapes, split integrity, metric sanity)
results/           metrics.md and metrics.json from the last run
requirements.txt
```

## Limitations

- The data is synthetic. Real EEG has artifacts (eye blinks, muscle, line noise), non-stationarity, and much weaker class differences.
- The model is a single baseline with no hyperparameter tuning, and the evaluation uses a single subject-wise split. Grouped cross-validation would give a more stable estimate.
- The demo is for learning and portfolio purposes only.

## License

MIT
