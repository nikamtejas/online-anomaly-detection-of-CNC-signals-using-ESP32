"""Split by year, score test recordings and log metrics. Run from the repo root: python -m src.evaluate"""
import csv
import datetime
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve

from src.features import FS, STEP, features, index_recordings, load_recording
from src.train import MODELS, fit_with_threshold, prepare

METRICS = Path(__file__).parents[1] / 'results' / 'metrics.csv'
WINDOWS_PER_HOUR = 3600 * FS / STEP
# Each step adds one accuracy rule to the one before, so the gain of each rule is visible.
STEPS = [
    ('base', dict(per_op=False, drop_idle=False, log_bands=False, shrink=False)),
    ('+rule1 per operation', dict(per_op=True, drop_idle=False, log_bands=False, shrink=False)),
    ('+rule2 drop idle', dict(per_op=True, drop_idle=True, log_bands=False, shrink=False)),
    ('+rule3 log bands', dict(per_op=True, drop_idle=True, log_bands=True, shrink=False)),
    ('+rule4 Ledoit-Wolf', dict(per_op=True, drop_idle=True, log_bands=True, shrink=True)),
]
ALL_RULES = STEPS[-1][1]


def load_data():
    """Index plus the time split. Good 2019 trains, good 2020 sets thresholds, all of 2021 tests."""
    idx = index_recordings()
    year, good = idx['period'].str[:4], idx['label'] == 'good'
    idx['split'] = None
    idx.loc[good & (year == '2019'), 'split'] = 'train'
    idx.loc[good & (year == '2020'), 'split'] = 'val'
    idx.loc[year == '2021', 'split'] = 'test'
    return idx, {i: features(load_recording(p)) for i, p in idx['path'].items() if idx.loc[i, 'split']}


def stack(F, rows, cfg):
    return np.vstack([prepare(F[i], cfg['drop_idle'], cfg['log_bands']) for i in rows.index])


def score_test(name, idx, F, per_op, drop_idle, log_bands, shrink):
    """Window scores per test recording. Dividing by the threshold makes operations comparable."""
    cfg = dict(drop_idle=drop_idle, log_bands=log_bands)
    scores = {}
    for _, g in idx.groupby('op') if per_op else [('all', idx)]:
        train, val, test = (g[g['split'] == s] for s in ['train', 'val', 'test'])
        model, threshold = fit_with_threshold(name, stack(F, train, cfg), stack(F, val, cfg), shrink)
        for i in test.index:
            scores[i] = MODELS[name][1](model, prepare(F[i], **cfg)) / threshold
    return scores


def metrics(y, s):
    fpr, tpr, _ = roc_curve(y, s)
    return dict(AUROC=roc_auc_score(y, s), PR_AUC=average_precision_score(y, s), recall=tpr[fpr <= .01].max())


def evaluate(idx, scores):
    """Window-level and recording-level results. A recording scores its 95th percentile window."""
    bad = {i: idx.loc[i, 'label'] == 'bad' for i in scores}
    windows = np.concatenate(list(scores.values()))
    y_win = np.concatenate([np.full(len(s), bad[i]) for i, s in scores.items()])
    window = metrics(y_win, windows)
    window['false_alarms_per_hour'] = (windows[~y_win] > 1).mean() * WINDOWS_PER_HOUR  # before alarm smoothing
    recording = metrics(list(bad.values()), [np.percentile(s, 95) for s in scores.values()])
    return dict(window=window, recording=recording)


def log_row(step, name, level, m, source='evaluate.py'):
    row = dict(date=datetime.date.today(), source=source, step=step, split='time', level=level, model=name,
               features=33, quantization='float', SNR='', AUROC=round(m['AUROC'], 4), PR_AUC=round(m['PR_AUC'], 4),
               recall_at_1pct_FPR=round(m['recall'], 4), false_alarms_per_hour=round(m.get('false_alarms_per_hour', 0)) if level == 'window' else '',
               detection_delay_s=round(m['delay'], 2) if 'delay' in m else '', latency_ms='')
    new = not METRICS.exists()
    with METRICS.open('a', newline='') as f:
        out = csv.DictWriter(f, row.keys())
        if new:
            out.writeheader()
        out.writerow(row)


def report(idx, F, step, name, cfg):
    result = evaluate(idx, score_test(name, idx, F, **cfg))
    for level, m in result.items():
        log_row(step, name, level, m)
    w, r = result['window'], result['recording']
    print(f"{step:22} {name:12} rec AUROC {r['AUROC']:.3f} PR {r['PR_AUC']:.3f} recall {r['recall']:.3f} | "
          f"win AUROC {w['AUROC']:.3f} PR {w['PR_AUC']:.3f} recall {w['recall']:.3f} FA/h {w['false_alarms_per_hour']:.0f}")


if __name__ == '__main__':
    idx, F = load_data()
    print(idx.groupby(['split', 'label']).size().to_string())
    print('bad test recordings per operation:', idx[(idx['split'] == 'test') & (idx['label'] == 'bad')].groupby('op').size().to_dict())
    for step, cfg in STEPS:
        report(idx, F, step, 'mahalanobis', cfg)
    for name in ['rms', 'pca']:
        report(idx, F, 'all rules', name, ALL_RULES)
