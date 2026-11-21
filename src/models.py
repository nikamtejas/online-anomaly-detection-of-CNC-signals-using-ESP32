"""Per-operation Mahalanobis model with a centre per period (from notebooks/03_models.ipynb, Days 5 and 6)."""
import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf

from src.features import is_idle

N_MED = 40  # windows in the 5 s median


def split_by_period(idx, seed=0):
    """'train', 'val' or 'test' per recording; every (op, period) keeps at least one good train recording."""
    rng = np.random.default_rng(seed)
    split = pd.Series('test', idx.index)  # all bad recordings
    for _, g in idx[idx['label'] == 'good'].groupby(['op', 'period']):
        order = rng.permutation(g.index)
        n_train, n_val = max(1, round(.6 * len(order))), round(.2 * len(order))
        split.loc[order[:n_train]] = 'train'
        split.loc[order[n_train:n_train + n_val]] = 'val'
    return split


def running(X):
    """Non-idle windows only."""
    return X[~is_idle(X)]


def fit(F, train):
    """Centre per period, then std and Ledoit-Wolf inverse covariance of all periods together."""
    centre = {p: np.vstack([running(F[i]) for i in t.index]).mean(axis=0) for p, t in train.groupby('period')}
    D = np.vstack([running(F[i]) - centre[p] for i, p in zip(train.index, train['period'])])
    std = D.std(axis=0)
    inv = np.linalg.inv(LedoitWolf(assume_centered=True).fit(D / std).covariance_)
    return dict(centre=centre, std=std, inv=inv)


def score(m, X, period):
    """Mahalanobis distance of each window from its period's centre."""
    Z = (X - m['centre'][period]) / m['std']
    return np.sqrt(np.einsum('ij,jk,ik->i', Z, m['inv'], Z))


def median_score(s):
    """Median of the last N_MED window scores; nan until N_MED windows exist."""
    out = np.full(len(s), np.nan)
    if len(s) >= N_MED:
        out[N_MED - 1:] = np.median(np.lib.stride_tricks.sliding_window_view(s, N_MED), axis=1)
    return out


def threshold(m, F, val):
    """Median over validation recordings of their own 99th percentile of the 5 s median score."""
    return np.median([np.nanpercentile(median_score(score(m, running(F[i]), p)), 99) for i, p in zip(val.index, val['period'])])
