"""Fit anomaly models on normal windows and pick thresholds."""
import numpy as np
from sklearn.covariance import LedoitWolf
from sklearn.decomposition import PCA

from src.features import N_BANDS, N_PER_AXIS

IDLE_COUNTS = 100  # window RMS below this is treated as machine idle
THRESHOLD_PCT = 99  # percentile of normal validation scores
BANDS = np.arange(3 * N_PER_AXIS) % N_PER_AXIS >= N_PER_AXIS - N_BANDS  # band columns


def window_rms(X):
    """One level per window, over all axes."""
    return np.sqrt((X[:, ::N_PER_AXIS] ** 2).sum(axis=1))


def prepare(F, drop_idle=True, log_bands=True):
    """Apply rule 2 (idle) and rule 3 (log bands) to one recording's features."""
    X = F[window_rms(F) >= IDLE_COUNTS if drop_idle else slice(None)].astype('float64')
    if not log_bands:
        X[:, BANDS] = np.exp(X[:, BANDS])
    return X


def fit_rms(X, shrink):
    return {}


def score_rms(model, X):
    return window_rms(X)


def fit_mahalanobis(X, shrink):
    """Standardizing doesn't change the true distance, but keeps the inverse numerically stable."""
    mean, std = X.mean(axis=0), X.std(axis=0)
    Z = (X - mean) / std
    cov = LedoitWolf(assume_centered=True).fit(Z).covariance_ if shrink else np.cov(Z, rowvar=False)
    return dict(mean=mean, std=std, inv=np.linalg.inv(cov))


def score_mahalanobis(m, X):
    Z = (X - m['mean']) / m['std']
    return np.sqrt(np.maximum(np.einsum('ij,jk,ik->i', Z, m['inv'], Z), 0))


def fit_pca(X, shrink):
    """Keep the components holding 95% of normal variance."""
    mean, std = X.mean(axis=0), X.std(axis=0)
    return dict(mean=mean, std=std, components=PCA(.95).fit((X - mean) / std).components_)


def score_pca(m, X):
    Z = (X - m['mean']) / m['std']
    return np.linalg.norm(Z - Z @ m['components'].T @ m['components'], axis=1)


MODELS = dict(rms=(fit_rms, score_rms), mahalanobis=(fit_mahalanobis, score_mahalanobis), pca=(fit_pca, score_pca))


def fit_with_threshold(name, X_train, X_val, shrink=True):
    """Return (model, threshold). Validation holds normal recordings only."""
    fit, score = MODELS[name]
    model = fit(X_train, shrink)
    return model, np.percentile(score(model, X_val), THRESHOLD_PCT)
