import numpy as np
import pymc3 as pm
from scipy import stats

def from_posterior(param, samples):
    smin, smax = np.min(samples), np.max(samples)
    x = np.linspace(smin, smax, 100)
    y = stats.gaussian_kde(samples)(x)

    # what was never sampled should have a small probability but not 0,
    # so we'll extend the domain and use linear approximation of density on it
    # Note that the domain can not be extended beyond the interval [0,1].
    x = np.concatenate([[0], x, [1]])
    y = np.concatenate([[0], y, [0]])
    return pm.Interpolated(param, x, y)