# Part a

# The domain should not be extended beyond the interval [0,1]
# because we know that theta is between 0 and 1 (is the probability
# of success.

# Part b
import numpy as np
import pymc3 as pm
from scipy import stats


def from_posterior(param, samples):
    smin, smax = np.min(samples), np.max(samples)
    width = smax - smin
    x = np.linspace(smin, smax, 100)
    y = stats.gaussian_kde(samples)(x)

    # what was never sampled should have a small probability but not 0,
    # so we'll extend the domain and use linear approximation of density on it.
    # The number 3 is arbitrary here, you  can put any positive number you want.
    x = np.concatenate([[x[0] - 3 * width], x, [x[-1] + 3 * width]])
    y = np.concatenate([[0], y, [0]])
    return pm.Interpolated(param, x, y)
