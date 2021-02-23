import pandas as pd
import numpy as np
import pymc3 as pm
import arviz as az
from scipy import stats
import matplotlib.pyplot as plt


def analyze_smoker():
    tips = pd.read_csv('tips.csv')
    tips.tail()
    tip = tips['tip'].values
    idx = pd.Categorical(tips['smoker'], categories=['No', 'Yes']).codes
    groups = len(np.unique(idx))

    with pm.Model():
        # mu and sigma are vectors because we are passing the parameter shape.
        # With this, we are telling PyMC3 that we want to analyze 2 different mu's
        # and 2 different sigma's
        mu = pm.Normal('mu', mu=0, sd=10, shape=groups)
        sigma = pm.HalfNormal('sigma', sd=10, shape=groups)

        # We use the variable idx to index the mean and the standard deviation
        y = pm.Normal('y', mu=mu[idx], sd=sigma[idx], observed=tip)
        trace_cg = pm.sample(5000)
        az.plot_trace(trace_cg)

    dist = stats.norm()

    means_diff = trace_cg['mu'][:, 0] - trace_cg['mu'][:, 1]
    # Obtain Cohen's d
    d_cohen = (means_diff / np.sqrt((trace_cg['sigma'][:, 0] ** 2 + trace_cg['sigma'][:, 1] ** 2) / 2)).mean()
    # Obtain the probability of superiority
    ps = dist.cdf(d_cohen / (2 ** 0.5))
    ax = plt.axes()
    # Plot the posterior distribution of the difference between the means
    az.plot_posterior(means_diff, ref_val=0, ax=ax, hdi_prob=0.97)
    ax.set_title('Difference between Non Smoker and Smoker')
    ax.plot(0, label=f"Cohen's d = {d_cohen:.2f}\nProb sup = {ps:.2f}", alpha=0)
    ax.legend()
    plt.show()


if __name__ == "__main__":
    analyze_smoker()

    # The data suggest that smoking has almost no influence on the tips.
