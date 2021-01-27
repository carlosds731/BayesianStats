import matplotlib.pyplot as plt
import pymc3 as pm
import numpy as np
import arviz as az
import theano
from scipy import stats


# Part a
def get_posterior_part_a(data, a, b, number_of_samples):
    with pm.Model() as our_model:
        theta = pm.Uniform('theta', lower=a, upper=b)
        y = pm.Poisson('y', mu=1 / theta,
                       observed=data)

        return pm.sample(number_of_samples, progressbar=True)


# Part b
def get_posterior_part_b(data, l, h, sigma_0, number_of_samples):
    with pm.Model() as our_model:
        theta = pm.Uniform('theta', lower=l, upper=h)
        sigma = pm.HalfNormal('sigma', sigma=sigma_0)

        y = pm.Normal('y', mu=theta, sigma=sigma,
                      observed=data)

        return pm.sample(number_of_samples, progressbar=True)


# With this code you can see the Part a in action.
if __name__ == "__main__":
    # Generating 20 samples of a Poisson distribution
    # with parameter 10 (this means that our theta will be 0.1)
    data = stats.poisson.rvs(mu=10, size=20)
    trace = get_posterior_part_a(data, 0, 1, 1000)
    az.plot_posterior(trace)
    # When we plot the posterior, we see that
    # is very close to the real value (0.1)
    plt.show()
