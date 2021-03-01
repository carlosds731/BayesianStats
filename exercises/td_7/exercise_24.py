import pymc3 as pm
import theano
import numpy as np
import arviz as az
import matplotlib.pyplot as plt
from scipy import stats

# Part a
# H0: mu=177
# H1: mu is not 177

data = [182, 172, 173, 176, 176, 180, 173, 174, 179, 175]

if __name__ == "__main__":
    # Part b
    # Under H0 (mu=177) the likelihood is just
    # the multiplication of the values of the probability
    # density function
    vals = stats.norm(loc=177, scale=3).pdf(data)
    likelihood_under_h0 = np.prod(vals)

    print("The marginal likelihood under H0 is {0}".format(likelihood_under_h0))

    # Part c
    # To obtain the likelihood under H1, we need
    # to study the model under H1. To obtain the marginal
    # likelihood under H1 we will use the Monte-Carlo sampler
    with pm.Model() as model_under_h1:
        # Normal prior for mu
        mu = pm.Normal('mu', mu=177, sd=4)
        # Likelihood
        y = pm.Normal('y', mu=mu, sigma=3, observed=data)
        traces_under_h1 = pm.sample_smc(2500)

    # Obtaining the marginal likelihood
    likelihoods_under_h1 = np.exp(traces_under_h1.report.log_marginal_likelihood)
    # We need to take the average because the Monte-Carlo sampler
    # returns one marginal likelihood approximation for each chain.
    likelihood_under_h1 = np.average(likelihoods_under_h1)
    print("The marginal likelihood under H1 is {0}".format(likelihood_under_h1))

    # Part d
    bayes_factor = likelihood_under_h0 / likelihood_under_h1
    print("The Bayes factor for the model is {:.7f}".format(bayes_factor))

    # Part e
    # The value of the Bayes factor indicates some evidence in favor of H0. Note that this
    # statement is very vague. See part f for a more precise argument.

    # Part f
    # The probability that Barry's weight didn't change is the probability that H0 is True.
    # Because H0 and H1 have both 50% prior probabilities, the posterior probability
    # of H0 is just
    posterior_probability_h0 = bayes_factor / (bayes_factor + 1)
    print("The probability that Barry's weight didn't change is {:.7f}".format(posterior_probability_h0))
