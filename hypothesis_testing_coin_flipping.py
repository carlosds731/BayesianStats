import pymc3 as pm
import theano
import numpy as np
import arviz as az
import matplotlib.pyplot as plt

if __name__ == "__main__":
    # These are the initial probabilities
    p_1 = 0.46
    p_2 = 0.52

    # This is the data
    number_tosses = 1
    heads = 1

    with pm.Model() as ht_coin_flipping:
        # Here we indicate that we have 2 coins, and the probability
        # of selecting each of the coins is 0.5
        coin = pm.Categorical('coin', p=[0.5, 0.5])

        # This is just a way to tell that if we select the first coin,
        # the probability of getting heads is p_1 and if we select
        # the second coin, the probability of heads is p_2
        p = theano.shared(np.array([p_1, p_2]))[coin]

        # Here the likelihood is binomial, because we don't have the result
        # of each individual toss
        y = pm.Binomial('y', p=p, n=number_tosses, observed=[heads])

        trace = pm.sample(1000, progressbar=True)

        # The mode is the MAP
        az.plot_posterior(trace, point_estimate='mode')

        plt.show()

        pm.Bound

        print(az.summary(trace))
