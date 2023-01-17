import pymc3 as pm
import theano
import numpy as np
import arviz as az
import matplotlib.pyplot as plt

if __name__ == "__main__":
    # These are the values of the messages
    mu_1 = -0.3
    mu_2 = 0.2
    mu_3 = 0.7

    data = [-1.26318642,  0.5758068 , -0.16657075,  0.08515534, -0.89754746,
        0.20262975,  0.19462712,  0.52539873, -0.34519777,  0.34857591]

    with pm.Model() as ht_coin_flipping:
        # Here we indicate that we have 3 different sources, and the prior probabilities
        source = pm.Categorical('source', p=[0.25, 0.25, 0.5])

        # This is just a way to tell that if the data comes from the first source
        # the mean should be mu_1 and the same with the other sources
        mu = theano.shared(np.array([mu_1, mu_2, mu_3]))[source]

        # The likelihood is normal
        y = pm.Normal('y', mu=mu, sigma=1, observed=data)

        trace = pm.sample(1000, progressbar=True)

        # The mode is the MAP
        az.plot_posterior(trace, point_estimate='mode')

        plt.show()

        print(az.summary(trace))
