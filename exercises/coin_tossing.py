import matplotlib.pyplot as plt
import pymc3 as pm
import arviz as az

# We have a biased and we flip it many times. We count the number of heads. What's the posterior distribution
# for the probability of getting heads.

# Sample data
data = [1, 1, 1, 0, 1, 1, 0, 0, 1, 1, 1, 1, 1, 0, 1, 1, 0, 0, 1, 1, 1, 1, 1, 0, 1, 1, 0, 0, 1, 1, 1, 1, 1, 0, 1, 1, 0, 0, 1, 1]


def coin_tossing():
    # This line is just to initialize the model
    with pm.Model() as coin_flipping_model:
        # The following line defines the prior, theta distributes as Beta(1,1)
        theta = pm.Beta('theta', alpha=1., beta=1.)
        # The following line defines the likelihood Bernoulli with p=theta
        y = pm.Bernoulli('y', p=theta, observed=data)
        # This line generates 1000 samples of the posterior
        trace = pm.sample(1000,  progressbar=True)

        # Plot the traces
        az.plot_trace(trace)
        plt.show()

        # Plot the sampled posterior
        pp = az.plot_posterior(trace, ref_val=0.75, hdi_prob=0.95, point_estimate='mode')
        plt.show()

        # Print the summary
        print(az.summary(trace))


if __name__ == "__main__":
    coin_tossing()
