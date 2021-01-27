import matplotlib.pyplot as plt
import pymc3 as pm
import arviz as az
from scipy import stats

# We have a biased and we flip it many times. We count the number of heads. What's the posterior distribution
# for the probability of getting heads.

# Sample data
data = [1, 1, 1, 0, 1, 1, 0, 0, 1, 1]


# Part a
def coin_tossing():
    # This line is just to initialize the model
    with pm.Model() as coin_flipping_model:
        # The following line defines the prior, theta distributes as Beta(3,2)
        theta = pm.Beta('theta', alpha=3, beta=2)
        # The following line defines the likelihood Bernoulli with p=theta
        y = pm.Bernoulli('y', p=theta, observed=data)
        # This line generates 1000 samples of the posterior
        trace = pm.sample(1000)

        # Plot the traces
        az.plot_trace(trace)
        plt.show()

        # Plot the sampled posterior
        pp = az.plot_posterior(trace)
        plt.show()

        # Print the summary
        print(az.summary(trace))


# Part b
def coin_tossing_with_given_prior_and_data(a_prior, b_prior, data):
    # This line is just to initialize the model
    with pm.Model() as coin_flipping_model:
        # The following line defines the prior, theta distributes as Beta(a_prior,b_prior)
        theta = pm.Beta('theta', alpha=a_prior, beta=b_prior)
        # The following line defines the likelihood Bernoulli with p=theta
        y = pm.Bernoulli('y', p=theta, observed=data)
        # This line generates 1000 samples of the posterior
        trace = pm.sample(1000)

        # Plot the traces
        az.plot_trace(trace)
        plt.show()

        # Plot the sampled posterior
        pp = az.plot_posterior(trace)
        plt.show()

        # Print the summary
        print(az.summary(trace))


if __name__ == "__main__":
    # Executes part a
    coin_tossing()

    # Executes part b with alpha_prior=3, beta_prior=4 and the data
    coin_tossing_with_given_prior_and_data(3, 4, data)
