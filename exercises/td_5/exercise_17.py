import matplotlib.pyplot as plt
import pymc3 as pm
import arviz as az

# The code is equivalent because the prior it uses for the second data is
# the posterior obtained for the first data; remember that in a binomial model,
# if your prior is Beta(1,1) and you have k successes and n-k failures,
# then the posterior is Beta(1+k, 1+n-k), in this case in particular, initial_data
# has 7 successes and 3 failures, therefore, the posterior is Beta(8,4).


if __name__ == "__main__":
    second_data = [1, 1, 1, 1, 1, 0, 1, 0, 0, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1]

    with pm.Model() as coin_flipping_model:
        # The following line defines the prior, theta distributes as Beta(8,4)
        theta = pm.Beta('theta', alpha=8, beta=4)
        # The following line defines the likelihood Bernoulli with p=theta
        y = pm.Bernoulli('y', p=theta, observed=second_data)
        # This line generates 1000 samples of the posterior
        trace = pm.sample(1000, progressbar=True)

        # Plot the sampled posterior
        pp = az.plot_posterior(trace)
        plt.show()
