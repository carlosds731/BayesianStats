# Import pymc3
import pymc3 as pm
import matplotlib.pyplot as plt
import arviz as az


# This line is just to avoid this error https://github.com/apache/mxnet/issues/10562#issuecomment-907360893
if __name__ == "__main__":
    # This is just the data
    data = [1, 1, 1, 0, 1, 1, 0, 0, 1, 1]

    # The following code generates 1000 samples of the posterior distribution
    # of the parameter theta, given the data, in a Bayesian model with
    # a Beta(1,1) prior and a binomial likelihood.

    # This line is just to initialize the model
    with pm.Model() as coin_flipping_model:
        # The following line defines the prior, theta distributes as Beta(1,1)
        theta = pm.Beta('theta', alpha=1., beta=1.)
        # The following line defines the likelihood Bernoulli with p=theta
        y = pm.Bernoulli('y', p=theta, observed=data)
        # This line generates 1000 samples of the posterior
        trace = pm.sample(1000, return_inferencedata=True)

    # This code plots the traces
    result = az.plot_trace(trace)
    plt.show()

    # The following line summarizes the posterior distribution
    print(az.summary(trace))

    # This code plots the probability density function of the
    # posterior distribution with a reference value of 0.5
    az.plot_posterior(trace, ref_val=0.5)
    plt.show()