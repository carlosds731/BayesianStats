import warnings

import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pymc3 as pm
import theano.tensor as tt

warnings.simplefilter(action="ignore", category=FutureWarning)

az.style.use("arviz-darkgrid")
print(f"Running on PyMC3 v{pm.__version__}")
print(f"Running on ArviZ v{az.__version__}")


# Sample data
data = [1,0,1,0,1,1,1,1,0,0,0,1,1,0,1,0,0,1,1,0,1]


def basic_model():
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

if __name__ == "__main__":
    basic_model()
