import matplotlib.pyplot as plt
import pymc3 as pm
import numpy as np
import arviz as az
import theano

num_questions = 10


def exercise4():
    with pm.Model() as basic_model:
        probabilities = [0.3, 0.7, 0.95]

        likelihood_params = np.array([np.divide(1, 3) * (1 + 2 * prob) for prob in probabilities])

        group = pm.Categorical('group', p=np.array([1, 1, 1]))

        p = pm.Deterministic('p', theano.shared(likelihood_params)[group])

        positive_answers = pm.Binomial('positive_answers', n=num_questions,
                                       p=p, observed=[7])

        trace = pm.sample(4000, progressbar=True)

        az.plot_trace(trace)

        plt.show()

        az.plot_posterior(trace)

        plt.show()

        az.summary(trace)
        return trace


if __name__ == "__main__":
    trace = exercise4()
