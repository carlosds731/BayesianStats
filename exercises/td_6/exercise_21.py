import matplotlib.pyplot as plt
import pymc3 as pm
import numpy as np
import arviz as az
import theano

num_questions = 10


# Part c

def find_map(num_correct_answers):
    with pm.Model():
        likelihood_params = np.array([1.6/3, 2.4/3, 2.9/3])

        group = pm.Categorical('group', p=np.array([1/3, 1/3, 1/3]))

        p = theano.shared(likelihood_params)[group]

        positive_answers = pm.Binomial('positive_answers', n=num_questions,
                                       p=p, observed=[num_correct_answers])

        trace = pm.sample(1000, progressbar=True, return_inferencedata=False)

        az.plot_posterior(trace, point_estimate='mode')

        plt.show()


if __name__ == "__main__":
    find_map(7)
