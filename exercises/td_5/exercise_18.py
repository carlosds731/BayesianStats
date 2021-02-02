import matplotlib.pyplot as plt
import pymc3 as pm
import numpy as np
import arviz as az
import theano
from scipy import stats

if __name__ == "__main__":
    # Read the data
    file1 = open('juliet_is_late_by.txt', 'r')
    Lines = file1.readlines()
    data = list()
    for line in Lines:
        data.append(np.float(line))

    with pm.Model() as juliette_is_late_model:
        theta = pm.Uniform('theta', lower=max(data), upper=1)

        y = pm.Uniform('y', lower=0, upper=theta,
                       observed=data)

        # Part a
        map_estimator = pm.find_MAP()
        print("The MAP estimator for theta is {0}".format(map_estimator['theta']))

        # Part b
        # The quadratic estimation is just the expected value
        # of the posterior. To calculate this,
        # we need to obtain some samples of the posterior
        posterior_samples = pm.sample(1000, progressbar=True)

        quadratic_estimation = np.average(posterior_samples['theta'])
        print("The quadratic estimation for theta is {0}".format(quadratic_estimation))

        # Part c
        # This will generate 1 array of 10 predictions
        predictions = pm.sample_posterior_predictive(posterior_samples, samples=2, model=juliette_is_late_model)['y'][0]

        # Part d
        # For a posterior predictive check, let's not specify the number of samples
        predictions_for_ppc = pm.sample_posterior_predictive(posterior_samples,
                                                             model=juliette_is_late_model)
        data_ppc = az.from_pymc3(trace=posterior_samples, posterior_predictive=predictions_for_ppc)
        ax = az.plot_ppc(data_ppc, figsize=(12, 6), mean=False)
        ax[0].legend(fontsize=15)
        plt.show()
        # The graph shows that the model is not too far away from the data.

        # Part e
        pm.model_to_graphviz(model=juliette_is_late_model)
        plt.show()
