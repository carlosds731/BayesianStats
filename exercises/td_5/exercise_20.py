import matplotlib.pyplot as plt
import pymc3 as pm
import numpy as np
import arviz as az
import theano
from scipy import stats

if __name__ == "__main__":
    # Read the data
    file1 = open('inter_arrival_times.txt', 'r')
    Lines = file1.readlines()
    data = list()
    for line in Lines:
        data.append(line)

    with pm.Model() as inter_arrival_times:
        # Interpolated prior
        theta_1 = pm.Interpolated('theta_1', x_points=np.array([0, np.divide(1, np.sqrt(5))]),
                                  pdf_points=np.array([0, np.divide(10, np.sqrt(5))]))
        # Exponential likelihood
        times = pm.Exponential('times', lam=theta_1, observed=data)

        # Part a
        map_estimator = pm.find_MAP()
        print("The MAP estimator for theta_1 is {0}".format(map_estimator['theta_1']))

        # Part b
        # The quadratic estimation is just the expected value
        # of the posterior. To calculate this,
        # we need to obtain some samples of the posterior
        posterior_samples = pm.sample(1000, progressbar=True)

        quadratic_estimation = np.average(posterior_samples['theta_1'])
        print("The quadratic estimation for theta_1 is {0}".format(quadratic_estimation))

        # Part c
        # This will generate 1 array of 1000 predictions each (this is a 1x1000 matrix),
        # because we want 200 predictions, we can return the first 200 elements.
        predictions = pm.sample_posterior_predictive(posterior_samples, samples=1, model=inter_arrival_times)['times'].flatten()[:200]

        # If for example we would like to obtain 2500 predictions we could do as follows:
        # This will generate 3 arrays of 1000 predictions each (this is a 3x1000 matrix),
        # because we want 2500 predictions, we can put all the rows together (obtaining a 1x3000 matrix of predictions)
        # and we can return the first 2500 elements.
        predictions = pm.sample_posterior_predictive(posterior_samples, samples=3, model=inter_arrival_times)['times'].flatten()[:2500]

        # Part d
        # For a posterior predictive check, let's not specify the number of samples
        predictions_for_ppc = pm.sample_posterior_predictive(posterior_samples,
                                                             model=inter_arrival_times)
        data_ppc = az.from_pymc3(trace=posterior_samples, posterior_predictive=predictions_for_ppc)
        ax = az.plot_ppc(data_ppc, figsize=(12, 6), mean=False)
        ax.legend(fontsize=15)
        plt.show()
        # The graph shows that the model is quite accurate.

        # Part e
        pm.model_to_graphviz(model=inter_arrival_times)
        plt.show()
