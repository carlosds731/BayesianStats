import matplotlib.pyplot as plt
import pymc3 as pm
import numpy as np
import arviz as az
import theano
from scipy import stats

# Part a
# theta~f
# y~Exponential(theta)

# Part b
def get_posterior_samples(data, number_of_samples):
    with pm.Model() as our_model:
        # Interpolated prior
        theta_1 = pm.Interpolated('theta_1', x_points=np.array([0, np.divide(1, np.sqrt(5))]),
                                  pdf_points=np.array([0, np.divide(10, np.sqrt(5))]))
        # Exponential likelihood
        times = pm.Exponential('times', lam=theta_1, observed=data)
        trace = pm.sample(number_of_samples, progressbar=True)
        return trace


if __name__ == "__main__":
    # Part c
    # Read the data from the file
    file1 = open('inter_arrival_times.txt', 'r')
    Lines = file1.readlines()
    data = list()
    for line in Lines:
        data.append(np.float(line))
    # Obtain the posterior samples
    posterior_samples = get_posterior_samples(data=data, number_of_samples=1000)

    # Part c and d
    # Plot the posterior with the 97% HDI
    az.plot_posterior(posterior_samples, hdi_prob=0.97)
    plt.show()
    # The 97% HDI is between 0.03 and 0.035. It means that, given the data and the model, we can
    # say with 97% of accuracy that the parameter theta is between 0.03 and 0.035. This means
    # that, with 97% accuracy the mean inter arrival time is between 28.57 (1/0.035)
    # and 33.3 (1/0.03) minutes.

    # Part e
    # Plot the posterior with the 95% HDI and ROPE between 1/20 (20 minutes) and 1/25 (25 minutes)
    az.plot_posterior(posterior_samples, hdi_prob=0.95, rope=[1 / 25, 1 / 20])
    plt.show()
    # As we can see on the plot, there is no intersection between the ROPE and the 95% HDI.
    # This means that, with 95% accuracy, we can say that the mean inter arrival time is not
    # between 20 and 25 minutes.
