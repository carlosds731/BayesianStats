import matplotlib.pyplot as plt
import pymc3 as pm
import numpy as np
import arviz as az
import theano
from scipy import stats


# Part a
# theta~Uniform(0,1)
# y~Uniform(0,theta)


# Part b
def get_posterior(data, number_of_samples):
    with pm.Model() as our_model:
        theta = pm.Uniform('theta', lower=max(data), upper=1)
        
        y = pm.Uniform('y', lower=0, upper=theta,
                       observed=data)

        return pm.sample(number_of_samples, progressbar=True)


if __name__ == "__main__":
    # Read the data
    file1 = open('juliet_is_late_by.txt', 'r')
    Lines = file1.readlines()
    data = list()
    for line in Lines:
        data.append(float(line))

    # Part c
    posterior_samples = get_posterior(data, 1500)

    # Part d
    # Obtain the summary
    result = az.summary(posterior_samples)
    # Write the solution.
    # Notice that az.summary returns a data frame object, so we can access it's members.
    # Also notice that the summary gives us the standard deviation, so, to get the variance
    # we need to square.
    # If Romeo gathers more data, the variance should decrease
    print(
        'The expected value is {0} and the variance is {1}'.format(result['mean']['theta'],
                                                                   np.power(result['sd']['theta'], 2)))

    # Part e
    # We plot the posterior using the reference value 0.75 (45 minutes is 0.75 of 1 hour).
    az.plot_posterior(posterior_samples, ref_val=0.75)
    plt.show()
    # According to the graph, the probability that theta is bigger than 0.75 is 0.532.
