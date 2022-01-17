import matplotlib.pyplot as plt
import pymc3 as pm
import numpy as np
import arviz as az
import theano
from scipy import stats


# Part b
def get_posterior_samples(data, number_of_samples):
    with pm.Model() as AniaBougherara:
        # Interpolated prior
        theta = pm.HalfNormal('theta', sigma=10)
        # Exponential likelihood
        times = pm.Exponential('times', lam=theta, observed=data)
        trace = pm.sample(number_of_samples, progressbar=True)

        az.plot_posterior(trace)
        plt.show()

        return trace, AniaBougherara


if __name__ == "__main__":
    # Read the data
    file1 = open('AniaBougherara.csv', 'r')
    Lines = file1.readlines()
    data = list()
    for line in Lines:
        data.append(np.float(line))

    posterior_samples, AniaBougherara = get_posterior_samples(data, 2000)

    ##The 94% HDI is between 1.1 and 1.8 It means that, given the data and the model, we can
    # say with 94% of accuracy that the parameter theta is between 1.1 and 1.8.This means
    # that, with 94% accuracy the mean AniaBougherara is between (1/1.1)
    # and (1/1.8).

    # part d
    result = az.summary(posterior_samples)
    print('The expected value is {0} and the variance is {1}'.format(result['mean']['theta'],
                                                                     np.power(result['sd']['theta'], 2)))

    # part e
    az.plot_posterior(posterior_samples, ref_val=1.5)
    plt.show()

    # part f
    map_estimator = pm.find_MAP(model=AniaBougherara)
    print("The MAP estimator for theta_1 is {0}".format(map_estimator['theta']))

    # Part g
    # The quadratic estimation is just the expected value
    # of the posterior. To calculate this,
    # we need to obtain some samples of the posterior
    posterior_samples = pm.sample(2000, progressbar=True, model=AniaBougherara)

    quadratic_estimation = np.average(posterior_samples['theta'])
    print("The quadratic estimation for theta_1 is {0}".format(quadratic_estimation))

    # part h
    predictions = pm.sample_posterior_predictive(posterior_samples, samples=50, model=AniaBougherara)['y'][0]

    # part i
    predictions_for_ppc = pm.sample_posterior_predictive(posterior_samples,
                                                         model=AniaBougherara)
    data_ppc = az.from_pymc3(trace=posterior_samples, posterior_predictive=predictions_for_ppc)
    ax = az.plot_ppc(data_ppc, figsize=(12, 6), mean=False)
    plt.show()
