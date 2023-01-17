import matplotlib.pyplot as plt
import pymc3 as pm
import numpy as np
import arviz as az
from scipy import stats

def exercise3():
    with pm.Model() as basic_model:
        # Priors
        theta_1 = pm.Interpolated('theta_1', x_points=np.array([0, np.divide(1, np.sqrt(5))]),
                                  pdf_points=np.array([0, 10 * np.sqrt(5)]))

        # define likelihood
        times = pm.Exponential('times', lam=theta_1, observed=[30, 25, 15, 40, 20])

        map_estimator = pm.find_MAP(vars=[theta_1])['theta_1']

        print("The MAP estimator for the model is {0}".format(map_estimator))

        trace = pm.sample(10000, progressbar=True)

        az.plot_posterior(trace)

        plt.show()

        az.summary(trace)


if __name__ == "__main__":
    unf = stats.expon(scale=30)
    data = unf.rvs(1000)
    print(np.mean(data))

    with open('inter_arrival_times.txt', 'w') as f:
        for item in data:
            f.write("%s\n" % format(item, ".2f"))
    # exercise3()
