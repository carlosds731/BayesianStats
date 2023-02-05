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