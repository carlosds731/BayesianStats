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
        data.append(line)

    with pm.Model() as juliette_is_late_model:
        theta = pm.Uniform('theta', lower=max(data), upper=1)

        y = pm.Uniform('y', lower=0, upper=theta,
                       observed=data)