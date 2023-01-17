import csv
from scipy.stats import expon
import pymc3 as pm
import arviz as az
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def read_csv():
    student_names = list()
    with open('names_2022.csv', newline='') as csvfile:
        spamreader = csv.reader(csvfile, quotechar='|')
        for row in spamreader:
            student_names.append(''.join(row))
    return student_names


def get_posterior_exponential_likelihood(data):
    with pm.Model() as model_g:
        param = pm.HalfNormal(sigma=10)
        y = pm.Exponential(lam=param, observed=data)
        trace_g = pm.sample(2000)
        az.plot_trace(trace_g)


if __name__ == "__main__":
    # student_names = read_csv()
    # params = np.linspace(start=2 / 3 - 0.1, stop=2 / 3 + 0.1, num=len(student_names))
    # selected_params = np.random.choice(params, replace=False, size=len(student_names))
    #
    # for j, name in enumerate(student_names):
    #     data = expon(scale=selected_params[j]).rvs(50)
    #     pd.DataFrame(data).to_csv(os.path.join('students_data', '{0}.csv'.format(name)), header=False, index=False)

    for name in ['CelineSkander']:
        data = pd.read_csv(os.path.join('students_data', "{0}.csv".format(name)))
        with pm.Model() as model_g:
            param = pm.HalfNormal('param', sigma=10)
            y = pm.Exponential('y', lam=param, observed=data)
            trace_g = pm.sample(2000)
            az.plot_posterior(trace_g, ref_val=1.5)
            plt.show()
