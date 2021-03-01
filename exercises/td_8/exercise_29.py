import pymc3 as pm
import theano
import numpy as np
import arviz as az
import matplotlib.pyplot as plt
from scipy import stats
import pandas as pd
import os

if __name__ == "__main__":
    # Reading all the data
    howell_data = pd.read_csv("howell.csv")
    weight = howell_data["weight"]
    height = howell_data["height"]

    # Plotting the weight against the height
    plt.scatter(weight, height)
    plt.xlabel('weight')
    plt.xlabel('height')
    plt.show()

    # In this case, the relationship between the weight
    # and the height does not seem to be linear.
