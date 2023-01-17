# This piece of code generates 1000 samples of a Beta(8,4) random variable
# and then plots its distribution, with an HDI of 99%

import arviz as az
import matplotlib.pyplot as plt
from scipy import stats
# Let's generate 1000 samples of a Beta(8,4) distribution
data = stats.beta.rvs(8, 4, size=1000)
# Plot the distribution, with its mean and the HDI, specifying a 99% of
result = az.plot_posterior(data={'theta':data}, var_names='theta', hdi_prob=0.99)
plt.show()