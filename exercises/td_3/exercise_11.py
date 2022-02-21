from scipy import stats
import arviz as az
import matplotlib.pyplot as plt

# Part a
# Because the prior for theta is Normal(0,10) and the likelihood of the data
# is Normal(theta,10), then, the model can be written as
# theta~Normal(0,10)
# y~Normal(theta,10)

if __name__ == "__main__":
    # Part b
    # P(X<0)
    prob = stats.norm.cdf(0, loc=0.5, scale=2)
    print("Given the data, the probability that theta is smaller of equal than 0 is {0:.2f}".format(prob))

    # Part c and d
    # Generate 100000 samples of a Normal(0.5, 4)
    data = stats.norm.rvs(loc=0.5, scale=2, size=100000)
    # Plot the posterior
    az.plot_posterior(data={'theta': data}, var_names='theta', hdi_prob=0.8)
    plt.show()

    # Part e and f
    az.plot_posterior(data={'theta': data}, var_names='theta', hdi_prob=0.8, rope=[-3.5, 3.5])
    plt.show()

    # Because 90% of the distribution is inside the ROPE, we can say, with at least 90% confidence that the experiment
    # was successful.
