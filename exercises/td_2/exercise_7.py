from scipy import stats

# Part a
def get_posterior(a_prior, b_prior, data):
    n = len(data)
    successes = sum(data)
    return stats.beta(a=a_prior + successes, b=b_prior + n - successes)


def calculate_mean_and_prod(data):
    posterior = get_posterior(a_prior=1, b_prior=1, data=data)
    # Part c
    print("Information with {0} samples".format(len(data)))
    print("Posterior's mean is {0}".format(posterior.mean()))
    print("Posterior's variance is {0}".format(posterior.var()))
    # Part d
    top = 0.35
    bottom = 0.25
    prob_posterior_between = posterior.cdf(top) - posterior.cdf(bottom)
    print(
        "The probability that the posterior is between {0} and {1} is {2}".format(bottom, top, prob_posterior_between))

if __name__ == "__main__":
    # Part b
    bernoulli = stats.bernoulli(p=0.3)
    samples = bernoulli.rvs(10)
    print(samples)
    posterior = get_posterior(a_prior=1, b_prior=1, data=samples)
    # Parts c and d
    calculate_mean_and_prod(samples)
    # Part e
    samples = bernoulli.rvs(100)
    calculate_mean_and_prod(samples)