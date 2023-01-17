def get_posterior_samples(data, a_prior , b_prior ,number_of_samples):
    with pm.Model() as coin_flipping_model:
        theta = pm.Beta('theta', alpha=a_prior , beta=b_prior)
        y = pm.Bernoulli('y', p=theta , observed=data)
        return pm.sample(number_of_samples)


def sums(a,b):
    return a+b