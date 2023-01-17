import numpy as np
import pymc3 as pm
import matplotlib.pyplot as plt
import arviz as az

if __name__ == "__main__":
    data_nantes = [1, 0, 2, 1, 1, 1, 1, 1, 1, 0, 0, 0, 2, 1, 2, 0, 1, 1, 1, 2, 0, 2, 1, 3, 0, 3, 0, 0]
    with pm.Model() as model_prior_1:
        lmbda = pm.Gamma('lmbda', alpha=4.57, beta=1.43)
        # Likelihood
        y = pm.Poisson('y', mu=lmbda, observed=data_nantes)
        traces_model_prior_1 = pm.sample_smc(2500)
        az.plot_posterior(traces_model_prior_1)
        plt.show()

    with pm.Model() as model_prior_2:
        lmbda = pm.Lognormal('lmbda', mu=1, sigma=0.5)
        # Likelihood
        y = pm.Poisson('y', mu=lmbda, observed=data_nantes)
        traces_model_prior_2 = pm.sample_smc(2500)
        #az.plot_posterior(traces_model_prior_2)
        #plt.show()

    with pm.Model() as model_prior_3:
        lmbda = pm.Lognormal('lmbda', mu=2, sigma=0.5)
        # Likelihood
        y = pm.Poisson('y', mu=lmbda, observed=data_nantes)
        traces_model_prior_3 = pm.sample_smc(2500)
        #az.plot_posterior(traces_model_prior_3)
        #plt.show()

    with pm.Model() as model_prior_4:
        lmbda = pm.Lognormal('lmbda', mu=1, sigma=2)
        # Likelihood
        y = pm.Poisson('y', mu=lmbda, observed=data_nantes)
        traces_model_prior_4 = pm.sample_smc(2500)
        #az.plot_posterior(traces_model_prior_4)
        #plt.show()

    # Obtaining the marginal likelihoods for all the models
    likelihoods_under_model_1 = np.average(np.exp(traces_model_prior_1.report.log_marginal_likelihood))
    likelihoods_under_model_2 = np.average(np.exp(traces_model_prior_2.report.log_marginal_likelihood))
    likelihoods_under_model_3 = np.average(np.exp(traces_model_prior_3.report.log_marginal_likelihood))
    likelihoods_under_model_4 = np.average(np.exp(traces_model_prior_4.report.log_marginal_likelihood))

    # Calculate the Bayes factors of model 4 against the other models
    print(likelihoods_under_model_4 / likelihoods_under_model_1)
    print(likelihoods_under_model_4 / likelihoods_under_model_2)
    print(likelihoods_under_model_4 / likelihoods_under_model_3)

    # The BF indicates that Model 4 works better, specially against model 3.
    # Note that the BF shows that models 1, 2 and 4 are very similar.
