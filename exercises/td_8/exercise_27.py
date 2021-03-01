import pymc3 as pm
import arviz as az
import matplotlib.pyplot as plt
import pandas as pd


def get_filtered_data():
    howell_data = pd.read_csv("howell.csv")
    return howell_data[howell_data['age'] >= 18]


if __name__ == "__main__":
    # Part a
    filtered_data = get_filtered_data()
    weight = filtered_data["weight"]
    height = filtered_data["height"]

    with pm.Model() as over_18_heights:
        alpha = pm.Normal("alpha", sd=10)
        beta = pm.Normal("beta", sd=10)
        epsilon = pm.HalfNormal("epsilon", sd=10)

        mu = pm.Deterministic("mu", alpha + beta * weight)

        height_pred = pm.Normal("height_pred", mu=mu, sd=epsilon, observed=height)
        trace_over_18_heights = pm.sample(4000)

    # Part b
    plt.scatter(weight, height)
    plt.xlabel('weight')
    plt.xlabel('height')
    plt.show()

    # When looking at the plot above this is with consistent our expectations.
    # As weight increases, height increases as well. From visual inspection,
    # it looks like a linear fit with some noise is best.

    # Part c
    az.plot_posterior(az.from_pymc3(trace_over_18_heights, model=over_18_heights), var_names=['alpha', 'beta'])
    plt.show()

    # Part d
    # Obtaining the mean value of alpha and beta
    alpha_m = trace_over_18_heights['alpha'].mean()
    beta_m = trace_over_18_heights['beta'].mean()

    # Plot the 99% HDI
    ax = az.plot_hdi(weight, trace_over_18_heights['mu'], hdi_prob=0.99, color='k')
    ax.set_xlabel('weight')
    ax.set_ylabel('height')
    # Plot the regression line
    ax.plot(weight, alpha_m + beta_m * weight, c='k', label=f'y = {alpha_m:.2f} + {beta_m:.2f} * x')
    ax.legend()
    plt.show()

    # Part e
    # With this line we obtain some predictions
    ppc = pm.sample_posterior_predictive(trace_over_18_heights, samples=2000, model=over_18_heights)
    # We plot the 75% HDI of the predicted data
    ax = az.plot_hdi(weight, ppc['height_pred'], hdi_prob=0.75, color='gray')
    # and the 95% HDI of the predicted data
    az.plot_hdi(weight, ppc['height_pred'], hdi_prob=0.95, color='gray', ax=ax)

    # This is just to plot the original data
    plt.plot(weight, height, 'b.')
    # This is to plot the regression line
    plt.plot(weight, alpha_m + beta_m * weight, c='k',
             label=f'weight = {alpha_m:.2f} + {beta_m:.2f} * x')
    plt.xlabel('weight')
    plt.ylabel('height')
    plt.show()

    # Parts f and g

    # From visual inspection the average parameters of the fit look quite good,
    # and the 99% interval of the posterior predictive checks covers most of
    # the distribution. Overall, it looks like a linear fit is great for height vs weight for people over 18!
