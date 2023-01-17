from scipy.stats import binom

sigma_1 = 0.3
sigma_2 = 0.7
sigma_3 = 0.95

number_success = 5
number_trials = 10

prob_data_sigma_1 = binom.pmf(number_success, number_trials, (1 / 3) * (1 + 2 * sigma_1))
prob_data_sigma_2 = binom.pmf(number_success, number_trials, (1 / 3) * (1 + 2 * sigma_2))
prob_data_sigma_3 = binom.pmf(number_success, number_trials, (1 / 3) * (1 + 2 * sigma_3))

total_prob_data = sum([prob_data_sigma_1, prob_data_sigma_2, prob_data_sigma_3])

p_sigma_1 = prob_data_sigma_1 / total_prob_data
p_sigma_2 = prob_data_sigma_2 / total_prob_data
p_sigma_3 = prob_data_sigma_3 / total_prob_data


def prob_M_given_data(m):
    s_1 = p_sigma_1 * binom.pmf(m, number_success, sigma_1 / (sigma_1 + (1 / 3) * (1 - sigma_1)))
    s_2 = p_sigma_2 * binom.pmf(m, number_success, sigma_2 / (sigma_2 + (1 / 3) * (1 - sigma_2)))
    s_3 = p_sigma_3 * binom.pmf(m, number_success, sigma_3 / (sigma_3 + (1 / 3) * (1 - sigma_3)))

    return s_1 + s_2 + s_3


def lms_estimator():
    return sum([m * prob_M_given_data(m) for m in range(0, 6)])


print(lms_estimator())
