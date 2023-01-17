from scipy import stats

if __name__ == "__main__":
    binom_distribution = stats.binom(n=10)
    binom_distribution.pmf(7, p=p)
    stats.norm()