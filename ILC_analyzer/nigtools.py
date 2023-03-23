import numpy as np
from scipy.optimize import minimize
from scipy.special import gammaln
from scipy.stats import invgamma, norm


def normal_inverse_gamma(x, sigma, mu, lambd, alpha, beta):
    return norm.pdf(x, loc=mu, scale=np.sqrt(1 / (lambd * sigma))) * invgamma.pdf(sigma, alpha, scale=beta)


def log_likelihood(params, x, sigma, alpha, beta):
    mu, lambda_ = params
    gamma_alpha = gammaln(alpha)
    log_pdf = (alpha * np.log(beta) + 0.5 * np.log(lambda_) - alpha * np.log(sigma)
               - 0.5 * lambda_ * (x - mu) ** 2 / sigma - beta / sigma
               - (alpha + 1) * np.log(sigma) - 0.5 * np.log(2 * np.pi) - gamma_alpha)
    return -np.sum(log_pdf)


def estimate_nig_params(x, sigma):
    # Estimate alpha and beta using method of moments
    mean_sigma = np.mean(sigma)
    var_sigma = np.var(sigma)
    alpha = (mean_sigma ** 2) / var_sigma
    beta = mean_sigma * (alpha - 1)

    # Initial guesses for mu and lambda
    initial_params = np.array([0, 1], dtype=np.float64)
    bounds = [(None, None), (1e-10, None)]  # Bounds for mu and lambda
    result = minimize(log_likelihood, initial_params, args=(x, alpha, beta, sigma), method='L-BFGS-B', bounds=bounds)
    mu, lambda_ = result.x
    return mu, lambda_, alpha, beta


def aleatoric_uncertainy_from_nig(alpha, beta, k=1):
    return k * (beta / (alpha - 1))


def epistemic_uncertainy_from_nig(lambda_, alpha, beta, k=1):
    return k * (beta / ((alpha - 1) * lambda_))
