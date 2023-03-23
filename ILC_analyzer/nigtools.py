import numpy as np
from scipy.optimize import minimize
from scipy.special import gammaln
from scipy.stats import invgamma, norm


class NigDist:
    def __init__(self, x, sigma):
        self.xi = x
        self.yi = sigma
        self.mu = None
        self.lambda_ = None
        self.alpha = None
        self.beta = None
        self.unc_aleatoric = None
        self.unc_epistemic = None
        self.k = None

    def estimate_nig_params(self):
        mean_sigma = np.mean(self.yi)
        var_sigma = np.var(self.yi)
        self.alpha = (mean_sigma ** 2) / var_sigma
        self.beta = mean_sigma * (self.alpha - 1)
        initial_params = np.array([0, 1], dtype=np.float64)
        bounds = [(None, None), (1e-10, None)]
        result = minimize(
            self.log_likelihood,
            initial_params,
            args=(self.xi, self.alpha, self.beta, self.yi),
            method="L-BFGS-B",
            bounds=bounds,
        )
        self.mu, self.lambda_ = result.x

        self.unc_aleatoric = self.aleatoric_uncertainty_from_nig()
        self.unc_epistemic = self.epistemic_uncertainty_from_nig()


        return self.mu, self.lambda_, self.alpha, self.beta

    def log_likelihood(self, params, x, alpha, beta, sigma):
        mu, lambda_ = params
        gamma_alpha = gammaln(alpha)
        log_pdf = (
            alpha * np.log(beta)
            + 0.5 * np.log(lambda_)
            - alpha * np.log(sigma)
            - 0.5 * lambda_ * (x - mu) ** 2 / sigma
            - beta / sigma
            - (alpha + 1) * np.log(sigma)
            - 0.5 * np.log(2 * np.pi)
            - gamma_alpha
        )
        return -np.sum(log_pdf)

    @staticmethod
    def normal_inverse_gamma(x, sigma, mu, lambd, alpha, beta):
        return (
            norm.pdf(x, loc=mu, scale=np.sqrt(1 / (lambd * sigma)))
            * invgamma.pdf(sigma, alpha, scale=beta)
        )

    def aleatoric_uncertainty_from_nig(self, k=1):
        self.k = k
        self.unc_aleatoric = self.k * (self.beta / (self.alpha - 1))
        print(self.unc_aleatoric)
        return self.unc_aleatoric

    def epistemic_uncertainty_from_nig(self, k=1):
        print(self.mu, self.lambda_, self.alpha, self.beta)
        self.k = k
        print(f'self.lambda_:{self.lambda_}')
        self.unc_epistemic=self.k * (self.beta / ((self.alpha - 1) * self.lambda_))
        print(f'epistemic:{self.unc_epistemic}')
        return self.unc_epistemic
###########################################################################
def aleatoric_uncertainy_from_nig(alpha, beta, k=1):
    return k * (beta / (alpha - 1))


def epistemic_uncertainy_from_nig(lambda_, alpha, beta, k=1):
    return k * (beta / ((alpha - 1) * lambda_))


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



