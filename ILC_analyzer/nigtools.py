import numpy as np
from scipy.optimize import minimize
from scipy.special import gammaln
from scipy.stats import invgamma, norm
from ILC_analyzer.plotting import plot_x_with_uncertainty
import matplotlib.pyplot as plt

#https://deebuls.github.io/devblog/probability/python/plotting/matplotlib/2020/05/19/probability-normalinversegamma.html
class NigDist:
    def __init__(self, x, sigma, k=1, num_points=100):
        self.xi = x
        self.sigmai = sigma

        self.num_points = num_points
        self.xi_min = np.min(self.xi)
        self.xi_max = np.max(self.xi)
        self.sigmai_min = np.min(self.sigmai)
        self.sigmai_max = np.max(self.sigmai)
        # Define arrays of x and sigma points
        self.multiplier = 0.2
        self.x_range = np.linspace(self.xi_min - self.multiplier * np.abs(self.xi_max - self.xi_min),
                                   self.xi_max + self.multiplier * np.abs(self.xi_max - self.xi_min), self.num_points)
        self.sigma_range = np.linspace(
            np.clip(self.sigmai_min - self.multiplier * np.abs(self.sigmai_max), a_min=1e-10, a_max=np.inf),
            self.sigmai_max + self.multiplier * np.abs(self.sigmai_max), self.num_points)

        self.mu = None
        self.lambda_ = None
        self.alpha = None
        self.beta = None

        self.k = k

        self.mu, self.lambda_, self.alpha, self.beta = self.estimate_nig_params(x=self.xi, sigma=self.sigmai)
        self.unc_aleatoric = self.aleatoric_uncertainy_from_nig(self.alpha, self.beta, k=self.k)
        self.unc_epistemic = self.epistemic_uncertainy_from_nig(self.lambda_, self.alpha, self.beta, self.k)

    def plot_x_with_uncertainty(self):
        plot_x_with_uncertainty(self.xi, self.sigmai, self.mu, self.unc_aleatoric, self.k)

    def plot_3d(self):
        """
        Plots a 3D surface plot of the function defined by Z over the (x, sigma) domain.
        """
        x, sigma = np.meshgrid(self.x_range, self.sigma_range)
        Z = normal_inverse_gamma(x, sigma, self.mu, self.lambda_, self.alpha, self.beta)
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_surface(x, sigma, Z, cmap='viridis')
        ax.set_xlabel('x')
        ax.set_ylabel('sigma')
        ax.set_zlabel('f(x, sigma)')
        ax.set_title('3D Plot')
        plt.show()

    @staticmethod
    def estimate_nig_params(x, sigma):
        # Estimate alpha and beta using method of moments
        mean_sigma = np.mean(sigma)
        var_sigma = np.var(sigma)
        alpha = (mean_sigma ** 2) / var_sigma
        beta = mean_sigma * (alpha - 1)

        # Initial guesses for mu and lambda
        initial_params = np.array([0, 1], dtype=np.float64)
        bounds = [(None, None), (1e-10, None)]  # Bounds for mu and lambda
        result = minimize(log_likelihood, initial_params, args=(x, alpha, beta, sigma), method='L-BFGS-B',
                          bounds=bounds)
        mu, lambda_ = result.x
        return mu, lambda_, alpha, beta

    @staticmethod
    def log_likelihood(params, x, sigma, alpha, beta):
        mu, lambda_ = params
        gamma_alpha = gammaln(alpha)
        log_pdf = (alpha * np.log(beta) + 0.5 * np.log(lambda_) - alpha * np.log(sigma)
                   - 0.5 * lambda_ * (x - mu) ** 2 / sigma - beta / sigma
                   - (alpha + 1) * np.log(sigma) - 0.5 * np.log(2 * np.pi) - gamma_alpha)
        return -np.sum(log_pdf)

    @staticmethod
    def normal_inverse_gamma(x, sigma, mu, lambd, alpha, beta):
        return (
                norm.pdf(x, loc=mu, scale=np.sqrt(1 / (lambd * sigma)))
                * invgamma.pdf(sigma, alpha, scale=beta)
        )

    @staticmethod
    def aleatoric_uncertainy_from_nig(alpha, beta, k=1):
        return k * (beta / (alpha - 1))

    @staticmethod
    def epistemic_uncertainy_from_nig(lambda_, alpha, beta, k=1):
        return k * (beta / ((alpha - 1) * lambda_))


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
