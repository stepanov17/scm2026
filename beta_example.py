import numpy as np
from scipy.optimize import minimize, approx_fprime
from scipy.special import betaincinv
from typing import Optional
import warnings


class BetaDistributionFitter:
    """
    Fit beta distribution on [a, b] using weighted least squares.
    """

    def __init__(self):
        self.fitted_params = None
        self.covariance = None
        self.phi_min = None

    def _theoretical_mean(self, alpha, beta_param, a, b):
        return a + (b - a) * alpha / (alpha + beta_param)

    def _theoretical_std(self, alpha, beta_param, a, b):
        total = alpha + beta_param
        var = (b - a) ** 2 * alpha * beta_param / (total ** 2 * (total + 1))
        return np.sqrt(var)

    def _theoretical_quantile(self, alpha, beta_param, a, b, p):
        if p <= 0:
            return a
        if p >= 1:
            return b
        q_std = betaincinv(alpha, beta_param, p)
        return a + (b - a) * q_std

    def _objective(self, params, mu_obs, u_obs, c1_obs, c2_obs, p_level, weights):
        alpha, beta_param, a, b = params
        w_mu, w_u, w_c1, w_c2 = weights

        if alpha <= 0 or beta_param <= 0 or a >= b:
            return 1e10

        mu_theta = self._theoretical_mean(alpha, beta_param, a, b)
        u_theta = self._theoretical_std(alpha, beta_param, a, b)

        p_lower = (1 - p_level) / 2
        p_upper = (1 + p_level) / 2
        c1_theta = self._theoretical_quantile(alpha, beta_param, a, b, p_lower)
        c2_theta = self._theoretical_quantile(alpha, beta_param, a, b, p_upper)

        phi = (w_mu * (mu_theta - mu_obs) ** 2 +
               w_u * (u_theta - u_obs) ** 2 +
               w_c1 * (c1_theta - c1_obs) ** 2 +
               w_c2 * (c2_theta - c2_obs) ** 2)
        return phi

    def fit(self, mu, u, c1, c2, p_level=0.9,
            w_mu=1.0, w_u=1.0, w_c1=1.0, w_c2=1.0,
            initial_guess=None, method='L-BFGS-B'):
        """Fit parameters using weighted least squares."""

        # Heuristic initial guess
        if initial_guess is None:
            margin = (c2 - c1) * 0.1
            a_guess = c1 - margin
            b_guess = c2 + margin
            mu_std = (mu - a_guess) / (b_guess - a_guess)
            mu_std = np.clip(mu_std, 0.01, 0.99)
            # Rough guess for shape parameters
            alpha_guess = 1.0
            beta_guess = 1.0
            initial_guess = np.array([alpha_guess, beta_guess, a_guess, b_guess])

        weights = (w_mu, w_u, w_c1, w_c2)
        bounds = [(1e-6, None), (1e-6, None), (None, None), (None, None)]

        def obj(params):
            return self._objective(params, mu, u, c1, c2, p_level, weights)

        result = minimize(obj, initial_guess, method=method, bounds=bounds)

        self.fitted_params = result.x
        self.phi_min = result.fun
        return result.success

    def get_params(self):
        """Return fitted parameters (alpha, beta, a, b)."""
        if self.fitted_params is None:
            return None
        return self.fitted_params

    def predict(self, params=None, p_level=0.9):
        """Compute theoretical characteristics for given parameters."""
        if params is None:
            params = self.fitted_params
        if params is None:
            return None

        alpha, beta_param, a, b = params
        mu = self._theoretical_mean(alpha, beta_param, a, b)
        u = self._theoretical_std(alpha, beta_param, a, b)

        p_lower = (1 - p_level) / 2
        p_upper = (1 + p_level) / 2
        c1 = self._theoretical_quantile(alpha, beta_param, a, b, p_lower)
        c2 = self._theoretical_quantile(alpha, beta_param, a, b, p_upper)

        return {'mean': mu, 'std': u, 'c1': c1, 'c2': c2}


if __name__ == "__main__":

    # Input data: true parameters (alpha=6, beta=2, a=0, b=2)
    # Theoretical values: mu=1.5, u=0.288675, c1=0.9556, c2=1.8178
    mu_obs = 1.5
    u_obs = 0.289
    c1_obs = 0.955
    c2_obs = 1.818
    p_level = 0.9

    fitter = BetaDistributionFitter()
    fitter.fit(mu_obs, u_obs, c1_obs, c2_obs, p_level,
               w_mu=1.0, w_u=1.0, w_c1=1.0, w_c2=1.0)

    alpha, beta, a, b = fitter.get_params()
    pred = fitter.predict(p_level=p_level)

    print(f"\nFitted parameters:")
    print(f"  α = {alpha:.6f}")
    print(f"  β = {beta:.6f}")
    print(f"  a = {a:.6f}")
    print(f"  b = {b:.6f}")
    print(f"\nObjective function Φ = {fitter.phi_min:.6e}")
    print(f"\nTheoretical characteristics:")
    print(f"  μ = {pred['mean']:.6f} (observed: {mu_obs})")
    print(f"  u = {pred['std']:.6f} (observed: {u_obs})")
    print(f"  c1 = {pred['c1']:.6f} (observed: {c1_obs})")
    print(f"  c2 = {pred['c2']:.6f} (observed: {c2_obs})")
