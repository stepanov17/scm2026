import numpy as np
from scipy.optimize import minimize
from scipy.special import beta as beta_func


class KumaraswamyFitter:
    """
    Fit Kumaraswamy distribution on [a, b] using weighted least squares.

    PDF: f(x) = (αβ/(b-a)) * ((x-a)/(b-a))^(α-1) * (1 - ((x-a)/(b-a))^α)^(β-1)
    CDF: F(x) = 1 - (1 - ((x-a)/(b-a))^α)^β
    """

    def __init__(self):
        self.fitted_params = None
        self.phi_min = None

    def quantile(self, p, alpha, beta, a, b):
        """Quantile function (inverse CDF)."""
        if p <= 0:
            return a
        if p >= 1:
            return b
        q_std = (1.0 - (1.0 - p) ** (1.0 / beta)) ** (1.0 / alpha)
        return a + (b - a) * q_std

    def theoretical_mean(self, alpha, beta, a, b):
        """Mean of Kumaraswamy on [a, b]."""
        mu_std = beta * beta_func(1 + 1.0 / alpha, beta)  # Use beta_func
        return a + (b - a) * mu_std

    def theoretical_var(self, alpha, beta, a, b):
        """Variance of Kumaraswamy on [a, b]."""
        e_u = beta * beta_func(1 + 1.0 / alpha, beta)
        e_u2 = beta * beta_func(1 + 2.0 / alpha, beta)
        var_std = e_u2 - e_u ** 2
        return (b - a) ** 2 * var_std

    def theoretical_std(self, alpha, beta, a, b):
        return np.sqrt(self.theoretical_var(alpha, beta, a, b))

    def _objective(self, params, mu_obs, u_obs, c1_obs, c2_obs, p_level, weights):
        alpha, beta, a, b = params
        w_mu, w_u, w_c1, w_c2 = weights

        if alpha <= 0 or beta <= 0 or a >= b:
            return 1e10

        mu_theta = self.theoretical_mean(alpha, beta, a, b)
        u_theta = self.theoretical_std(alpha, beta, a, b)

        p_lower = (1 - p_level) / 2
        p_upper = (1 + p_level) / 2
        c1_theta = self.quantile(p_lower, alpha, beta, a, b)
        c2_theta = self.quantile(p_upper, alpha, beta, a, b)

        phi = (w_mu * (mu_theta - mu_obs) ** 2 +
               w_u * (u_theta - u_obs) ** 2 +
               w_c1 * (c1_theta - c1_obs) ** 2 +
               w_c2 * (c2_theta - c2_obs) ** 2)
        return phi

    def fit(self, mu, u, c1, c2, p_level=0.95,
            w_mu=1.0, w_u=1.0, w_c1=1.0, w_c2=1.0,
            initial_guess=None, method='L-BFGS-B'):
        """Fit parameters using weighted least squares."""

        if initial_guess is None:
            margin = (c2 - c1) * 0.1
            a_guess = c1 - margin
            b_guess = c2 + margin
            initial_guess = np.array([2.0, 2.0, a_guess, b_guess])

        weights = (w_mu, w_u, w_c1, w_c2)
        bounds = [(1e-6, None), (1e-6, None), (None, None), (None, None)]

        def obj(params):
            return self._objective(params, mu, u, c1, c2, p_level, weights)

        result = minimize(obj, initial_guess, method=method, bounds=bounds)

        self.fitted_params = result.x
        self.phi_min = result.fun
        return result.success

    def get_params(self):
        return self.fitted_params

    def predict(self, params=None, p_level=0.95):
        if params is None:
            params = self.fitted_params
        if params is None:
            return None

        alpha, beta, a, b = params
        mu = self.theoretical_mean(alpha, beta, a, b)
        u = self.theoretical_std(alpha, beta, a, b)

        p_lower = (1 - p_level) / 2
        p_upper = (1 + p_level) / 2
        c1 = self.quantile(p_lower, alpha, beta, a, b)
        c2 = self.quantile(p_upper, alpha, beta, a, b)

        return {'mean': mu, 'std': u, 'c1': c1, 'c2': c2}


if __name__ == "__main__":

    print("example: Kumaraswamy distribution")

    # True parameters
    true_alpha = 3.0
    true_beta = 2.0
    true_a = 0.0
    true_b = 2.0

    fitter = KumaraswamyFitter()
    true_mu = fitter.theoretical_mean(true_alpha, true_beta, true_a, true_b)
    true_u = fitter.theoretical_std(true_alpha, true_beta, true_a, true_b)
    true_c1 = fitter.quantile(0.025, true_alpha, true_beta, true_a, true_b)
    true_c2 = fitter.quantile(0.975, true_alpha, true_beta, true_a, true_b)

    print(f"\nTrue parameters: α={true_alpha}, β={true_beta}, a={true_a}, b={true_b}")
    print(f"Theoretical: μ={true_mu:.6f}, σ={true_u:.6f}, 95% CI=[{true_c1:.6f}, {true_c2:.6f}]")

    # Fit using these observed values
    fitter.fit(true_mu, true_u, true_c1, true_c2, p_level=0.95)

    alpha_fit, beta_fit, a_fit, b_fit = fitter.get_params()
    pred = fitter.predict(p_level=0.95)

    print(f"\nFitted parameters:")
    print(f"  α = {alpha_fit:.6f}")
    print(f"  β = {beta_fit:.6f}")
    print(f"  a = {a_fit:.6f}")
    print(f"  b = {b_fit:.6f}")
    print(f"\nObjective function Φ = {fitter.phi_min:.6e}")
    print(f"\nReconstructed characteristics:")
    print(f"  μ = {pred['mean']:.6f} (target: {true_mu})")
    print(f"  σ = {pred['std']:.6f} (target: {true_u})")
    print(f"  c1 = {pred['c1']:.6f} (target: {true_c1})")
    print(f"  c2 = {pred['c2']:.6f} (target: {true_c2})")
