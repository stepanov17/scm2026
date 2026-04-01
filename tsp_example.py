import numpy as np
from scipy.optimize import minimize
import sys


class TwoSidedPowerFitter:
    """
    Fit Two-Sided Power distribution with parameters (a, m, b, p).

    PDF: f(x) = (p/(b-a)) * ((x-a)/(m-a))^(p-1)   for a < x ≤ m
          f(x) = (p/(b-a)) * ((b-x)/(b-m))^(p-1)   for m ≤ x < b

    Parameters:
        a, b : support bounds
        m : mode (a < m < b)
        p : shape parameter (p > 0)
    """

    def __init__(self):
        self.fitted_params = None
        self.phi_min = None
        self.success = False

    def _cdf(self, x, a, m, b, p):
        """CDF of Two-Sided Power distribution."""
        x = np.asarray(x)
        result = np.zeros_like(x, dtype=float)

        # Handle scalar and array inputs
        if np.isscalar(x):
            if x <= a:
                return 0.0
            if x >= b:
                return 1.0
        else:
            result[x <= a] = 0.0
            result[x >= b] = 1.0

        # CDF at mode
        f_mode = (m - a) / (b - a)

        # Left branch
        if np.isscalar(x):
            if a < x <= m:
                return f_mode * ((x - a) / (m - a)) ** p
            elif m < x < b:
                return f_mode + (1 - f_mode) * (1 - ((b - x) / (b - m)) ** p)
            else:
                return result
        else:
            mask_left = (x > a) & (x <= m)
            mask_right = (x > m) & (x < b)
            result[mask_left] = f_mode * ((x[mask_left] - a) / (m - a)) ** p
            result[mask_right] = f_mode + (1 - f_mode) * (1 - ((b - x[mask_right]) / (b - m)) ** p)
            return result

    def quantile(self, p_level, a, m, b, p):
        """Quantile function (inverse CDF)."""
        if p_level <= 0:
            return a
        if p_level >= 1:
            return b

        # CDF at mode
        f_mode = (m - a) / (b - a)

        if p_level <= f_mode:
            # Left branch
            t = (p_level / f_mode) ** (1.0 / p)
            return a + (m - a) * t
        else:
            # Right branch
            t = ((1 - p_level) / (1 - f_mode)) ** (1.0 / p)
            return b - (b - m) * t

    def theoretical_mean(self, a, m, b, p):
        """Mean of Two-Sided Power distribution."""
        if p <= 0:
            return (a + b) / 2

        f_mode = (m - a) / (b - a)

        # Left branch: E[X | X ≤ m]
        left_mean = a + (m - a) * p / (p + 1)

        # Right branch: E[X | X ≥ m]
        right_mean = m + (b - m) / (p + 1)

        # Weighted average
        return f_mode * left_mean + (1 - f_mode) * right_mean

    def _theoretical_second_moment(self, a, m, b, p):
        """E[X^2] of Two-Sided Power distribution."""
        if p <= 0:
            # Uniform distribution on [a, b]
            return (a ** 2 + a * b + b ** 2) / 3

        f_mode = (m - a) / (b - a)

        # Left branch: E[X^2 | X ≤ m]
        left_e2 = a ** 2 + 2 * a * (m - a) * p / (p + 1) + (m - a) ** 2 * p / (p + 2)

        # Right branch: E[X^2 | X ≥ m]
        right_e2 = b ** 2 - 2 * b * (b - m) / (p + 1) + (b - m) ** 2 / (p + 2)

        return f_mode * left_e2 + (1 - f_mode) * right_e2

    def theoretical_std(self, a, m, b, p):
        """Standard deviation of Two-Sided Power distribution."""
        try:
            e_x = self.theoretical_mean(a, m, b, p)
            e_x2 = self._theoretical_second_moment(a, m, b, p)
            var = e_x2 - e_x ** 2
            # Numerical protection
            if var < 0:
                var = max(var, 0.0)
            return np.sqrt(var)
        except (ValueError, OverflowError):
            return 1.0

    def _objective(self, params, mu_obs, u_obs, a_obs, b_obs, c1_obs, c2_obs, p_level, weights):
        """
        Objective function: weighted sum of squared deviations.

        Parameters to estimate: a, m, b, p (4 parameters)
        User provides: μ, u, a, b, c₁, c₂ (6 characteristics)
        """
        a, m, b, p = params
        w_mu, w_u, w_a, w_b, w_c1, w_c2 = weights

        # Constraints: a < m < b, p > 0
        if a >= m or m >= b or p <= 1e-6:
            return 1e10

        # Protect against extreme p values
        if p > 100:
            p = 100.0

        mu_theta = self.theoretical_mean(a, m, b, p)
        u_theta = self.theoretical_std(a, m, b, p)

        # Check for NaN
        if np.isnan(mu_theta) or np.isnan(u_theta):
            return 1e10

        p_lower = (1 - p_level) / 2
        p_upper = (1 + p_level) / 2
        c1_theta = self.quantile(p_lower, a, m, b, p)
        c2_theta = self.quantile(p_upper, a, m, b, p)

        phi = (w_mu * (mu_theta - mu_obs) ** 2 +
               w_u * (u_theta - u_obs) ** 2 +
               w_a * (a - a_obs) ** 2 +
               w_b * (b - b_obs) ** 2 +
               w_c1 * (c1_theta - c1_obs) ** 2 +
               w_c2 * (c2_theta - c2_obs) ** 2)

        # Add small regularization to avoid numerical issues
        phi += 1e-8 * (a ** 2 + m ** 2 + b ** 2 + p ** 2)

        return phi

    def fit(self, mu, u, a_obs, b_obs, c1, c2, p_level=0.95,
            w_mu=1.0, w_u=1.0, w_a=1.0, w_b=1.0, w_c1=1.0, w_c2=1.0,
            initial_guess=None, method='L-BFGS-B', verbose=False):
        """
        Fit TSP parameters using weighted least squares.

        User provides estimates for: μ, u, a, b, c₁, c₂ (6 characteristics)
        Parameters to estimate: a, m, b, p (4 parameters)
        """
        # Heuristic initial guesses
        if initial_guess is None:
            # a near c1, b near c2
            margin = (c2 - c1) * 0.1
            a_guess = c1 - margin
            b_guess = c2 + margin
            # m near median or mid-range
            m_guess = (c1 + c2) / 2
            # p guess: 2.0 (triangular-like)
            p_guess = 2.0
            initial_guess = np.array([a_guess, m_guess, b_guess, p_guess])

        weights = (w_mu, w_u, w_a, w_b, w_c1, w_c2)

        # Bounds: a < m < b, p > 0
        bounds = [(None, None), (None, None), (None, None), (1e-3, 50.0)]

        def obj(params):
            return self._objective(params, mu, u, a_obs, b_obs, c1, c2, p_level, weights)

        # Try optimization with multiple starting points
        best_result = None
        best_phi = np.inf

        # Generate a few initial guesses
        guesses = [initial_guess]
        for offset in [-0.1, 0.1, -0.2, 0.2]:
            guess_adj = initial_guess.copy()
            guess_adj[0] += offset * abs(guess_adj[0]) if guess_adj[0] != 0 else offset
            guess_adj[1] += offset * abs(guess_adj[1]) if guess_adj[1] != 0 else offset
            guess_adj[2] += offset * abs(guess_adj[2]) if guess_adj[2] != 0 else offset
            guesses.append(guess_adj)

        for guess in guesses:
            result = minimize(obj, guess, method=method, bounds=bounds)
            if result.success and result.fun < best_phi:
                best_phi = result.fun
                best_result = result

        if best_result is not None and best_result.success:
            a_fit, m_fit, b_fit, p_fit = best_result.x
            self.fitted_params = np.array([a_fit, m_fit, b_fit, p_fit])
            self.phi_min = best_result.fun
            self.success = True
            if verbose:
                print(f"Optimization successful: {best_result.message}")
            return True
        else:
            self.fitted_params = None
            self.phi_min = None
            self.success = False
            if verbose:
                print(f"Optimization failed: {best_result.message if best_result else 'No valid result'}")
            return False

    def get_params(self):
        """Return fitted parameters (a, m, b, p)."""
        return self.fitted_params

    def predict(self, params=None, p_level=0.95):
        """Compute theoretical characteristics for given parameters."""
        if params is None:
            params = self.fitted_params
        if params is None:
            return None

        a, m, b, p = params
        mu = self.theoretical_mean(a, m, b, p)
        u = self.theoretical_std(a, m, b, p)

        p_lower = (1 - p_level) / 2
        p_upper = (1 + p_level) / 2
        c1 = self.quantile(p_lower, a, m, b, p)
        c2 = self.quantile(p_upper, a, m, b, p)

        return {'mean': mu, 'std': u, 'c1': c1, 'c2': c2}

    def pdf(self, x, params=None):
        """Compute PDF at given points."""
        if params is None:
            params = self.fitted_params
        if params is None:
            return None

        a, m, b, p = params
        x = np.asarray(x)
        pdf_vals = np.zeros_like(x, dtype=float)

        # Left branch
        mask_left = (x > a) & (x <= m)
        if np.any(mask_left):
            pdf_vals[mask_left] = (p / (b - a)) * ((x[mask_left] - a) / (m - a)) ** (p - 1)

        # Right branch
        mask_right = (x >= m) & (x < b)
        if np.any(mask_right):
            pdf_vals[mask_right] = (p / (b - a)) * ((b - x[mask_right]) / (b - m)) ** (p - 1)

        # Mode at x = m
        pdf_vals[x == m] = p / (b - a)

        return pdf_vals


def main():

    # True parameters
    true_a = 0.0
    true_m = 0.8
    true_b = 2.0
    true_p = 3.0

    fitter = TwoSidedPowerFitter()

    # Compute theoretical characteristics (these would be user's observations)
    true_mu = fitter.theoretical_mean(true_a, true_m, true_b, true_p)
    true_u = fitter.theoretical_std(true_a, true_m, true_b, true_p)
    true_c1 = fitter.quantile(0.025, true_a, true_m, true_b, true_p)
    true_c2 = fitter.quantile(0.975, true_a, true_m, true_b, true_p)

    print(f"\nTrue parameters: a={true_a}, m={true_m}, b={true_b}, p={true_p}")
    print(f"\nTheoretical characteristics:")
    print(f"  μ = {true_mu:.6f}")
    print(f"  σ = {true_u:.6f}")
    print(f"  a = {true_a:.6f}")
    print(f"  b = {true_b:.6f}")
    print(f"  c₁ (2.5%% quantile) = {true_c1:.6f}")
    print(f"  c₂ (97.5%% quantile) = {true_c2:.6f}")

    # introduce some errors (rounding) and fit
    success = fitter.fit(round(true_mu, 2), round(true_u, 3), true_a, true_b, round(true_c1, 2), round(true_c2, 2),
                         p_level=0.95, verbose=True)

    # ... or fit using accurate observed values
    # success = fitter.fit(true_mu, true_u, true_a, true_b, true_c1, true_c2,
    #                      p_level=0.95, verbose=True)

    if not success:
        print("\nFitting failed. Exiting.")
        sys.exit(1)

    fitted_params = fitter.get_params()
    if fitted_params is None:
        print("\nNo fitted parameters available.")
        sys.exit(1)

    a_fit, m_fit, b_fit, p_fit = fitted_params
    pred = fitter.predict(p_level=0.95)

    print(f"\nFitted parameters (6 characteristics, 4 parameters):")
    print(f"  a = {a_fit:.6f}")
    print(f"  m = {m_fit:.6f}")
    print(f"  b = {b_fit:.6f}")
    print(f"  p = {p_fit:.6f}")
    print(f"\nObjective function Φ = {fitter.phi_min:.6e}")
    print(f"\nReconstructed characteristics:")
    print(f"  μ = {pred['mean']:.3f} (target: {true_mu})")
    print(f"  σ = {pred['std']:.3f} (target: {true_u})")
    print(f"  c₁ = {pred['c1']:.3f} (target: {true_c1})")
    print(f"  c₂ = {pred['c2']:.3f} (target: {true_c2})")

    # Check residuals
    print(f"\nResiduals:")
    print(f"  Δμ = {pred['mean'] - true_mu:.6e}")
    print(f"  Δσ = {pred['std'] - true_u:.6e}")
    print(f"  Δc₁ = {pred['c1'] - true_c1:.6e}")
    print(f"  Δc₂ = {pred['c2'] - true_c2:.6e}")

    # optional: plot PDF
    # try:
    #     import matplotlib.pyplot as plt
    #
    #     x = np.linspace(true_a - 0.1, true_b + 0.1, 500)
    #     pdf_true = fitter.pdf(x, [true_a, true_m, true_b, true_p])
    #     pdf_fitted = fitter.pdf(x, [a_fit, m_fit, b_fit, p_fit])
    #
    #     plt.figure(figsize=(10, 5))
    #     plt.plot(x, pdf_true, 'b-', label='True PDF', linewidth=2)
    #     plt.plot(x, pdf_fitted, 'r--', label='Fitted PDF', linewidth=2)
    #     plt.axvline(true_m, color='b', linestyle=':', alpha=0.5, label='True mode')
    #     plt.axvline(m_fit, color='r', linestyle=':', alpha=0.5, label='Fitted mode')
    #     plt.xlabel('x')
    #     plt.ylabel('f(x)')
    #     plt.title('Two-Sided Power Distribution: True vs Fitted')
    #     plt.legend()
    #     plt.grid(True, alpha=0.3)
    #     plt.show()
    # except ImportError:
    #     print("\nMatplotlib not available for plotting.")
    # except Exception as e:
    #     print(f"\nPlotting error: {e}")


if __name__ == "__main__":

    main()
