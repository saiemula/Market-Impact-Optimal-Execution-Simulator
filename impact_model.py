"""
impact_model.py
Implements Almgren-Chriss style market impact model.
"""
import numpy as np

class AlmgrenChrissImpact:
    """
    Models temporary and permanent market impact, execution cost, slippage, and recovery.
    """
    def __init__(self, gamma=2e-6, eta=1e-3, sigma=0.02, T=1.0, N=100):
        """
        Args:
            gamma: Permanent impact coefficient
            eta: Temporary impact coefficient
            sigma: Volatility
            T: Total execution time
            N: Number of slices
        """
        self.gamma = gamma
        self.eta = eta
        self.sigma = sigma
        self.T = T
        self.N = N
        self.dt = T / N

    def optimal_trajectory(self, X, lambda_):
        """
        Computes optimal execution trajectory (Almgren-Chriss solution).
        Args:
            X: Total shares to execute
            lambda_: Risk aversion parameter
        Returns:
            x: Array of shares to execute at each time step
        """
        kappa = np.sqrt(lambda_ * self.sigma**2 / self.eta)
        times = np.linspace(0, self.T, self.N)
        x = X * np.sinh(kappa * (self.T - times)) / np.sinh(kappa * self.T)
        return x

    def temporary_impact(self, v):
        """
        Temporary impact: eta * v
        v: execution rate
        """
        return self.eta * v

    def permanent_impact(self, X):
        """
        Permanent impact: gamma * X
        X: total shares executed
        """
        return self.gamma * X

    def execution_cost(self, x, v):
        """
        Total execution cost: sum of temporary and permanent impact
        x: shares executed
        v: execution rate
        """
        temp = self.temporary_impact(v)
        perm = self.permanent_impact(np.sum(x))
        return np.sum(temp) + perm

    def slippage(self, realized_cost, expected_cost):
        """
        Slippage: realized - expected cost
        """
        return realized_cost - expected_cost

    def recovery_curve(self, t, tau=0.2):
        """
        Market recovery curve: exponential decay
        t: time since shock
        tau: recovery time constant
        """
        return np.exp(-t / tau)
