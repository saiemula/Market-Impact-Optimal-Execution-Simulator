"""
simulator.py
Core simulation engine for market impact and optimal execution.
"""
import numpy as np
import pandas as pd

class MarketSimulator:
    """
    Simulates order book, price impact, volatility, and liquidity decay.
    """
    def __init__(self, n_steps=1000, initial_price=100.0, order_book_depth=10000,
                 volatility=0.02, liquidity_decay=0.001):
        self.n_steps = n_steps
        self.initial_price = initial_price
        self.order_book_depth = order_book_depth
        self.volatility = volatility
        self.liquidity_decay = liquidity_decay
        self.reset()

    def reset(self):
        self.prices = np.full(self.n_steps, self.initial_price, dtype=float)
        self.depths = np.full(self.n_steps, self.order_book_depth, dtype=float)
        self.liquidity = np.full(self.n_steps, self.order_book_depth, dtype=float)
        self.shocks = np.zeros(self.n_steps, dtype=float)

    def inject_order(self, step, order_size, participation_rate):
        """
        Injects a large order shock at a given step.
        Args:
            step (int): Time step to inject order.
            order_size (float): Size of the order.
            participation_rate (float): Fraction of market volume.
        """
        self.shocks[step] += order_size
        self.liquidity[step:] -= order_size * participation_rate

    def simulate(self):
        """
        Runs the simulation, updating price and liquidity.
        """
        for t in range(1, self.n_steps):
            # Volatility shock
            dP = np.random.normal(0, self.volatility)
            # Price impact from order shock
            impact = self.shocks[t] / max(self.liquidity[t], 1)
            # Liquidity decay
            self.liquidity[t] = max(self.liquidity[t-1] * (1 - self.liquidity_decay), 1)
            # Price update
            self.prices[t] = self.prices[t-1] + dP - impact
            # Depth update
            self.depths[t] = max(self.depths[t-1] - self.shocks[t], 1)

    def get_state(self):
        return pd.DataFrame({
            'price': self.prices,
            'depth': self.depths,
            'liquidity': self.liquidity,
            'shock': self.shocks
        })
