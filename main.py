"""
main.py
Entry point for Market Impact & Optimal Execution Simulator.
"""
import numpy as np
from simulator import MarketSimulator
from impact_model import AlmgrenChrissImpact
from ai_model import ExecutionCostRegressor, SlicingNN
from visualization import Visualizer

# --- Simulation Parameters ---
N_STEPS = 500
INITIAL_PRICE = 100.0
ORDER_BOOK_DEPTH = 10000
VOLATILITY = 0.02
LIQUIDITY_DECAY = 0.001
ORDER_SIZE = 2000
PARTICIPATION_RATE = 0.1

# --- Run Simulation ---
sim = MarketSimulator(n_steps=N_STEPS, initial_price=INITIAL_PRICE,
                     order_book_depth=ORDER_BOOK_DEPTH,
                     volatility=VOLATILITY, liquidity_decay=LIQUIDITY_DECAY)
sim.inject_order(step=50, order_size=ORDER_SIZE, participation_rate=PARTICIPATION_RATE)
sim.simulate()
state = sim.get_state()

# --- Impact Model ---
impact = AlmgrenChrissImpact(gamma=2e-6, eta=1e-3, sigma=VOLATILITY, T=1.0, N=N_STEPS)
trajectory = impact.optimal_trajectory(X=ORDER_SIZE, lambda_=0.01)
v = np.diff(trajectory, prepend=trajectory[0]) / impact.dt
cost = impact.execution_cost(trajectory, v)
slippage = impact.slippage(cost, expected_cost=ORDER_SIZE * INITIAL_PRICE)
recovery = impact.recovery_curve(np.linspace(0, 1, N_STEPS))

# --- AI Extension ---
X = np.column_stack([state['shock'], state['liquidity'], state['price']])
y = np.full(N_STEPS, cost)
regressor = ExecutionCostRegressor()
mse = regressor.train(X, y)
predicted_cost = regressor.predict(X)

# --- Visualization ---
Visualizer.impact_surface(order_sizes=np.linspace(100, 5000, 30),
                         volatilities=np.linspace(0.01, 0.05, 30),
                         costs=np.outer(np.linspace(100, 5000, 30), np.linspace(0.01, 0.05, 30)) * 1e-3)
Visualizer.execution_trajectory(state['price'])
Visualizer.slippage_histogram([slippage])
Visualizer.liquidity_recovery_curve(np.linspace(0, 1, N_STEPS), recovery)
Visualizer.cost_heatmap(order_sizes=np.linspace(100, 5000, 30),
                       volatilities=np.linspace(0.01, 0.05, 30),
                       costs=np.outer(np.linspace(100, 5000, 30), np.linspace(0.01, 0.05, 30)) * 1e-3)
