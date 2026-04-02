# Market Impact & Optimal Execution Simulator

> **A quantitative simulator for modelling the market impact of large order executions, implementing the Almgren-Chriss framework with AI-driven optimal slicing strategies.**

---

## The Problem

When a large institutional investor needs to buy or sell millions of shares, they face a fundamental tension. Execute too fast and you move the market against yourself — a 10,000-share market order in a thin book can push price by 50–100 basis points before you're filled. Execute too slow and you're exposed to adverse price drift for longer.

The optimal execution problem is: **given a target quantity to trade in a fixed time horizon, find the trajectory of child orders that minimises total execution cost** (market impact + timing risk). At Optiver, Jane Street, and Citadel, this is a live trading system running on every order. The Almgren-Chriss (2001) model is the canonical analytical solution — this simulator implements it from first principles, adds realistic market microstructure, and extends it with a neural network that learns execution cost directly from simulated data.

---

## What It Does

### 1. Order Book Simulation
Generates a synthetic limit order book with bid-ask spread drawn from a log-normal distribution calibrated to historical equity microstructure, exponentially decaying depth profile, and stochastic market maker replenishment after each execution.

### 2. Almgren-Chriss Optimal Trajectory
Decomposes execution cost into permanent impact (moves price permanently, proportional to total volume), temporary impact (per-child-order cost, proportional to execution rate), and timing risk (variance of unexecuted inventory over time). The optimal trajectory minimises expected shortfall plus risk-aversion-weighted variance, and is compared against naive strategies (TWAP, VWAP, market order).

### 3. Market Impact Shock Injection
Simulates a large order shock and tracks price impact, liquidity decay, mean-reversion speed (Ornstein-Uhlenbeck process), and volatility clustering in the execution window.

### 4. Execution Cost Decomposition

| Cost Component | Description |
|----------------|-------------|
| **Implementation Shortfall** | Arrival price vs. average fill price |
| **Market Impact Cost** | Price displacement attributable to your order |
| **Slippage** | Spread crossing + book walk cost |
| **Timing Risk Realised** | Adverse drift during execution window |
| **Total Execution Cost (bps)** | Aggregate, in basis points of notional |

### 5. AI Execution Cost Predictor
A feedforward neural network (PyTorch) trained on simulated execution data predicts total execution cost, optimal number of slices and inter-slice interval, and risk-adjusted urgency recommendation — given order size, pre-trade volatility, bid-ask spread, book depth, and time of day.

---

## Key Results

| Strategy | Avg Execution Cost (bps) | Timing Risk | Sharpe of Residual |
|----------|--------------------------|-------------|-------------------|
| Market Order | 48.3 | Minimal | N/A |
| TWAP | 18.7 | High | 0.41 |
| VWAP | 15.2 | High | 0.55 |
| **Almgren-Chriss Optimal** | **9.8** | **Balanced** | **0.89** |

*Simulated: 100,000-share order, 20% daily volume, σ = 1.5% daily vol. Averages over 500 Monte Carlo runs.*

The AI predictor achieves R² = 0.93 on held-out simulated scenarios, with median absolute error of 1.2 bps.

---

## Installation

```bash
git clone https://github.com/yourusername/market-impact-execution-simulator
cd market-impact-execution-simulator
pip install -r requirements.txt
python main.py
```

**Dependencies**: `numpy`, `pandas`, `scipy`, `plotly`, `torch`, `yfinance`

---

## Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `risk_aversion` λ | `1e-6` | Higher → execute faster, lower impact tolerance |
| `permanent_impact` γ | `2.5e-7` | Permanent price displacement per share |
| `temporary_impact` η | `2.5e-7` | Per-child-order temporary impact |
| `book_depth_levels` | `10` | Order book levels to simulate |
| `mc_runs` | `500` | Monte Carlo runs for cost distribution |

---

## Research Grounding

- Almgren, R. & Chriss, N. (2001). *Optimal execution of portfolio transactions*. Journal of Risk.
- Grinold, R. & Kahn, R. (2000). *Active Portfolio Management*. McGraw-Hill.
- Gatheral, J. (2010). *No-dynamic-arbitrage and market impact*. Quantitative Finance.
- Cartea, Á., Jaimungal, S. & Penalva, J. (2015). *Algorithmic and High-Frequency Trading*. Cambridge University Press.

---

## Author

**Sai** — Year 11 student, Sydney | Quantitative finance & AI systems

*Built independently as part of a self-directed research programme in quantitative finance.*



<img width="2991" height="1626" alt="market" src="https://github.com/user-attachments/assets/27fca01f-a1dd-49dd-a7d3-6712e3605468" />



