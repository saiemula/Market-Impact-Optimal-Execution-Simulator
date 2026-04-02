
# Market Impact & Optimal Execution Simulator

**Creator/Dev:** tubakhxn
<img width="2991" height="1626" alt="market" src="https://github.com/user-attachments/assets/27fca01f-a1dd-49dd-a7d3-6712e3605468" />

## What is this project about?
This project is a professional quantitative finance simulator for modeling and visualizing the market impact of large order executions. It uses microstructure theory and the Almgren-Chriss model to simulate order book depth, price impact, volatility, and liquidity decay. The system can inject large order shocks, compute execution cost, slippage, and market recovery, and includes an AI extension to predict execution cost and optimal slicing strategies. Interactive Plotly visualizations provide deep insight into market dynamics with a dark quant aesthetic.

## Files
- simulator.py: Core simulation engine for order book, price impact, volatility, and liquidity decay
- impact_model.py: Almgren-Chriss style market impact model (temporary/permanent impact, cost, slippage, recovery)
- ai_model.py: AI models (XGBoost, Neural Network) for predicting execution cost and optimal slicing
- visualization.py: Interactive Plotly visualizations (3D impact surface, trajectory, slippage, recovery, heatmap)
- main.py: Entry point to run the simulation, modeling, AI, and visualization

## How to fork this project
1. Click the "Fork" button on the top right of the GitHub repository page.
2. Clone your forked repository to your local machine:
	```
	git clone https://github.com/your-username/Market-Impact-Optimal-Execution-Simulator.git
	```
3. Install Python 3.8+ and required packages:
	```
	pip install numpy pandas plotly xgboost scikit-learn torch
	```
4. Run `main.py` to start the simulator:
	```
	python main.py
	```

## Requirements
- Python 3.8+
- numpy, pandas, plotly, xgboost, scikit-learn, torch

## Usage
Run main.py to start the simulation and view results.

## Documentation
All financial formulas are documented in code comments.

