"""
visualization.py
Interactive Plotly visualizations for market impact and execution.
"""
import plotly.graph_objs as go
import plotly.express as px
import numpy as np
import pandas as pd

DARK_TEMPLATE = 'plotly_dark'

class Visualizer:
    """
    Professional dark quant aesthetic visualizations.
    """
    @staticmethod
    def impact_surface(order_sizes, volatilities, costs):
        fig = go.Figure(data=[go.Surface(z=costs, x=order_sizes, y=volatilities)])
        fig.update_layout(title='3D Impact Surface',
                          scene=dict(xaxis_title='Order Size', yaxis_title='Volatility', zaxis_title='Cost'),
                          template=DARK_TEMPLATE)
        fig.show()

    @staticmethod
    def execution_trajectory(prices):
        fig = px.line(prices, title='Execution Trajectory', template=DARK_TEMPLATE)
        fig.update_xaxes(title='Time')
        fig.update_yaxes(title='Price')
        fig.show()

    @staticmethod
    def slippage_histogram(slippages):
        fig = px.histogram(slippages, nbins=30, title='Slippage Distribution', template=DARK_TEMPLATE)
        fig.update_xaxes(title='Slippage')
        fig.update_yaxes(title='Frequency')
        fig.show()

    @staticmethod
    def liquidity_recovery_curve(times, recovery):
        fig = px.line(x=times, y=recovery, title='Liquidity Recovery Curve', template=DARK_TEMPLATE)
        fig.update_xaxes(title='Time')
        fig.update_yaxes(title='Recovery')
        fig.show()

    @staticmethod
    def cost_heatmap(order_sizes, volatilities, costs):
        fig = px.imshow(costs, x=order_sizes, y=volatilities, color_continuous_scale='Viridis',
                        title='Cost Sensitivity Heatmap', template=DARK_TEMPLATE)
        fig.update_xaxes(title='Order Size')
        fig.update_yaxes(title='Volatility')
        fig.show()
