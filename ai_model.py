"""
ai_model.py
AI extension for predicting execution cost and optimal slicing.
"""
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor
import torch
import torch.nn as nn

class ExecutionCostRegressor:
    """
    Regression model using XGBoost to predict execution cost.
    """
    def __init__(self):
        self.model = XGBRegressor(objective='reg:squarederror')

    def train(self, X, y):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        self.model.fit(X_train, y_train)
        preds = self.model.predict(X_test)
        mse = mean_squared_error(y_test, preds)
        return mse

    def predict(self, X):
        return self.model.predict(X)

class SlicingNN(nn.Module):
    """
    Neural network to predict optimal slicing strategy.
    """
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 32)
        self.fc2 = nn.Linear(32, 32)
        self.fc3 = nn.Linear(32, output_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x
