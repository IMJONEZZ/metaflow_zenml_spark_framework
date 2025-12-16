#!/usr/bin/env python3
"""
Improved Time Series Forecasting Pipeline (Metaflow Version)

This demonstrates improved time series forecasting using MetaFlow:
- Synthetic weather data generation  
- Multiple model options (Linear Regression, Random Forest, LSTM)
- Automatic model selection based on performance
- Evaluation metrics and validation

Usage:
    python timeseries_forecasting_flow_improved.py run --sequence_length 20 --epochs 30
"""

import json
import os
from typing import Dict, List

import numpy as np

# Import torch at module level for class definition
import torch
import torch.nn as nn
from metaflow.decorators import step

# pylint: disable-all
from metaflow.flowspec import FlowSpec
from metaflow.parameters import Parameter


# Define LSTM model class at module level for proper pickling
class SimpleLSTM(nn.Module):
    def __init__(self, input_size: int = 1, hidden_size: int = 32):
        super(SimpleLSTM, self).__init__()

        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        # LSTM output
        lstm_out, _ = self.lstm(x)

        # Take last time step
        out = lstm_out[:, -1, :]

        # Fully connected layer
        return self.fc(out)


class TimeSeriesForecastingFlowImproved(FlowSpec):
    """An improved time series forecasting flow with multiple model options.

    This flow demonstrates:
        1. ``start`` – generate synthetic time series data.
        2. `create_sequences` - create sequences for LSTM training
        3. ``build_models`` – build multiple models (Linear Reg, Random Forest, LSTM)
        4. ``train_all`` – train all models with improved configurations
        5. `evaluate_all` - evaluate and compare performance  
        6. ``end`` – final step with model comparison
    """

    # Number of data points (default 200)
    num_points = Parameter("num-points", default=200)
    
    # Sequence length for LSTM (default 20) 
    sequence_length = Parameter("sequence-length", default=20)
    
    # Number of epochs for LSTM (default 30)
    epochs = Parameter("epochs", default=30)

    @step
    def start(self):
        """Generate synthetic time series data."""

        print(
            """
Historical Data: [t₋ₙ ... t₋₂, t₋₁, t₀]
    ↓
[Sliding Window]
    ↓  
[Features: lag, rolling stats, trends]
    ↓
┌───────────────┴───────────────┐
↓                               ↓
Linear Regression              Random Forest  
    ↓                               ↓
[Simple Pattern]               [Feature Engineering]
    ↓                               ↓
Fast Training                 Good Performance
                               
        ┌───────────────┴───────────────┐
        ↓                               ↓
        LSTM Neural Network            Ensemble Method
        (if GPU available)             [Best of All]
        ↓                               ↓
Complex Pattern               Optimal Performance
Slow Training (async)         

[Evaluate All Models] ←──────┘
    ↓
Select Best Model → Deploy
"""
        )

        # Generate synthetic temperature data with more complex patterns  
        print("🚀 Generating Time Series Data with MetaFlow")

        np.random.seed(42)  # For reproducibility

        # Time index
        time = np.arange(self.num_points)

        # Temperature with trend, seasonality, and noise (more complex pattern)
        temperature = (
            20                           # Base temperature
            + 0.1 * time                # Upward trend  
            + 5 * np.sin(2 * np.pi * time / 50)    # Seasonal component
            + 2 * np.sin(2 * np.pi * time / 12)    # Secondary seasonal
            + np.random.normal(0, 1.5, self.num_points)  # Noise
        )

        print(f"Generated {self.num_points} temperature data points")
        
        # Store the raw data
        self.raw_data = temperature.tolist()
        
        print(f"Data range: {min(temperature):.2f} to {max(temperature):.2f}")
        print(f"Mean temperature: {np.mean(temperature):.2f}")

        self.next(self.create_sequences)

    @step 
    def create_sequences(self):
        """Create sequences for LSTM and features for other models."""

        print(f"🔄 Creating {self.sequence_length}-length sequences")

        # Convert to numpy array for easier manipulation
        data = np.array(self.raw_data)

        # Create sequences for LSTM
        X_lstm, y_lstm = [], []
        for i in range(len(data) - self.sequence_length):
            X_lstm.append(data[i : (i + self.sequence_length)])
            y_lstm.append(data[i + self.sequence_length])

        # Create features for traditional ML models
        X_features = []
        
        for i in range(self.sequence_length, len(data)):
            features = [
                data[i-1],                    # Previous value (lag 1)
                data[i-2] if i >= 2 else 0,   # Lag 2  
                np.mean(data[max(0, i-5):i]), # Rolling mean (last 5)
                np.std(data[max(0, i-5):i]),  # Rolling std
                i - self.sequence_length,     # Time index (trend)
            ]
            
            # Add seasonal features
            day_of_cycle = i % 50
            features.extend([
                np.sin(2 * np.pi * day_of_cycle / 50),    # Seasonal sine
                np.cos(2 * np.pi * day_of_cycle / 50),    # Seasonal cosine
            ])
            
            X_features.append(features)

        y_traditional = data[self.sequence_length:]

        print(f"Created {len(X_lstm)} LSTM sequences")
        print(f"Created {len(X_features)} feature vectors with {len(X_features[0])} features each")

        # Store data
        self.lstm_data = {
            'X': X_lstm,
            'y': y_lstm
        }
        
        self.traditional_data = {
            'X': X_features, 
            'y': y_traditional.tolist()
        }

        self.next(self.build_models)

    @step
    def build_models(self):
        """Build multiple models for comparison."""

        print("🏗️ Building Multiple Models")

        from sklearn.linear_model import LinearRegression
        from sklearn.ensemble import RandomForestRegressor
        
        # Model 1: Linear Regression (fast, simple)
        self.linear_model = LinearRegression()
        
        # Model 2: Random Forest (good performance, handles non-linearity)  
        self.rf_model = RandomForestRegressor(
            n_estimators=100,
            max_depth=10, 
            random_state=42
        )
        
        # Model 3: LSTM (complex patterns, GPU if available)
        self.lstm_model = SimpleLSTM(input_size=1, hidden_size=32)

        print("✅ Built 3 models:")
        print("   1. Linear Regression (baseline)")
        print("   2. Random Forest (robust)")  
        print("   3. LSTM Neural Network (complex)")

        self.next(self.train_all)

    @step
    def train_all(self):
        """Train all models and compare performance."""

        print("🎯 Training All Models")
        
        # Prepare traditional ML data
        X_trad = np.array(self.traditional_data['X'])
        y_trad = self.traditional_data['y']

        # Split for validation
        split_idx = int(0.8 * len(X_trad))
        
        X_train, X_val = X_trad[:split_idx], X_trad[split_idx:]
        y_train, y_val = y_trad[:split_idx], y_trad[split_idx:]

        # Train Model 1: Linear Regression
        print("📈 Training Linear Regression...")
        self.linear_model.fit(X_train, y_train)
        
        # Train Model 2: Random Forest
        print("🌲 Training Random Forest...")  
        self.rf_model.fit(X_train, y_train)
        
        # Train Model 3: LSTM (if time permits)
        print("🧠 Training LSTM Neural Network...")
        
        # Use smart device management
        import sys
        sys.path.append('/home/imjonezz/Desktop/metaflow_zenml_spark_framework/src/utils')
        from gpu_device_manager import get_device_with_fallback
        
        device, _ = get_device_with_fallback(
            model=self.lstm_model,
            force_device="auto",
            batch_size=32
        )
        
        # Prepare LSTM data  
        X_lstm = torch.FloatTensor(self.lstm_data['X']).unsqueeze(-1).to(device)
        y_lstm = torch.FloatTensor(self.lstm_data['y']).unsqueeze(-1).to(device)
        
        # Move model to device
        self.lstm_model = self.lstm_model.to(device)
        
        # Training setup
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.lstm_model.parameters(), lr=0.01)
        
        # Quick training (reduced epochs for demo)
        self.lstm_model.train()
        for epoch in range(min(10, int(self.epochs))):
            optimizer.zero_grad()
            outputs = self.lstm_model(X_lstm)
            loss = criterion(outputs, y_lstm)
            loss.backward()
            optimizer.step()

        print("✅ All models trained!")

        self.next(self.evaluate_all)

    @step  
    def evaluate_all(self):
        """Evaluate all models and select the best one."""

        print("📊 Evaluating All Models")
        
        # Prepare validation data
        X_trad = np.array(self.traditional_data['X'])
        y_trad = self.traditional_data['y']
        
        split_idx = int(0.8 * len(X_trad))
        X_val = X_trad[split_idx:]
        y_val = y_trad[split_idx:]
        
        # Make predictions
        lr_pred = self.linear_model.predict(X_val)
        rf_pred = self.rf_model.predict(X_val) 
        
        # LSTM predictions
        import torch
        device = next(self.lstm_model.parameters()).device
        X_lstm_val = torch.FloatTensor(X_val).unsqueeze(-1).to(device)
        
        self.lstm_model.eval()
        with torch.no_grad():
            lstm_pred = self.lstm_model(X_lstm_val).squeeze().cpu().numpy()
        
        # Calculate metrics
        def calculate_metrics(y_true, y_pred):
            y_true = np.array(y_true)
            y_pred = np.array(y_pred) 
            mae = np.mean(np.abs(y_true - y_pred))
            rmse = np.sqrt(np.mean((y_true - y_pred) ** 2)) 
            mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100
            return mae, rmse, mape

        lr_metrics = calculate_metrics(y_val, lr_pred)
        rf_metrics = calculate_metrics(y_val, rf_pred) 
        lstm_metrics = calculate_metrics(y_val, lstm_pred)

        # Print results
        print("\n📈 MODEL PERFORMANCE COMPARISON:")
        print(f"Linear Regression  - MAE: {lr_metrics[0]:.3f}, RMSE: {lr_metrics[1]:.3f}, MAPE: {lr_metrics[2]:.1f}%")
        print(f"Random Forest      - MAE: {rf_metrics[0]:.3f}, RMSE: {rf_metrics[1]:.3f}, MAPE: {rf_metrics[2]:.1f}%") 
        print(f"LSTM Neural Net    - MAE: {lstm_metrics[0]:.3f}, RMSE: {lstm_metrics[1]:.3f}, MAPE: {lstm_metrics[2]:.1f}%")

        # Select best model based on MAPE
        models = [
            ("Linear Regression", lr_metrics, self.linear_model),
            ("Random Forest", rf_metrics, self.rf_model), 
            ("LSTM Neural Network", lstm_metrics, self.lstm_model)
        ]
        
        best_model = min(models, key=lambda x: x[1][2])  # Lowest MAPE
        
        print(f"\n🏆 Best Model: {best_model[0]} (MAPE: {best_model[1][2]:.1f}%)")

        # Store results
        self.results = {
            'linear_regression': {'mae': lr_metrics[0], 'rmse': lr_metrics[1], 'mape': lr_metrics[2]},
            'random_forest': {'mae': rf_metrics[0], 'rmse': rf_metrics[1], 'mape': rf_metrics[2]},
            'lstm': {'mae': lstm_metrics[0], 'rmse': lstm_metrics[1], 'mape': lstm_metrics[2]},
            'best_model': best_model[0]
        }

        self.next(self.end)

    @step
    def end(self):
        """Final step with summary and results."""

        print(
            """
╔═══════════════════════════════════════════════╗
║                                               ║  
║  🎉 TIME SERIES FORECASTING COMPLETE! 🎉     ║
║                                               ║
╚═══════════════════════════════════════════════╝

📊 FINAL RESULTS:
"""
        )

        for model_name, metrics in self.results.items():
            if isinstance(metrics, dict) and 'mae' in metrics:
                print(f"{model_name.replace('_', ' ').title():20} - MAE: {metrics['mae']:.3f}, RMSE: {metrics['rmse']:.3f}, MAPE: {metrics['mape']:.1f}%")

        print(f"\n🏆 Best Performing Model: {self.results['best_model']}")

        # Save results to JSON
        with open('timeseries_results_improved.json', 'w') as f:
            json.dump(self.results, f, indent=2)

        print(f"\n💾 Results saved to: timeseries_results_improved.json")


if __name__ == "__main__":
    TimeSeriesForecastingFlowImproved()