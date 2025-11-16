#!/usr/bin/env python3
"""
Simple Time Series Forecasting Pipeline (Metaflow Version)

This demonstrates time series forecasting using Metaflow:
- Synthetic weather data generation
- LSTM-based prediction model  
- Evaluation metrics and validation

Usage:
    python timeseries_forecasting_flow.py run --sequence_length 20 --epochs 30
"""

import os
import json
import numpy as np
from typing import Dict, List

# pylint: disable-all
from metaflow.flowspec import FlowSpec
from metaflow.decorators import step  
from metaflow.parameters import Parameter

# Import torch at module level for class definition
import torch
import torch.nn as nn

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


class TimeSeriesForecastingFlow(FlowSpec):
    """
    Simple time series forecasting pipeline using Metaflow.
    
    This demonstrates:
    - Data generation for time series
    - LSTM model implementation  
    - Training and evaluation workflow
    """
    
    # Model Configuration  
    sequence_length = Parameter('sequence_length',
                              help='Length of input sequences (default 20)',
                              default=20)
    
    epochs = Parameter('epochs',
                      help='Number of training epochs (default 30)',
                      default=30)
    
    hidden_size = Parameter('hidden_size',
                          help='LSTM hidden layer size (default 32)',
                          default=32)

    @step
    def start(self):
        """Generate synthetic time series data."""
        
        print("🚀 Starting Time Series Forecasting Pipeline")
        print(f"Sequence Length: {self.sequence_length}")
        print(f"Epochs: {self.epochs}")
        
        # Generate synthetic weather data
        np.random.seed(42)
        n_points = 500
        
        # Create time index
        dates = np.arange(n_points)
        
        # Generate temperature with seasonal patterns + noise  
        time_trend = 0.001 * dates
        seasonal_pattern = np.sin(2 * np.pi * dates / 365.25) 
        
        temperature = 20 + 10 * seasonal_pattern + time_trend + np.random.normal(0, 2, n_points)
        
        # Create sequences
        def create_sequences(data: np.ndarray, seq_length: int):
            """Create input sequences and targets."""
            X, y = [], []
            
            for i in range(len(data) - seq_length):
                # Input sequence
                X.append(data[i:(i + seq_length)])
                # Next day temperature as target  
                y.append(data[i + seq_length])
            
            return np.array(X), np.array(y)
        
        # Create sequences using converted parameters
        seq_len = int(str(self.sequence_length))
        self.X, self.y = create_sequences(temperature, seq_len)
        
        # Normalize data
        from sklearn.preprocessing import StandardScaler
        
        scaler = StandardScaler()
        X_flat = self.X.reshape(-1, 1)
        X_scaled = scaler.fit_transform(X_flat).reshape(self.X.shape)
        
        # Split data
        n_samples = len(X_scaled)
        train_size = int(0.7 * n_samples)
        
        self.X_train = X_scaled[:train_size]
        self.y_train = self.y[:train_size]
        
        self.X_test = X_scaled[train_size:]
        self.y_test = self.y[train_size:]
        
        print(f"Generated {n_samples} sequences")
        print(f"Training samples: {train_size}")
        print(f"Test samples: {n_samples - train_size}")
        
        self.next(self.build_model)

    @step
    def build_model(self):
        """Build LSTM model."""

        # Initialize model using converted parameter
        hidden = int(str(self.hidden_size))
        self.model = SimpleLSTM(hidden_size=hidden).float()

        # Loss and optimizer
        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)

        print(f"Model built with hidden size: {hidden}")
        self.next(self.train)

    @step
    def train(self):
        """Train the model."""
        
        # Convert to PyTorch tensors
        X_train_tensor = torch.FloatTensor(self.X_train).unsqueeze(-1)
        y_train_tensor = torch.FloatTensor(self.y_train).unsqueeze(-1) 
        
        batch_size = 16
        epochs = int(str(self.epochs))
        
        # Training loop  
        for epoch in range(epochs):
            total_loss = 0
            
            # Create batches
            for i in range(0, len(X_train_tensor), batch_size):
                batch_X = X_train_tensor[i:i+batch_size]
                batch_y = y_train_tensor[i:i+batch_size]
                
                # Forward pass
                outputs = self.model(batch_X)
                loss = self.criterion(outputs, batch_y)
                
                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                total_loss += loss.item()
            
            # Print progress
            if (epoch + 1) % 10 == 0:
                avg_loss = total_loss / len(X_train_tensor)
                print(f"Epoch [{epoch+1}/{epochs}] - Loss: {avg_loss:.4f}")
        
        print("Training completed!")
        self.next(self.evaluate)

    @step
    def evaluate(self):
        """Evaluate model performance."""
        
        # Make predictions
        self.model.eval()
        with torch.no_grad():
            X_test_tensor = torch.FloatTensor(self.X_test).unsqueeze(-1)
            predictions = self.model(X_test_tensor).cpu().numpy()
        
        # Calculate metrics
        from sklearn.metrics import mean_absolute_error, mean_squared_error
        
        mae = mean_absolute_error(self.y_test.flatten(), predictions.flatten())
        rmse = np.sqrt(mean_squared_error(self.y_test.flatten(), predictions.flatten()))
        
        # MAPE (avoid division by zero)
        mask = self.y_test != 0
        mape = np.mean(np.abs((self.y_test[mask] - predictions.flatten()[mask]) / self.y_test[mask])) * 100
        
        # Store results
        self.results = {
            'mae': mae,
            'rmse': rmse,  
            'mape': mape
        }
        
        print(f"\n📊 Evaluation Results:")
        print(f"   MAE: {mae:.3f}")
        print(f"   RMSE: {rmse:.3f}") 
        print(f"   MAPE: {mape:.2f}%")
        
        self.next(self.end)

    @step
    def end(self):
        """Complete the forecasting workflow."""
        
        print("\n" + "="*60)
        print("🎉 TIME SERIES FORECASTING COMPLETE")
        print("="*60)
        
        # Print results
        if hasattr(self, 'results'):
            print(f"📈 Forecast Performance:")
            for metric, value in self.results.items():
                if metric == 'mape':
                    print(f"   {metric.upper()}: {value:.2f}%")
                else:
                    print(f"   {metric.upper()}: {value:.3f}")
        
        # Model info
        total_params = sum(p.numel() for p in self.model.parameters())
        print(f"\n🔧 Model Information:")
        print(f"   Total Parameters: {total_params:,}")
        
        # Configuration summary
        print(f"\n⚙️  Configuration:")
        print(f"   Sequence Length: {self.sequence_length}")
        print(f"   Epochs: {self.epochs}") 
        print(f"   Hidden Size: {self.hidden_size}")
        
        # Save results
        with open('timeseries_results.json', 'w') as f:
            json.dump({
                'configuration': {
                    'sequence_length': str(self.sequence_length),
                    'epochs': str(self.epochs),
                    'hidden_size': str(self.hidden_size)
                },
                'metrics': getattr(self, 'results', {}),
                'model_info': {
                    'total_parameters': total_params
                }
            }, f, indent=2)
        
        print(f"\n💾 Results saved to: timeseries_results.json")
        print("\n✅ Time series forecasting workflow completed!")


if __name__ == "__main__":
    # Add missing imports at the module level
    import numpy as np
    
    TimeSeriesForecastingFlow()