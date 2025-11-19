#!/usr/bin/env python3
"""
Simple Time Series Forecasting Pipeline (ZenML Version)

This demonstrates time series forecasting using ZenML:
- Synthetic data generation
- Simple prediction model
- Evaluation metrics

Usage:
    python timeseries_forecasting_zen_simple.py
"""

import os
from typing import Annotated, Any, Dict, List

import numpy as np
# ZenML imports
from zenml import pipeline, step


@step(enable_cache=True)
def generate_time_series(num_points: int = 200) -> Annotated[List[float], "time_series_data"]:
    """Generate simple time series data."""
    print("""
        past    now    future
         ·       │      ?
        · ·      │     ? ?
       ·   ·     │    ?   ?
      ·     ·    │   ?     ?
     ─────────────┼──────────→
                 │
      we ride    │   into mist
      the wave   │
      """)

    import numpy as np

    print("🚀 Generating Time Series Data with ZenML")

    # Generate synthetic temperature data
    np.random.seed(42)

    # Time index
    time = np.arange(num_points)

    # Temperature with trend and seasonality
    temperature = (
        20
        + 0.1 * time
        + 5 * np.sin(2 * np.pi * time / 50)
        + np.random.normal(0, 1, num_points)
    )

    print(f"Generated {num_points} temperature data points")
    return temperature.tolist()


@step(enable_cache=True)
def create_sequences(data: List[float], sequence_length: int = 10) -> Annotated[Dict[str, Any], "sequences"]:
    """Create input sequences and targets for prediction."""

    print(f"Creating sequences with length {sequence_length}")

    X, y = [], []

    for i in range(len(data) - sequence_length):
        # Input sequence
        X.append(data[i : (i + sequence_length)])
        # Next value as target
        y.append(data[i + sequence_length])

    print(f"Created {len(X)} sequences")

    return {"features": X, "targets": y, "sequence_length": sequence_length}


@step(enable_cache=True)
def build_simple_model() -> Annotated[Any, "model"]:
    """Build a simple predictive model."""

    try:
        from sklearn.linear_model import LinearRegression

        print("✅ Using Linear Regression model")
        return LinearRegression()
    except ImportError:
        # Simple moving average fallback
        print("⚠️ sklearn not available, using moving average")

        class MovingAverage:
            def __init__(self):
                self.window = 5

            def fit(self, X, y=None):
                return self

            def predict(self, X):
                # Simple moving average prediction
                predictions = []
                for seq in X:
                    if len(seq) >= self.window:
                        pred = sum(seq[-self.window :]) / self.window
                    else:
                        pred = seq[-1] if seq else 0
                    predictions.append(pred)
                return (
                    np.array(predictions) if len(predictions) > 1 else [predictions[0]]
                )

        return MovingAverage()


@step(enable_cache=False)
def train_model(features_data: Dict[str, Any], model: Any) -> Annotated[Any, "trained_model"]:
    """Train the time series forecasting model."""

    import numpy as np

    try:
        X = features_data["features"]
        y = features_data["targets"]

        # Convert to appropriate format
        if hasattr(X[0], "__iter__") and not isinstance(X[0], str):
            X_2d = [np.array(seq).reshape(-1) for seq in X]
        else:
            X_2d = [[x] for x in X]

        # Train model
        if hasattr(model, "fit"):
            if len(X_2d) > 0 and len(y) > 0:
                model.fit(X_2d, y)
            else:
                print("Warning: Empty training data")
        else:
            print("Warning: Model doesn't have fit method")

        print(f"✅ Model trained on {len(X_2d)} samples")

    except Exception as e:
        print(f"❌ Training error: {e}")

    return model


@step(enable_cache=False)
def evaluate_model(features_data: Dict[str, Any], trained_model: Any) -> Annotated[Dict[str, Any], "evaluation_results"]:
    """Evaluate the time series forecasting model."""

    import numpy as np

    try:
        X = features_data["features"]
        y = features_data["targets"]

        # Make predictions
        if hasattr(X[0], "__iter__") and not isinstance(X[0], str):
            X_2d = [np.array(seq).reshape(-1) for seq in X]
        else:
            X_2d = [[x] for x in X]

        if hasattr(trained_model, "predict"):
            predictions = trained_model.predict(X_2d)
        else:
            # Fallback: use last value
            predictions = [seq[-1] if seq else 0 for seq in X]

        # Calculate metrics
        y_array = np.array(y) if hasattr(y, "__iter__") else [y]

        # Handle different prediction formats
        if hasattr(predictions, "__iter__") and not isinstance(predictions, str):
            pred_array = np.array(predictions)
        else:
            pred_array = np.array([predictions])

        # Ensure same length
        min_len = min(len(y_array), len(pred_array))
        y_true = y_array[:min_len]
        y_pred = pred_array[:min_len]

        # Calculate metrics
        mae = np.mean(np.abs(y_true - y_pred))
        mse = np.mean((y_true - y_pred) ** 2)
        rmse = np.sqrt(mse)

        # MAPE (avoid division by zero)
        mask = y_true != 0
        if mask.any():
            mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
        else:
            mape = float("inf")

        results = {
            "mae": mae,
            "rmse": rmse,
            "mape": mape if mape != float("inf") else 0,
        }

        print(f"\n📊 Evaluation Results:")
        print(f"   MAE: {mae:.3f}")
        print(f"   RMSE: {rmse:.3f}")
        print(f"   MAPE: {mape:.2f}%")

        return results

    except Exception as e:
        print(f"❌ Evaluation error: {e}")
        return {"error": str(e)}


@pipeline
def simple_timeseries_pipeline(
    num_points: int = 200, sequence_length: int = 10
) -> Dict[str, Any]:
    """Complete time series forecasting pipeline."""

    # Generate data
    temperature_data = generate_time_series(num_points)
    features_data = create_sequences(temperature_data, sequence_length)

    # Build and train model
    model = build_simple_model()
    trained_model = train_model(features_data, model)

    # Evaluate
    results = evaluate_model(features_data, trained_model)

    return results


if __name__ == "__main__":
    print("🎉 Running Simple ZenML Time Series Forecasting Pipeline")

    try:
        # Run the pipeline
        results = simple_timeseries_pipeline(num_points=200, sequence_length=10)

        print("\n" + "=" * 60)
        print("🎉 SIMPLE TIME SERIES FORECASTING COMPLETE")
        print("=" * 60)

        # For ZenML, results might be a PipelineRunResponse object
        print("✅ Time series forecasting pipeline completed successfully!")

    except Exception as e:
        print(f"❌ Pipeline failed: {e}")

    print("\n✅ ZenML pipeline execution finished!")
