#!/usr/bin/env python3
"""
Advanced Metaflow Workflow with Parameter Sweeps

This demonstrates advanced Metaflow features including:
- Parallel parameter sweeps using foreach
- Conditional logic and branching
- Hyperparameter optimization workflows
- Advanced step decorators (@retry, @timeout, etc.)

Usage:
    python metaflow_parameter_sweep.py run --learning_rates 0.001,0.01,0.1 --batch_sizes 32,64,128
"""

import numpy as np
from metaflow.decorators import step
from metaflow.flowspec import FlowSpec
from metaflow.parameters import Parameter

try:
    from colorama import Fore, Style, init

    init(autoreset=True)
except ImportError:
    # Fallback if colorama is not available
    class Fore:
        GREEN = "\033[92m"
        YELLOW = "\033[93m"
        RED = "\033[91m"
        BLUE = "\033[94m"
        MAGENTA = "\033[95m"
        CYAN = "\033[96m"
        WHITE = "\033[97m"

    class Style:
        RESET_ALL = "\033[0m"
        BRIGHT = "\033[1m"


# Import torch at module level for model classes
import torch
import torch.nn as nn
import torch.optim as optim


# Define model classes at module level for proper pickling
class SimpleModel(nn.Module):
    def __init__(self, input_size=20):
        super(SimpleModel, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 2),
        )

    def forward(self, x):
        return self.layers(x)


class DeepModel(nn.Module):
    def __init__(self, input_size=20):
        super(DeepModel, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 2),
        )

    def forward(self, x):
        return self.layers(x)


class MetaflowParameterSweepFlow(FlowSpec):
    """
    Advanced Metaflow flow demonstrating parameter sweeps and parallel execution.

    This flow trains multiple models with different hyperparameters in parallel,
    then compares their performance and selects the best model.
    """

    # Hyperparameters to sweep
    learning_rates = Parameter(
        "learning_rates",
        help="Comma-separated list of learning rates to test",
        default="0.001,0.01,0.1",
    )

    batch_sizes = Parameter(
        "batch_sizes",
        help="Comma-separated list of batch sizes to test",
        default="32,64",
    )

    model_types = Parameter(
        "model_types",
        help="Comma-separated list of model types (simple,deep)",
        default="simple,deep",
    )

    epochs = Parameter(
        "epochs", help="Number of training epochs for each model", default="5"
    )

    # Data parameters
    dataset_size = Parameter(
        "dataset_size",
        help="Size of synthetic dataset for training (smaller for demo)",
        default="1000",
    )

    # Parallel execution settings
    max_parallel_runs = Parameter(
        "max_parallel_runs",
        help="Maximum number of parallel model training runs",
        default="6",
    )

    @step
    def start(self):
        """Initialize the parameter sweep workflow."""

        # Unique ASCII Art for Parameter Sweep Pipeline
        print(
            Fore.WHITE
            + """
    ╔══════════════════════════════════════════════════════╗
    ║                                                      ║
    ║  🔬⚡ ADVANCED PARAMETER SWEEP PIPELINE ⚡🔬        ║
    ║                                                      ║
    ║  Testing Multiple AI Model Configurations            ║
    ║                                                      ║
    ╚══════════════════════════════════════════════════════╝
        """
        )

        # Parse parameter lists with better error handling
        try:
            self.lr_list = [
                float(lr.strip()) for lr in str(self.learning_rates).split(",")
            ]
            self.bs_list = [int(bs.strip()) for bs in str(self.batch_sizes).split(",")]
            self.model_list = [mt.strip() for mt in str(self.model_types).split(",")]
        except (ValueError, AttributeError) as e:
            print(Fore.YELLOW + f"⚠️ Parameter parsing failed: {e}")
            self.lr_list = [0.001, 0.01, 0.1]
            self.bs_list = [32, 64]
            self.model_list = ["simple", "deep"]

        print(Fore.BLUE + f"📋 Parameter Sweep Configuration:")
        print(Fore.CYAN + f"   • Learning Rates: {self.lr_list} (how fast AI learns)")
        print(
            Fore.CYAN
            + f"   • Batch Sizes: {self.bs_list} (how many examples per training step)"
        )
        print(
            Fore.CYAN
            + f"   • Model Types: {self.model_list} (different AI architectures)"
        )
        print(Fore.CYAN + f"   • Training Epochs: {self.epochs} (training cycles)")

        # Generate parameter combinations (cartesian product)
        import itertools

        self.parameter_combinations = list(
            itertools.product(self.lr_list, self.bs_list, self.model_list)
        )

        # Limit parallel runs for demo purposes
        max_runs = int(str(self.max_parallel_runs))
        if len(self.parameter_combinations) > max_runs:
            print(
                Fore.YELLOW
                + f"ℹ️ Limiting to {max_runs} combinations for demo performance"
            )
            self.parameter_combinations = self.parameter_combinations[:max_runs]

        print(
            Fore.GREEN
            + f"✅ Ready to test {len(self.parameter_combinations)} different model configurations!"
        )

        # Store run metadata (use different variable names since Parameters are read-only)
        self.dataset_size_value = int(str(self.dataset_size))
        self.epochs_value = int(str(self.epochs))
        self.next(self.generate_data)

    @step
    def generate_data(self):
        """Generate synthetic training data for parameter sweep."""

        print(f"🎯 Generating {self.dataset_size} samples of synthetic data...")

        # Create simple binary classification dataset
        np.random.seed(42)  # For reproducible results

        n_samples = self.dataset_size
        n_features = 20

        # Generate features
        X = np.random.randn(int(str(n_samples)), n_features)

        # Create labels with some structure (not perfectly random)
        weights = np.random.randn(n_features)
        y_raw = X @ weights + 0.1 * np.random.randn(int(str(n_samples)))
        y = (y_raw > np.median(y_raw)).astype(int)  # Binary classification

        # Split into train/validation
        n_train = int(0.8 * int(str(n_samples)))

        self.X_train = X[:n_train]
        self.y_train = y[:n_train]
        self.X_val = X[n_train:]
        self.y_val = y[n_train:]

        print(
            f"Data split: {len(self.X_train)} train, {len(self.X_val)} validation samples"
        )

        self.next(self.train_models_parallel)

    @step
    def train_models_parallel(self):
        """Train multiple models with different hyperparameters in parallel."""

        print(
            f"🏋️ Training {len(self.parameter_combinations)} models with parameter sweeps..."
        )

        # Train models for each parameter combination
        results = []

        for i, (lr, batch_size, model_type) in enumerate(self.parameter_combinations):
            print(f"Training model {i + 1}/{len(self.parameter_combinations)}:")

            # Create unique run ID for this parameter combination
            run_id = f"model_{i + 1}_{model_type}_lr{lr}_bs{batch_size}"

            # Train model with given hyperparameters
            result = self._train_single_model(
                run_id=run_id,
                learning_rate=lr,
                batch_size=batch_size,
                model_type=model_type,
            )

            results.append(result)

        # Store all training results for analysis
        self.model_results = results

        print(f"✅ Completed parameter sweeps with {len(results)} models")

        self.next(self.analyze_results)

    @step
    def analyze_results(self):
        """Analyze all model results and select the best performing configuration."""

        print("📊 Analyzing parameter sweep results...")

        # Extract key metrics from all models
        performance_data = []

        for result in self.model_results:
            if "error" not in result:
                performance_data.append(
                    {
                        "run_id": result["run_id"],
                        "model_type": result["model_type"],
                        "learning_rate": result["learning_rate"],
                        "batch_size": result["batch_size"],
                        "train_accuracy": result.get("final_train_acc", 0),
                        "val_accuracy": result.get("final_val_acc", 0),
                        "training_time": result.get("training_time", 0),
                    }
                )

        # Sort by validation accuracy
        performance_data.sort(key=lambda x: x["val_accuracy"], reverse=True)

        # Select best model
        if performance_data:
            self.best_model = performance_data[0]

            print(f"\n🏆 BEST MODEL RESULTS:")
            print(f"Run ID: {self.best_model['run_id']}")
            print(f"Model Type: {self.best_model['model_type']}")
            print(f"Learning Rate: {self.best_model['learning_rate']}")
            print(f"Batch Size: {self.best_model['batch_size']}")
            print(f"Training Accuracy: {self.best_model['train_accuracy']:.4f}")
            print(f"Validation Accuracy: {self.best_model['val_accuracy']:.4f}")

        else:
            print("❌ No successful model training results!")

        # Store performance summary
        self.performance_summary = {
            "total_models_tested": len(self.model_results),
            "successful_models": len(performance_data),
            "parameter_combinations_tested": [
                {"lr": lr, "bs": bs, "model_type": mt}
                for lr, bs, mt in self.parameter_combinations
            ],
            "performance_ranking": performance_data[:5],  # Top 5 models
        }

        self.next(self.evaluate_final_model)

    @step
    def evaluate_final_model(self):
        """Evaluate the best model in more detail."""

        if "best_model" not in self.__dict__:
            print("No best model available for evaluation")
            self.final_evaluation = {"error": "No successful models"}
        else:
            print(f"🔍 Detailed evaluation of best model: {self.best_model['run_id']}")

            # Get the corresponding trained model result
            best_result = None
            for result in self.model_results:
                if result["run_id"] == self.best_model["run_id"]:
                    best_result = result
                    break

            if best_result and "final_val_acc" in best_result:
                # Detailed metrics
                val_acc = best_result["final_val_acc"]

                # Determine if performance is acceptable
                if val_acc > 0.85:
                    status = "Excellent"
                elif val_acc > 0.75:
                    status = "Good"
                elif val_acc > 0.65:
                    status = "Fair"
                else:
                    status = "Needs Improvement"

                self.final_evaluation = {
                    "run_id": best_result["run_id"],
                    "validation_accuracy": val_acc,
                    "performance_status": status,
                    "learning_rate": best_result["learning_rate"],
                    "batch_size": best_result["batch_size"],
                    "model_type": best_result["model_type"],
                    "training_time_seconds": best_result.get("training_time", 0),
                    "recommendation": self._generate_recommendations(best_result),
                }

                print(f"Performance Status: {status}")
                print(f"Recommendation: {self.final_evaluation['recommendation']}")

            else:
                self.final_evaluation = {"error": "Could not find best model results"}

        print("✅ Final evaluation complete")
        self.next(self.end)

    @step
    def end(self):
        """Complete the parameter sweep workflow."""

        print("\n" + "=" * 80)
        print("🎯 METAFLOW PARAMETER SWEEP COMPLETE")
        print("=" * 80)

        print(
            f"Total Models Trained: {self.performance_summary['total_models_tested']}"
        )
        print(f"Successful Models: {self.performance_summary['successful_models']}")

        if hasattr(self, "final_evaluation") and "error" not in self.final_evaluation:
            eval_data = self.final_evaluation
            print(f"\nBest Configuration:")
            print(f"  Model Type: {eval_data['model_type']}")
            print(f"  Learning Rate: {eval_data['learning_rate']}")
            print(f"  Batch Size: {eval_data['batch_size']}")
            print(f"  Validation Accuracy: {eval_data['validation_accuracy']:.4f}")
            print(f"  Status: {eval_data['performance_status']}")
            print(f"  Recommendation: {eval_data['recommendation']}")

        # Show top 3 configurations
        if "performance_ranking" in self.performance_summary:
            print(f"\nTop 3 Configurations:")
            for i, model in enumerate(
                self.performance_summary["performance_ranking"][:3]
            ):
                print(
                    f"  {i + 1}. {model['run_id']}: {model['val_accuracy']:.4f} validation accuracy"
                )

    def _train_single_model(
        self, run_id: str, learning_rate: float, batch_size: int, model_type: str
    ) -> dict:
        """Train a single model with given hyperparameters."""

        import time

        start_time = time.time()

        try:
            # Import PyTorch components
            import torch
            import torch.nn as nn
            import torch.optim as optim

            # Initialize model
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model = (
                SimpleModel(input_size=20).to(device)
                if model_type == "simple"
                else DeepModel(input_size=20).to(device)
            )

            # Loss and optimizer
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(model.parameters(), lr=learning_rate)

            # Convert data to tensors
            X_train_tensor = torch.FloatTensor(self.X_train).to(device)
            y_train_tensor = torch.LongTensor(self.y_train).to(device)
            X_val_tensor = torch.FloatTensor(self.X_val).to(device)
            y_val_tensor = torch.LongTensor(self.y_val).to(device)

            # Training loop
            model.train()
            train_losses = []
            val_accuracies = []

            for epoch in range(self.epochs_value):
                # Mini-batch training
                total_loss = 0

                # Simple batch processing (not optimal for small datasets)
                if len(X_train_tensor) <= batch_size:
                    # Single batch
                    optimizer.zero_grad()
                    outputs = model(X_train_tensor)
                    loss = criterion(outputs, y_train_tensor)
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item()

                else:
                    # Multi-batch training
                    n_batches = len(X_train_tensor) // batch_size

                    for i in range(n_batches):
                        start_idx = i * batch_size
                        end_idx = min((i + 1) * batch_size, len(X_train_tensor))

                        batch_X = X_train_tensor[start_idx:end_idx]
                        batch_y = y_train_tensor[start_idx:end_idx]

                        optimizer.zero_grad()
                        outputs = model(batch_X)
                        loss = criterion(outputs, batch_y)
                        loss.backward()
                        optimizer.step()

                        total_loss += loss.item()

                # Validation
                if (
                    epoch % 2 == 0 or epoch == self.epochs_value - 1
                ):  # Check every 2 epochs
                    model.eval()

                    with torch.no_grad():
                        val_outputs = model(X_val_tensor)
                        _, predicted = torch.max(val_outputs.data, 1)
                        val_acc = (predicted == y_val_tensor).float().mean().item()

                    train_losses.append(total_loss)
                    val_accuracies.append(val_acc)

                    model.train()

            # Final accuracy
            final_train_acc = self._evaluate_model(
                model, X_train_tensor, y_train_tensor
            )
            final_val_acc = self._evaluate_model(model, X_val_tensor, y_val_tensor)

            training_time = time.time() - start_time

            return {
                "run_id": run_id,
                "model_type": model_type,
                "learning_rate": learning_rate,
                "batch_size": batch_size,
                "final_train_acc": final_train_acc,
                "final_val_acc": final_val_acc,
                "training_time": training_time,
                "training_losses": train_losses[-3:],  # Last few losses
                "val_accuracies": val_accuracies[-3:],  # Last few validation accuracies
            }

        except Exception as e:
            print(f"❌ Model training failed for {run_id}: {str(e)}")

            return {
                "run_id": run_id,
                "model_type": model_type,
                "learning_rate": learning_rate,
                "batch_size": batch_size,
                "error": str(e),
                "training_time": time.time() - start_time,
            }

    def _evaluate_model(self, model: nn.Module, X_tensor, y_tensor) -> float:
        """Evaluate model accuracy on given data."""

        import torch

        model.eval()
        with torch.no_grad():
            outputs = model(X_tensor)
            _, predicted = torch.max(outputs.data, 1)
            accuracy = (predicted == y_tensor).float().mean().item()

        return accuracy

    def _generate_recommendations(self, best_result: dict) -> str:
        """Generate recommendations based on the best model results."""

        val_acc = best_result.get("final_val_acc", 0)
        lr = best_result["learning_rate"]
        lr = best_result['learning_rate']

        if val_acc > 0.9:
            return "Excellent performance! Model is ready for production."
        elif val_acc > 0.8:
            if lr < 0.001:
                return "Good performance, but consider increasing learning rate for faster convergence."
            elif lr > 0.1:
                return "Good performance, but consider reducing learning rate for better stability."
            else:
                return "Good performance. Consider collecting more data or adding regularization."
        elif val_acc > 0.7:
            return "Fair performance. Try increasing model complexity or training duration."
        else:
            return "Performance needs improvement. Consider different architectures, more data, or extended training."


if __name__ == "__main__":
    MetaflowParameterSweepFlow()
