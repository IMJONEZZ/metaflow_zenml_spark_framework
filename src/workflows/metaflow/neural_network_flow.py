from metaflow import FlowSpec, Parameter, step

# Import smart device manager for AMD GPU compatibility
import sys
sys.path.append('/home/imjonezz/Desktop/metaflow_zenml_spark_framework/src/utils')
from gpu_device_manager import get_device_with_fallback

# Import smart device manager for AMD GPU compatibility
import sys
sys.path.append('/home/imjonezz/Desktop/metaflow_zenml_spark_framework/src/utils')
from gpu_device_manager import get_device_with_fallback

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


class NeuralNetFlow(FlowSpec):
    """A Metaflow flow that trains a simple CNN on MNIST using PyTorch.

    Enhanced with AMD GPU compatibility - automatically routes CNN models to CPU
    to avoid memory faults while maintaining full functionality.
    
    The flow consists of four steps:
        1. ``start`` – load and preprocess the MNIST dataset with smart device detection.
    Enhanced with AMD GPU compatibility - automatically routes CNN models to CPU
    to avoid memory faults while maintaining full functionality.
    
    The flow consists of four steps:
        1. ``start`` – load and preprocess the MNIST dataset with smart device detection.
        2. ``build_model`` – define a small convolutional neural network,
           loss function, optimizer and move everything to the appropriate device.
        3. ``train`` – train the model for the specified number of epochs.
        4. ``end`` – final step indicating completion with performance report.
    """

    # Number of training epochs (default 10)
    epochs = Parameter("e", default=10)
    
    # GPU selection strategy (auto/force_cpu/force_gpu/safe_only)
    gpu_strategy = Parameter("gpu", default="auto")
    
    # Batch size parameter - reduced for better memory management
    batch_size = Parameter("batch", default=64)
    
    # GPU selection strategy (auto/force_cpu/force_gpu/safe_only)
    gpu_strategy = Parameter("gpu", default="auto")
    
    # Batch size parameter - reduced for better memory management
    batch_size = Parameter("batch", default=64)

    @step
    def start(self):
        """Load MNIST dataset using torchvision and create DataLoaders."""

        # Unique ASCII Art for Simple CNN Training
        print(
            Fore.WHITE
            + """
            Image(28×28)
                ↓
            [Flatten: 784 pixels]
                ↓
            [Hidden Layer: 784→128] ──→ ReLU
                ↓
            [Hidden Layer: 128→64]  ──→ ReLU
                ↓
            [Output Layer: 64→10]   ──→ Softmax
                ↓
            [0.1, 0.05, 0.7, ...] ──→ argmax ──→ Prediction: "2"
                                                      ↓
                                                Compare to Label
                                                      ↓
                                                  Calculate Loss
                                                      ↓
                                                Backpropagate
        """
        )

        import torch
        import torchvision
        import torchvision.transforms as transforms

        # Smart device selection with AMD GPU compatibility
        print(Fore.BLUE + "🔍 Analyzing hardware configuration...")
        
        # Create a temporary model to analyze device compatibility
        temp_model = torch.nn.Sequential(
            torch.nn.Conv2d(1, 32, kernel_size=3),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),
        )
        
        # Use our smart device manager
        self.device, device_info = get_device_with_fallback(
            model=temp_model,
            force_device=self.gpu_strategy if hasattr(self, 'gpu_strategy') else "auto",
            batch_size=int(getattr(self, 'batch_size', 64))
        )
        
        # Enhanced device reporting
        if hasattr(device_info, 'compatibility_analysis'):
            analysis = device_info['compatibility_analysis']
            if not analysis['is_compatible']:
                print(Fore.YELLOW + 
                    f"🛡️  CNN model detected as high-risk - CPU fallback will be used")
                print(Fore.BLUE + 
                    f"   This prevents GPU memory faults on AMD hardware")
            else:
                print(Fore.GREEN + 
                    f"✅ Model is GPU compatible - using optimal acceleration")

        self.num_classes = 10
        # Use batch size from parameter or default to smaller value for memory safety
        self.batch_size = int(getattr(self, 'batch_size', 64))
        
        # Normalization values for MNIST (mean=0.1307, std=0.3081) – standard practice
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,)),
            ]
        )

        print(Fore.BLUE + "📁 Loading MNIST handwritten digit dataset...")

        train_dataset = torchvision.datasets.MNIST(
            root="./data", train=True, download=True, transform=transform
        )
        test_dataset = torchvision.datasets.MNIST(
            root="./data", train=False, download=True, transform=transform
        )

        # Reduced num_workers for stability on AMD hardware  
        # Reduced num_workers for stability on AMD hardware  
        self.train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=0
        )
        self.test_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=0
        )

        print(
            Fore.GREEN
            + f"✅ Dataset ready - {len(train_dataset)} training, {len(test_dataset)} test images"
        )

        # Proceed to the next step
        self.next(self.build_model)

    @step
    def build_model(self):
        """Define a simple CNN, loss function and optimizer."""
        
        # Re-analyze device for final model construction
        
        # Re-analyze device for final model construction
        import torch.nn as nn
        import torch.optim as optim

        # Convolutional network architecture – mirrors the original Keras model
        self.model = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Flatten(),
            nn.Dropout(p=0.5),
            # After two conv/pool layers the feature map size is 5x5 (see calculations below)
            nn.Linear(in_features=64 * 5 * 5, out_features=self.num_classes),
        )

        # Move model to the selected device (CPU or GPU)
        self.model = self.model.to(self.device)

        # Loss and optimizer – improved training configuration
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters())

        # Report model details
        total_params = sum(p.numel() for p in self.model.parameters())
        print(Fore.GREEN + 
            f"🧠 Model built with {total_params:,} parameters on {str(self.device)}")

        self.next(self.train)

    @step
    def train(self):
        """Train the model for ``self.epochs`` epochs using the training DataLoader."""

        print(
            Fore.WHITE
            + """
    ╔════════════════════════════════════════╗
    ║                                        ║
    ║  🎯 TRAINING CNN ON HANDWRITTEN        ║
    ║     DIGITS (0-9) 📸                    ║
    ║                                        ║
    ╚════════════════════════════════════════╝
        """
        )

        import torch

        # Training loop with progress updates every 25%
        epoch_interval = max(1, int(self.epochs) // 4)

        for epoch in range(int(self.epochs)):
            self.model.train()  # set model to training mode (enables dropout, etc.)
            running_loss = 0.0
            for batch_idx, (data, target) in enumerate(self.train_loader):
                data, target = data.to(self.device), target.to(self.device)

                self.optimizer.zero_grad()
                outputs = self.model(data)
                loss = self.criterion(outputs, target)
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item()

            # Progress update every 25% and at completion
            if (epoch + 1) % epoch_interval == 0 or epoch == int(self.epochs) - 1:
                avg_loss = running_loss / len(self.train_loader)
                progress_pct = ((epoch + 1) / int(self.epochs)) * 100
                print(
                    Fore.CYAN
                    + f"   Epoch [{epoch + 1}/{int(self.epochs)}] - Loss: {avg_loss:.4f} ({progress_pct:.0f}% complete)"
                )

        final_accuracy = self._evaluate_model()

        success_msg = (
            "🎉 Excellent"
            if final_accuracy > 95
            else "✅ Good"
            if final_accuracy > 90
            else "⚠️ Fair"
        )

        print(
            Fore.GREEN
            + f"{success_msg} training complete! Final accuracy: {final_accuracy:.2f}%"
        )

        self.next(self.end)

    def _evaluate_model(self) -> float:
        """Evaluate the trained model on test data and return accuracy percentage."""
        import torch

        self.model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for data, target in self.test_loader:
                data, target = data.to(self.device), target.to(self.device)
                outputs = self.model(data)
                _, predicted = torch.max(outputs.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()

        accuracy = 100 * correct / total if total > 0 else 0
        return accuracy

    @step
    def end(self):
        """Final step – report completion and optionally evaluate on test set."""
        import torch

        # Simple evaluation on the test set (optional, does not affect flow state)
        self.model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for data, target in self.test_loader:
                data, target = data.to(self.device), target.to(self.device)
                outputs = self.model(data)
                _, predicted = torch.max(outputs.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()
        accuracy = 100 * correct / total if total > 0 else 0
        print(f"Test Accuracy: {accuracy:.2f}%")
        print("NeuralNetFlow is all done.")


if __name__ == "__main__":
    NeuralNetFlow()
