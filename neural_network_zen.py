# Neural Network pipeline using ZenML
"""
This file implements the same functionality as ``neural_network_flow.py``
but uses ZenML instead of Metaflow.
It loads the MNIST dataset, builds a simple CNN model with PyTorch,
trains it for a number of epochs and reports test accuracy.
"""

from typing import Tuple
# numpy is not needed for this pipeline; removed import

# ZenML imports
from zenml import pipeline, step

@step
def start() -> Tuple[object, object]:
    """Load MNIST dataset and return train and test DataLoaders."""
    import torch
    import torchvision
    import torchvision.transforms as transforms

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 128
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    train_dataset = torchvision.datasets.MNIST(root="./data", train=True, download=True, transform=transform)
    test_dataset = torchvision.datasets.MNIST(root="./data", train=False, download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    # Return loaders as generic objects (ZenML will treat them as artifacts)
    return train_loader, test_loader

# Define the CNN architecture at module level so ZenML can resolve it.
import torch
import torch.nn as nn

class SimpleCNN(nn.Module):
    """A simple convolutional neural network for MNIST classification."""
    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2)
        self.flatten = nn.Flatten()
        # After two conv/pool layers the feature map size is 5x5
        self.fc = nn.Linear(64 * 5 * 5, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network."""
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = self.flatten(x)
        return self.fc(x)

@step
def build_model() -> object:
    """Instantiate the CNN model and return it.
    The loss function and optimizer will be created inside the training step
    to avoid pickling issues across ZenML steps."""
    # No need for additional imports here; SimpleCNN is defined at module level.
    model = SimpleCNN()
    return model

@step
def train(
    model: object,
    train_loader: object,
    test_loader: object,
) -> float:
    """Train the CNN for a fixed number of epochs and return test accuracy.

    The loss function (CrossEntropyLoss) and optimizer (Adam) are instantiated
    inside this step to avoid cross‑step pickling issues.
    """
    import torch
    import torch.nn as nn
    import torch.optim as optim

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Define loss function and optimizer locally
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters())

    epochs = 10  # default number of epochs (can be parameterized later)
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, target)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        avg_loss = running_loss / len(train_loader)
        print(f"Epoch [{epoch + 1}/{epochs}] - Average loss: {avg_loss:.4f}")

    # Evaluation on test set
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            outputs = model(data)
            _, predicted = torch.max(outputs.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    accuracy = 100 * correct / total if total > 0 else 0.0
    return float(accuracy)

@step
def end(test_accuracy: float) -> None:
    """Final step – report test accuracy."""
    print(f"Test Accuracy: {test_accuracy:.2f}%")
    print("NeuralNetworkPipeline is all done.")

@pipeline
def neural_network_pipeline():
    train_loader, test_loader = start()
    model = build_model()
    accuracy = train(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
    )
    end(test_accuracy=accuracy)

if __name__ == "__main__":
    # Running the pipeline locally via ZenML's default orchestrator.
    neural_network_pipeline()
