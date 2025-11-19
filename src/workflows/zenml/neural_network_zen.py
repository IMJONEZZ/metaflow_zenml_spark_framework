# Neural Network pipeline using ZenML
"""
This file implements the same functionality as ``neural_network_flow.py``
but uses ZenML instead of Metaflow.
It loads the MNIST dataset, builds a simple CNN model with PyTorch,
trains it for a number of epochs and reports test accuracy.
"""

import os
from typing import Annotated, Tuple, Type, Any

import torch
import torch.nn as nn
import torch.utils.data

# numpy is not needed for this pipeline; removed import
# ZenML imports
from zenml import get_step_context, log_metadata, pipeline, step
from zenml.artifacts.artifact_config import ArtifactConfig
from zenml.enums import ArtifactType
from zenml.io import fileio
from zenml.materializers.base_materializer import BaseMaterializer


@step
def start() -> Tuple[Annotated[torch.utils.data.DataLoader, "train_loader"], Annotated[torch.utils.data.DataLoader, "test_loader"]]:
    """Load MNIST dataset and return train and test DataLoaders."""
    print("""
        ████  ░░░░  ████
        ░░██  ████  ██░░
        ████  ░░██  ████
        ██░░  ████  ░░██
        ████  ░░░░  ████

         0     1     2

        simple lines
        teach machines
        to see
        """)
    import torch
    import torchvision
    import torchvision.transforms as transforms

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 128
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ]
    )
    train_dataset = torchvision.datasets.MNIST(
        root="./data", train=True, download=True, transform=transform
    )
    test_dataset = torchvision.datasets.MNIST(
        root="./data", train=False, download=True, transform=transform
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=2
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=2
    )
    # Return loaders as generic objects (ZenML will treat them as artifacts)
    return train_loader, test_loader


# Define the CNN architecture at module level so ZenML can resolve it.


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


class SimpleCNNMaterializer(BaseMaterializer):
    """Custom materializer for SimpleCNN models.
    
    This materializer saves the model's state dict and recreates the model
    when loading, avoiding class resolution issues with __main__ module.
    """
    
    ASSOCIATED_TYPES = (SimpleCNN, nn.Module)
    ASSOCIATED_ARTIFACT_TYPE = ArtifactType.MODEL
    
    def load(self, data_type: Type[Any]) -> SimpleCNN:
        """Load SimpleCNN from storage."""
        # Create a new instance of the model
        model = SimpleCNN()
        
        # Load the state dict
        state_dict_path = os.path.join(self.uri, "model_state_dict.pt")
        with fileio.open(state_dict_path, "rb") as f:
            state_dict = torch.load(f, map_location="cpu")
        
        # Load the state dict into the model
        model.load_state_dict(state_dict)
        model.eval()
        
        return model
    
    def save(self, data: SimpleCNN) -> None:
        """Save SimpleCNN to storage."""
        # Save the model's state dict
        state_dict_path = os.path.join(self.uri, "model_state_dict.pt")
        with fileio.open(state_dict_path, "wb") as f:
            torch.save(data.state_dict(), f)


@step
def build_model() -> Annotated[nn.Module, "model"]:
    """Instantiate the CNN model and return it.
    The loss function and optimizer will be created inside the training step
    to avoid pickling issues across ZenML steps."""
    # No need for additional imports here; SimpleCNN is defined at module level.
    model = SimpleCNN()
    return model


@step(output_materializers={"trained_model": SimpleCNNMaterializer})
def train(
    model: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    test_loader: torch.utils.data.DataLoader,
) -> Tuple[Annotated[nn.Module, ArtifactConfig(name="trained_model", is_model=True)], Annotated[float, "test_accuracy"]]:
    """Train the CNN for a fixed number of epochs, persist the model, and return test accuracy.

    The loss function (CrossEntropyLoss) and optimizer (Adam) are instantiated
    inside this step to avoid cross‑step pickling issues.
    All metadata is logged to the model using log_metadata with infer_model=True.
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
    epoch_losses = []
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
        epoch_losses.append(avg_loss)
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

    # Log all metadata to the model
    log_metadata(
        metadata={
            "training": {
                "epochs": epochs,
                "device": str(device),
                "optimizer": "Adam",
                "loss_function": "CrossEntropyLoss",
                "final_loss": float(epoch_losses[-1]) if epoch_losses else 0.0,
                "average_loss": float(sum(epoch_losses) / len(epoch_losses)) if epoch_losses else 0.0,
            },
            "evaluation": {
                "test_accuracy": float(accuracy),
                "test_correct": int(correct),
                "test_total": int(total),
            },
            "model": {
                "architecture": "SimpleCNN",
                "input_channels": 1,
                "num_classes": 10,
            },
        },
        infer_artifact=True,
        artifact_name="trained_model",
    )

    return model, float(accuracy)


@step
def end(test_accuracy: float) -> None:
    """Final step – report test accuracy."""
    print(f"Test Accuracy: {test_accuracy:.2f}%")
    print("NeuralNetworkPipeline is all done.")


@pipeline
def neural_network_pipeline():
    train_loader, test_loader = start()
    model = build_model()
    trained_model, accuracy = train(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
    )
    end(test_accuracy=accuracy)


if __name__ == "__main__":
    # Running the pipeline locally via ZenML's default orchestrator.
    neural_network_pipeline()
