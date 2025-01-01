import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
import os
import joblib
import json
from dataclasses import dataclass
from typing import Tuple, Dict, Any
from huggingface_hub import HfApi, HfFolder
from dotenv import load_dotenv
from huggingface_hub import login

# Load environment variables from .env file
load_dotenv()

# Login using HF_TOKEN
login(token=os.getenv("HF_TOKEN"))

# Variable to set where the artifacts will be stored
ARTIFACTS_PATH = "./local_artifacts"
HUGGINGFACE_MODEL_REPO = os.getenv("HUGGINGFACE_MODEL_REPO")


@dataclass
class ModelConfig:
    input_dim: int
    output_dim: int
    learning_rate: float = 0.1
    momentum: float = 0.9
    batch_size: int = 32
    num_epochs: int = 1000
    early_stopping_patience: int = 5
    early_stopping_delta: float = 0.001


class LogisticRegression(nn.Module):
    def __init__(self, input_dim: int, output_dim: int):
        super(LogisticRegression, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)

        # Initialize weights properly for logistic regression
        nn.init.xavier_uniform_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.linear(x)
        return torch.softmax(logits, dim=1)


class IrisModel:
    def __init__(self, config: ModelConfig):
        self.config = config
        self.model = LogisticRegression(config.input_dim, config.output_dim)
        self.scaler = StandardScaler()
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(
            self.model.parameters(), lr=config.learning_rate, momentum=config.momentum
        )

    def preprocess_data(self, X: np.ndarray, fit: bool = False) -> torch.Tensor:
        """Preprocess input data with scaling."""
        if fit:
            X_scaled = self.scaler.fit_transform(X)
        else:
            X_scaled = self.scaler.transform(X)
        return torch.FloatTensor(X_scaled)

    def train(self, X: np.ndarray, y: np.ndarray) -> Dict[str, list]:
        """Train the model and return training history."""
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        # Preprocess data
        X_train_tensor = self.preprocess_data(X_train, fit=True)
        X_val_tensor = self.preprocess_data(X_val)
        y_train_tensor = torch.LongTensor(y_train)
        y_val_tensor = torch.LongTensor(y_val)

        history = {"train_loss": [], "val_loss": [], "val_accuracy": []}

        best_val_loss = float("inf")
        patience_counter = 0

        for epoch in range(self.config.num_epochs):
            # Training
            self.model.train()
            indices = torch.randperm(len(X_train_tensor))
            total_train_loss = 0
            num_batches = len(X_train_tensor) // self.config.batch_size

            for i in range(num_batches):
                start_idx = i * self.config.batch_size
                end_idx = start_idx + self.config.batch_size

                batch_X = X_train_tensor[indices[start_idx:end_idx]]
                batch_y = y_train_tensor[indices[start_idx:end_idx]]

                outputs = self.model(batch_X)
                loss = self.criterion(outputs, batch_y)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                total_train_loss += loss.item()

            avg_train_loss = total_train_loss / num_batches

            # Validation
            val_loss, val_accuracy = self.evaluate(X_val_tensor, y_val_tensor)

            # Store metrics
            history["train_loss"].append(avg_train_loss)
            history["val_loss"].append(val_loss)
            history["val_accuracy"].append(val_accuracy)

            if (epoch + 1) % 50 == 0:
                print(f"Epoch [{epoch+1}/{self.config.num_epochs}]")
                print(f"Train Loss: {avg_train_loss:.4f}")
                print(f"Val Loss: {val_loss:.4f}")
                print(f"Val Accuracy: {val_accuracy:.4f}")

            # Early stopping
            if val_loss < best_val_loss - self.config.early_stopping_delta:
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= self.config.early_stopping_patience:
                print(f"Early stopping triggered at epoch {epoch+1}")
                break

        # Save model and scaler to Hugging Face Hub
        self.save_to_huggingface()

        return history

    def save_to_huggingface(self):
        """Save the model and scaler to Hugging Face Hub."""
        model_state = {
            "model_state_dict": self.model.state_dict(),
            "config": self.config.__dict__,
        }
        torch.save(model_state, "model.pth")
        joblib.dump(self.scaler, "scaler.joblib")

        # Upload model and scaler to Hugging Face
        HfApi().upload_file(
            path_or_fileobj="model.pth",
            path_in_repo="model.pth",
            repo_id=HUGGINGFACE_MODEL_REPO,
            repo_type="model",
        )
        HfApi().upload_file(
            path_or_fileobj="scaler.joblib",
            path_in_repo="scaler.joblib",
            repo_id=HUGGINGFACE_MODEL_REPO,
            repo_type="model",
        )

    def evaluate(self, X: torch.Tensor, y: torch.Tensor) -> Tuple[float, float]:
        """Evaluate the model and return loss and accuracy."""
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(X)
            loss = self.criterion(outputs, y).item()
            _, predicted = torch.max(outputs.data, 1)
            accuracy = (predicted == y).sum().item() / len(y)
        return loss, accuracy

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions on new data."""
        self.model.eval()
        X_tensor = self.preprocess_data(X)
        with torch.no_grad():
            outputs = self.model(X_tensor)
            _, predicted = torch.max(outputs.data, 1)
        return predicted.numpy()

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Get probability predictions."""
        self.model.eval()
        X_tensor = self.preprocess_data(X)
        with torch.no_grad():
            probabilities = self.model(X_tensor)
        return probabilities.numpy()

    def save(self, path: str):
        """Save the model and its components."""
        os.makedirs(os.path.dirname(path), exist_ok=True)

        # Save model state
        model_state = {
            "model_state_dict": self.model.state_dict(),
            "config": self.config.__dict__,
        }
        torch.save(model_state, f"{path}/model.pth")

        # Save scaler
        joblib.dump(self.scaler, f"{path}/scaler.joblib")

    @classmethod
    def load(cls, path: str) -> "IrisModel":
        """Load a saved model."""
        # Load model state
        model_state = torch.load(f"{path}/model.pth", weights_only=True)

        # Reconstruct config
        config = ModelConfig(**model_state["config"])

        # Create model instance
        iris_model = cls(config)

        # Load model weights
        iris_model.model.load_state_dict(model_state["model_state_dict"])

        # Load scaler
        iris_model.scaler = joblib.load(f"{path}/scaler.joblib")

        return iris_model


def main():
    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)

    # Load data
    iris = load_iris()
    X, y = iris.data, iris.target

    # Create model with config
    config = ModelConfig(
        input_dim=4,
        output_dim=3,
        learning_rate=0.001,
        momentum=0.9,
        batch_size=32,
        num_epochs=1000,
        early_stopping_patience=5,
        early_stopping_delta=0.001,
    )

    # Initialize and train model
    iris_model = IrisModel(config)
    history = iris_model.train(X, y)

    # Save training history
    with open(f"{ARTIFACTS_PATH}/training_history.json", "w") as f:
        json.dump(history, f)

    # Save model
    iris_model.save(ARTIFACTS_PATH)


if __name__ == "__main__":
    main()
