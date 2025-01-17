import torch
from matplotlib import pyplot as plt
from torch import nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import datasets, transforms
import torch.optim as optim
from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score
import numpy as np

# Define the Fully Connected Neural Network model
class FullyConnectedNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(FullyConnectedNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

def load_data(batch_size=32):
    data_dir = "data/Banana Images-Real Dataset"
    transform_train = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    transform_val_test = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    train_data = datasets.ImageFolder(root=f"{data_dir}/train", transform=transform_train)
    val_data = datasets.ImageFolder(root=f"{data_dir}/validation", transform=transform_val_test)
    test_data = datasets.ImageFolder(root=f"{data_dir}/test", transform=transform_val_test)

    # Calculate weights for each class
    class_counts = [0] * len(train_data.classes)
    for _, label in train_data:
        class_counts[label] += 1

    class_weights = 1. / torch.tensor(class_counts, dtype=torch.float)
    sample_weights = [class_weights[label] for _, label in train_data]

    sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)

    train_loader = DataLoader(train_data, batch_size=batch_size, sampler=sampler)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader, len(train_data.classes)

# Training loop
def train(model, train_loader, val_loader, optimizer, criterion, epochs=20, device=torch.device("cpu")):
    train_losses, val_losses = [], []
    train_accuracies, val_accuracies = [], []

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        correct_train, total_train = 0, 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()

        train_losses.append(total_loss / len(train_loader))
        train_accuracies.append(100 * correct_train / total_train)

        model.eval()
        correct, total = 0, 0
        val_loss = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        val_losses.append(val_loss / len(val_loader))
        val_accuracies.append(100 * correct / total)

        print(f"Epoch [{epoch + 1}/{epochs}], Train Loss: {train_losses[-1]:.4f}, Val Accuracy: {val_accuracies[-1]:.2f}%")

    plot_metrics(train_losses, val_losses, train_accuracies, val_accuracies)

# Evaluation and metrics
def test(model, test_loader, device=torch.device("cpu")):
    model.eval()
    correct, total = 0, 0
    all_predicted, all_labels = [], []

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)

            all_predicted.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f"Test Accuracy: {100 * correct / total:.2f}%")
    calculate_metrics(all_labels, all_predicted)

def calculate_metrics(true_labels, predicted_labels):
    print("\nMetrics:")
    accuracy = accuracy_score(true_labels, predicted_labels)
    print(f"Accuracy: {accuracy * 100:.2f}%")

    precisions = precision_score(true_labels, predicted_labels, average=None, zero_division=1)
    recalls = recall_score(true_labels, predicted_labels, average=None, zero_division=1)
    f1 = f1_score(true_labels, predicted_labels, average="weighted", zero_division=1)

    for i in range(len(precisions)):
        print(f"Class {i}: Precision: {precisions[i] * 100:.2f}%, Recall: {recalls[i] * 100:.2f}%")
    print(f"Weighted F1 Score: {f1 * 100:.2f}%")

def plot_metrics(train_losses, val_losses, train_accuracies, val_accuracies):
    epochs = range(1, len(train_losses) + 1)
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, label="Train Loss")
    plt.plot(epochs, val_losses, label="Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_accuracies, label="Train Accuracy")
    plt.plot(epochs, val_accuracies, label="Validation Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.title("Training and Validation Accuracy")
    plt.legend()

    plt.tight_layout()
    plt.show()

def main():
    input_dim = 224 * 224 * 3
    hidden_dim = 512
    batch_size = 16
    epochs = 50
    train_loader, val_loader, test_loader, num_classes = load_data(batch_size=batch_size)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = FullyConnectedNN(input_dim, hidden_dim, num_classes).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    criterion = nn.CrossEntropyLoss()

    train(model, train_loader, val_loader, optimizer, criterion, epochs, device)
    test(model, test_loader, device)

if __name__ == "__main__":
    main()
