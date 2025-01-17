import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torch.optim as optim
import matplotlib.pyplot as plt

# Define the Softmax classification model
class SoftmaxClassifier(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(SoftmaxClassifier, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten input
        return self.linear(x)  # Output logits


# Data preprocessing (without augmentation)
def load_data(batch_size=32):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalize to [-1, 1]
    ])

    data_dir = "data/Banana Images-Real Dataset"
    train_data = datasets.ImageFolder(root=f"{data_dir}/train", transform=transform)
    val_data = datasets.ImageFolder(root=f"{data_dir}/validation", transform=transform)
    test_data = datasets.ImageFolder(root=f"{data_dir}/test", transform=transform)

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader, len(train_data.classes)


# Training function with Early Stopping and metrics
def train(model, train_loader, val_loader, optimizer, criterion, scheduler, epochs=30, device=torch.device("cpu")):
    train_losses, val_losses = [], []
    train_accuracies, val_accuracies = [], []
    best_val_accuracy = 0  # Best validation accuracy
    patience, patience_counter = 10, 0  # Early Stopping

    # Initialize per-class loss lists
    train_class_losses = {i: [] for i in range(len(train_loader.dataset.classes))}
    val_class_losses = {i: [] for i in range(len(train_loader.dataset.classes))}

    for epoch in range(epochs):
        model.train()
        total_loss, correct_train, total_train = 0, 0, 0
        class_losses_train = {i: 0 for i in range(len(train_loader.dataset.classes))}

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct_train += (predicted == labels).sum().item()
            total_train += labels.size(0)

            # Update per-class loss for train data
            for label in labels.unique():
                class_losses_train[label.item()] += loss.item()

        train_losses.append(total_loss / len(train_loader))
        train_accuracies.append(100 * correct_train / total_train)

        # Record per-class loss
        for i in range(len(train_loader.dataset.classes)):
            train_class_losses[i].append(class_losses_train[i] / len(train_loader))

        # Validation
        model.eval()
        val_loss, correct_val, total_val = 0, 0, 0
        class_losses_val = {i: 0 for i in range(len(val_loader.dataset.classes))}

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                correct_val += (predicted == labels).sum().item()
                total_val += labels.size(0)

                # Update per-class loss for validation data
                for label in labels.unique():
                    class_losses_val[label.item()] += loss.item()

        val_losses.append(val_loss / len(val_loader))
        val_accuracy = 100 * correct_val / total_val
        val_accuracies.append(val_accuracy)

        # Record per-class loss
        for i in range(len(val_loader.dataset.classes)):
            val_class_losses[i].append(class_losses_val[i] / len(val_loader))

        # Early Stopping and Model Checkpoint
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            patience_counter = 0
            torch.save(model.state_dict(), "best_softmax_model.pth")
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print("Early stopping triggered!")
            break

        # Step the scheduler
        scheduler.step()

        print(f"Epoch [{epoch + 1}/{epochs}], Train Loss: {train_losses[-1]:.4f}, "
              f"Val Accuracy: {val_accuracy:.2f}%")

    # Plot metrics
    plot_metrics(train_losses, val_losses, train_accuracies, val_accuracies, train_class_losses, val_class_losses)


# Plot training and validation metrics
def plot_metrics(train_losses, val_losses, train_accuracies, val_accuracies, train_class_losses, val_class_losses):
    epochs = range(1, len(train_losses) + 1)

    plt.figure(figsize=(12, 6))

    # Plot Train and Validation Loss & Accuracy
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, label="Train Loss")
    plt.plot(epochs, val_losses, label="Validation Loss")
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_accuracies, label="Train Accuracy")
    plt.plot(epochs, val_accuracies, label="Validation Accuracy")
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.legend()

    plt.tight_layout()
    plt.show()

    # Plot per-class loss for train and validation
    for i in range(len(train_class_losses)):
        plt.plot(epochs, train_class_losses[i], label=f"Train Loss Class {i}")
        plt.plot(epochs, val_class_losses[i], label=f"Val Loss Class {i}", linestyle='--')

    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.tight_layout()
    plt.show()


# Main function
def main():
    input_dim = 224 * 224 * 3
    batch_size = 32
    epochs = 50
    learning_rate = 0.0001

    train_loader, val_loader, test_loader, num_classes = load_data(batch_size)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SoftmaxClassifier(input_dim, num_classes).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
    criterion = nn.CrossEntropyLoss()

    train(model, train_loader, val_loader, optimizer, criterion, scheduler, epochs, device)


if __name__ == "__main__":
    main()
