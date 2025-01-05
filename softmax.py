import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torch.optim as optim
from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score
import matplotlib.pyplot as plt


# Define the Softmax classification model
class SoftmaxClassifier(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(SoftmaxClassifier, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)  # Single linear layer

    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten the images to (batch_size, input_dim)
        logits = self.linear(x)  # Linear transformation
        return logits  # No need to apply softmax (CrossEntropyLoss handles it)


def load_data(batch_size=32):
    # Path to your dataset
    data_dir = "data/Banana Images-Real Dataset"

    # Define image transformations
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize images to 224x224
        transforms.ToTensor(),  # Convert images to PyTorch tensors
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalize to [-1, 1]
    ])

    # Load train, validation, and test datasets
    train_data = datasets.ImageFolder(root=f"{data_dir}/train", transform=transform)
    val_data = datasets.ImageFolder(root=f"{data_dir}/validation", transform=transform)
    test_data = datasets.ImageFolder(root=f"{data_dir}/test", transform=transform)

    # Create DataLoaders
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader, len(train_data.classes)


def count_images_per_category(data_loader):
    # Dictionary to hold count of images per category
    label_counts = {i: 0 for i in range(len(data_loader.dataset.classes))}

    # Count images for each category
    for images, labels in data_loader:
        for label in labels.cpu().numpy():
            label_counts[label] += 1

    return label_counts


def train(model, train_loader, val_loader, optimizer, criterion, epochs=10, device=torch.device("cpu")):
    # Dictionary to store image counts per category for training data
    label_counts = count_images_per_category(train_loader)

    # Lists to store data for graphing
    train_losses, val_losses = [], []
    train_accuracies, val_accuracies = [], []

    # Training loop
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        correct_train, total_train = 0, 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()  # Zero gradients
            outputs = model(images)  # Forward pass
            loss = criterion(outputs, labels)  # Compute loss
            loss.backward()  # Backpropagation
            optimizer.step()  # Update weights
            total_loss += loss.item()

            # Calculate accuracy for training set
            _, predicted = torch.max(outputs, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()

        # Calculate training loss and accuracy
        train_losses.append(total_loss / len(train_loader))
        train_accuracies.append(100 * correct_train / total_train)

        print(f"Epoch [{epoch + 1}/{epochs}], Loss: {total_loss / len(train_loader):.4f}")

        # Validate the model
        model.eval()
        correct, total = 0, 0
        val_loss = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)  # Compute validation loss
                val_loss += loss.item()

                # Calculate accuracy for validation set
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        val_losses.append(val_loss / len(val_loader))
        val_accuracies.append(100 * correct / total)

        print(f"Validation Accuracy: {100 * correct / total:.2f}%")

    # Print the number of images for each category at the end of training
    print("\nNumber of images for each category in the training set:")
    for target_class, count in label_counts.items():
        print(f"{label_to_name[target_class]}: {count} images")

    # Plot training/validation loss and accuracy
    plot_metrics(train_losses, val_losses, train_accuracies, val_accuracies)


def plot_metrics(train_losses, val_losses, train_accuracies, val_accuracies, test_accuracy=None):
    epochs = range(1, len(train_losses) + 1)

    plt.figure(figsize=(12, 6))

    # Loss Plot
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, label="Train Loss", color='blue')
    plt.plot(epochs, val_losses, label="Validation Loss", color='red')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Loss during Training and Validation')
    plt.legend()

    # Accuracy Plot
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_accuracies, label="Train Accuracy", color='blue')
    plt.plot(epochs, val_accuracies, label="Validation Accuracy", color='red')

    if test_accuracy is not None:
        plt.axhline(test_accuracy, color='green', linestyle='--', label="Test Accuracy")

    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.title('Accuracy during Training and Validation')
    plt.legend()

    plt.tight_layout()
    plt.show()


label_to_name = {
    0: "A_raw",
    1: "B_almost_ripe",
    2: "C_ripe",
    3: "D_banana_honey"
}


def test(model, test_loader, device=torch.device("cpu"), print_predict=False):
    model.eval()
    correct, total = 0, 0
    all_predicted, all_labels = [], []

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)

            # Store predictions and labels
            all_predicted.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    calculate_metrics(all_labels, all_predicted)
    test_accuracy = 100 * correct / total

    print(f"\nTest Accuracy: {test_accuracy:.2f}%")

    if print_predict:
        print("\nPrediction Results:")
        for i in range(len(all_labels)):
            pred_name = label_to_name[all_predicted[i]]
            actual_name = label_to_name[all_labels[i]]
            status = "Correct" if pred_name == actual_name else "Wrong"
            print(f"Sample {i + 1}: Predicted = {pred_name}, Actual = {actual_name} -> {status}")

    return test_accuracy


def calculate_metrics(truth_labels, test_labels):
    print("-----------------------------metrics-----------------------------")
    accuracy = accuracy_score(truth_labels, test_labels)
    print(f"\nTest Accuracy: {accuracy * 100:.3f}%")

    precisions = precision_score(truth_labels, test_labels, average=None, zero_division=1)
    for target_class, precision in enumerate(precisions):
        print(f"Precision for {label_to_name[target_class]}: {precision * 100:.3f}%")

    recall = recall_score(truth_labels, test_labels, average=None, zero_division=1)
    for target_class, recall_value in enumerate(recall):
        print(f"Recall for {label_to_name[target_class]}: {recall_value * 100:.3f}%")

    f1_weighted = f1_score(truth_labels, test_labels, average='weighted', zero_division=1)
    print(f"Weighted F1 Score: {f1_weighted * 100:.3f}%")


def main():
    input_dim = 224 * 224 * 3  # all images are 224x224 pixels with 3 channels (RGB)
    batch_size = 32
    epochs = 10

    # Load data
    train_loader, val_loader, test_loader, num_classes = load_data(batch_size=batch_size)

    # Print number of classes
    print(f"Number of classes: {num_classes}")

    # Initialize model, optimizer, and loss function
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SoftmaxClassifier(input_dim, num_classes).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    # Train the model
    train(model, train_loader, val_loader, optimizer, criterion, epochs, device=device)

    # Test the model
    test(model, test_loader, device=device, print_predict=False)


if __name__ == "__main__":
    main()
