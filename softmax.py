import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torch.optim as optim

# Define the Softmax classification model
class SoftmaxClassifier(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(SoftmaxClassifier, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)  # Single linear layer

    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten the images to (batch_size, input_dim)
        logits = self.linear(x)    # Linear transformation
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


# Training loop
def train(model, train_loader, val_loader, optimizer, criterion, epochs=10, device="cpu"):
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()  # Zero gradients
            outputs = model(images)  # Forward pass
            loss = criterion(outputs, labels)  # Compute loss
            loss.backward()  # Backpropagation
            optimizer.step()  # Update weights
            total_loss += loss.item()

        print(f"Epoch [{epoch + 1}/{epochs}], Loss: {total_loss / len(train_loader):.4f}")

        # Validate the model
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        print(f"Validation Accuracy: {100 * correct / total:.2f}%")


def test(model, test_loader, device="cpu"):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f"Test Accuracy: {100 * correct / total:.2f}%")


def main():
    # Define the input and output dimensions
    input_dim = 224 * 224 * 3  # all pictures are 224x224 pixels with 3 channels(RGB)
    batch_size = 32
    epochs = 10

    # Load the data
    train_loader, val_loader, test_loader, num_classes = load_data(batch_size=batch_size)

    # Print number of classes
    print(f"Number of classes: {num_classes}")

    # Initialize the model, optimizer, and loss function
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SoftmaxClassifier(input_dim, num_classes).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    # Train and test the model
    train(model, train_loader, val_loader, optimizer, criterion, epochs, device)
    test(model, test_loader, device)


if __name__ == "__main__":
    main()
