import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torch.optim as optim
from sklearn.metrics import precision_score, recall_score, accuracy_score

# Define the Convolutional Neural Network model
class ConvolutionalNN(nn.Module):
    def __init__(self, num_classes):
        super(ConvolutionalNN, self).__init__()
        # Convolutional layer 1
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        # Convolutional layer 2
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        # Fully connected layers
        self.fc1 = nn.Linear(64 * 56 * 56, 512)  # Adjust dimensions based on input size
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
        # Apply first convolutional layer + pooling + ReLU activation
        x = self.pool(torch.relu(self.conv1(x)))
        # Apply second convolutional layer + pooling + ReLU activation
        x = self.pool(torch.relu(self.conv2(x)))
        # Flatten the tensor
        x = x.view(x.size(0), -1)
        # Fully connected layer 1 + ReLU activation
        x = torch.relu(self.fc1(x))
        # Fully connected layer 2 (output layer)
        x = self.fc2(x)
        return x  # No need to apply softmax, CrossEntropyLoss handles it

# Load the data
def load_data(batch_size=32):
    data_dir = "data/Banana Images-Real Dataset"
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize images to 224x224
        transforms.ToTensor(),  # Convert images to PyTorch tensors
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalize to [-1, 1]
    ])

    train_data = datasets.ImageFolder(root=f"{data_dir}/train", transform=transform)
    val_data = datasets.ImageFolder(root=f"{data_dir}/validation", transform=transform)
    test_data = datasets.ImageFolder(root=f"{data_dir}/test", transform=transform)

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader, len(train_data.classes)

# Train the model
def train(model, train_loader, val_loader, optimizer, criterion, epochs=10, device="cpu"):
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch [{epoch + 1}/{epochs}], Loss: {total_loss / len(train_loader):.4f}")

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

# Test the model
label_to_name = {
    0: "A_raw",
    1: "B_almost_ripe",
    2: "C_ripe",
    3: "D_banana_honey"
}

def test(model, test_loader, device="cpu"):
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

    accuracy = accuracy_score(all_labels, all_predicted)
    print(f"\nTest Accuracy: {accuracy:.2f}%")

    precisions = precision_score(all_labels, all_predicted, average=None, zero_division=1)
    for target_class, precision in enumerate(precisions):
        print(f"Precision for {label_to_name[target_class]}: {precision:.4f}")

    recall = recall_score(all_labels, all_predicted, average=None, zero_division=1)
    for target_class, recall_value in enumerate(recall):
        print(f"Recall for {label_to_name[target_class]}: {recall_value:.4f}")

# Main function
def main():
    batch_size = 32
    epochs = 10

    train_loader, val_loader, test_loader, num_classes = load_data(batch_size=batch_size)
    print(f"Number of classes: {num_classes}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ConvolutionalNN(num_classes).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    train(model, train_loader, val_loader, optimizer, criterion, epochs, device)
    test(model, test_loader, device)

if __name__ == "__main__":
    main()
