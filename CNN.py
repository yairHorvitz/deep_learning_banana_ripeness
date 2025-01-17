from torch.utils.data import WeightedRandomSampler
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
import torch.optim as optim
from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score
from matplotlib import pyplot as plt

from CNN_for_app import test
from fully_connected_neural_network import plot_metrics





# Balance class weights
def calculate_class_weights(train_loader, device):
    # יצירת מונה לדירוגים
    class_counts = {i: 0 for i in range(len(train_loader.dataset.classes))}
    for _, labels in train_loader:
        for label in labels.cpu().numpy():
            class_counts[label] += 1

    # חישוב המשקלות של כל קטגוריה
    total_samples = sum(class_counts.values())
    class_weights = [total_samples / class_counts[i] for i in range(len(class_counts))]

    # המרת המשקלות למטריצה ב-torch ושליחתה ל-device
    class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)
    return class_weights

# Early stopping mechanism
class EarlyStopping:
    def __init__(self, patience=3, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0
# Load the data
def load_data(batch_size=32):
    data_dir = "data/Banana Images-Real Dataset"

    # Define data augmentations
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(20),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),
        transforms.RandomAffine(15),
        transforms.ToTensor(),  # Make sure ToTensor is here
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    val_test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    train_data = datasets.ImageFolder(root=f"{data_dir}/train", transform=train_transform)
    val_data = datasets.ImageFolder(root=f"{data_dir}/validation", transform=val_test_transform)
    test_data = datasets.ImageFolder(root=f"{data_dir}/test", transform=val_test_transform)

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader, len(train_data.classes)

class ConvolutionalNN(nn.Module):
    def __init__(self, num_classes):
        super(ConvolutionalNN, self).__init__()
        # Convolutional layer 1
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)  # Batch Normalization
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        # Convolutional layer 2
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)  # Batch Normalization
        # Convolutional layer 3
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(128)  # Batch Normalization
        # Fully connected layers
        self.fc1 = nn.Linear(128 * 28 * 28, 512)  # Adjust dimensions based on input size
        self.dropout = nn.Dropout(0.5)  # Dropout
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
        # Apply first convolutional layer + BatchNorm + pooling + ReLU activation
        x = self.pool(torch.relu(self.bn1(self.conv1(x))))
        # Apply second convolutional layer + BatchNorm + pooling + ReLU activation
        x = self.pool(torch.relu(self.bn2(self.conv2(x))))
        # Apply third convolutional layer + BatchNorm + pooling + ReLU activation
        x = self.pool(torch.relu(self.bn3(self.conv3(x))))
        # Flatten the tensor
        x = x.view(x.size(0), -1)
        # Fully connected layer 1 + Dropout + ReLU activation
        x = self.dropout(torch.relu(self.fc1(x)))
        # Fully connected layer 2 (output layer)
        x = self.fc2(x)
        return x

# Modify the training function to include class weights and early stopping
def train(model, train_loader, val_loader, optimizer, criterion, scheduler, epochs=10, device="cpu"):
    train_losses, val_losses = [], []
    train_accuracies, val_accuracies = [], []
    early_stopping = EarlyStopping(patience=5)  # Set patience for early stopping
    class_weights = calculate_class_weights(train_loader, device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)  # Use weighted loss

    for epoch in range(epochs):
        model.train()
        total_loss, correct_train, total_train = 0, 0, 0
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

        train_losses.append(total_loss / len(train_loader))
        train_accuracies.append(100 * correct_train / total_train)

        model.eval()
        correct_val, total_val, val_loss = 0, 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                total_val += labels.size(0)
                correct_val += (predicted == labels).sum().item()

        val_losses.append(val_loss / len(val_loader))
        val_accuracies.append(100 * correct_val / total_val)

        # Adjust learning rate based on validation loss
        scheduler.step(val_loss)

        print(f"Epoch [{epoch + 1}/{epochs}], Train Loss: {total_loss / len(train_loader):.4f}, "
              f"Val Loss: {val_loss / len(val_loader):.4f}, Validation Accuracy: {100 * correct_val / total_val:.2f}%")

        # Check for early stopping
        early_stopping(val_loss)
        if early_stopping.early_stop:
            print(f"Early stopping at epoch {epoch + 1}")
            break

    plot_metrics(train_losses, val_losses, train_accuracies, val_accuracies)


# Main function
def main():
    batch_size = 16
    epochs = 50

    train_loader, val_loader, test_loader, num_classes = load_data(batch_size=batch_size)
    print(f"Number of classes: {num_classes}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ConvolutionalNN(num_classes).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=0.0001)
    criterion = nn.CrossEntropyLoss()

    # Learning Rate Scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3)

    train(model, train_loader, val_loader, optimizer, criterion, scheduler, epochs, device)
    test(model, test_loader, device)

    # Save the model as a TorchScript file after training
    scripted_model = torch.jit.script(model)
    scripted_model.save("banana_classification_model_mobile10.pt")
    print("model saved pt file for application")


if __name__ == "__main__":
    main()
