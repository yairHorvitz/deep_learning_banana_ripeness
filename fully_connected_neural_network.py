import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torch.optim as optim
from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score


# Define the Fully Connected Neural Network model
class FullyConnectedNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(FullyConnectedNN, self).__init__()
        # Hidden layer 1 (fully connected)
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        # Hidden layer 2 (fully connected)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        # Output layer (fully connected)
        self.fc3 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # Flatten the input image to a vector
        x = x.view(x.size(0), -1)
        # Apply first hidden layer + ReLU activation
        x = torch.relu(self.fc1(x))
        # Apply second hidden layer + ReLU activation
        x = torch.relu(self.fc2(x))
        # Apply the output layer
        x = self.fc3(x)
        return x  # No need to apply softmax, CrossEntropyLoss handles it


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

def count_images_per_category(data_loader):
    # מילון לשמירת מספר התמונות לכל קטגוריה
    label_counts = {i: 0 for i in range(len(data_loader.dataset.classes))}

    # חישוב מספר התמונות לכל קטגוריה
    for images, labels in data_loader:
        for label in labels.cpu().numpy():
            label_counts[label] += 1

    return label_counts


# Training loop
def train(model, train_loader, val_loader, optimizer, criterion, epochs=10, device=torch.device("cpu")):
    # מילון לשמירת מספר התמונות לכל קטגוריה בסט האימון
    label_counts = count_images_per_category(train_loader)
    # לולאת האימון
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

    # הדפסת מספר התמונות לכל קטגוריה בסוף האימון
    print("\nNumber of images for each category in the training set:")
    for target_class, count in label_counts.items():
        print(f"{label_to_name[target_class]}: {count} images")


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

            # שמירה של הערכים
            all_predicted.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    Calculate_metrics(all_labels, all_predicted)
    if print_predict:
        # הדפסת התחזיות לעומת התגיות לפי שמות
        print("\nPrediction Results:")
        for i in range(len(all_labels)):
            pred_name = label_to_name[all_predicted[i]]
            actual_name = label_to_name[all_labels[i]]
            status = "Correct" if pred_name == actual_name else "Wrong"
            print(f"Sample {i + 1}: Predicted = {pred_name}, Actual = {actual_name} -> {status}")

def Calculate_metrics(truth_labels, test_labels) -> None:
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
    input_dim = 224 * 224 * 3
    batch_size = 32
    epochs = 10
    hidden_dim = 512  # You can adjust the hidden layer dimension

    train_loader, val_loader, test_loader, num_classes = load_data(batch_size=batch_size)
    print(f"Number of classes: {num_classes}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FullyConnectedNN(input_dim, hidden_dim, num_classes).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    train(model, train_loader, val_loader, optimizer, criterion, epochs, device)
    test(model, test_loader, device)


if __name__ == "__main__":
    main()
