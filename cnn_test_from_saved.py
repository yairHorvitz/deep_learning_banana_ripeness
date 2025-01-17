import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# פונקציה לטעינת המודל
def load_model(model_path, device):
    # טען את המודל השמור
    model = torch.jit.load(model_path)
    model.to(device)
    model.eval()
    print("Model loaded successfully!")
    return model

# פונקציה לבדיקת המודל על סט הבדיקה
def test_model(model, test_loader, device):
    correct = 0
    total = 0
    all_labels = []
    all_predictions = []

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())

    accuracy = 100 * correct / total
    print(f"Test Accuracy: {accuracy:.2f}%")
    return all_labels, all_predictions

# פונקציה להצגת גרף של מטריקות
def plot_confusion_matrix(labels, predictions, class_names):
    cm = confusion_matrix(labels, predictions, labels=range(len(class_names)))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(cmap=plt.cm.Blues, xticks_rotation=45)
    plt.title("Confusion Matrix")
    plt.show()

# פונקציה ראשית
def main():
    # הגדר נתיב למודל השמור ונתיב לנתונים
    model_path = "banana_classification_model_mobile2.pt"
    data_dir = "data/Banana Images-Real Dataset"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # טען את המודל
    model = load_model(model_path, device)

    # הגדר את DataLoader לסט הבדיקה
    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    test_data = datasets.ImageFolder(root=f"{data_dir}/test", transform=test_transform)
    test_loader = DataLoader(test_data, batch_size=16, shuffle=False)

    # הרץ את המודל על סט הבדיקה
    labels, predictions = test_model(model, test_loader, device)

    # הצג דוח ביצועים
    print("\nClassification Report:")
    print(classification_report(labels, predictions, target_names=test_data.classes))

    # הצג גרף מטריקס הבלבול
    plot_confusion_matrix(labels, predictions, test_data.classes)

if __name__ == "__main__":
    main()
