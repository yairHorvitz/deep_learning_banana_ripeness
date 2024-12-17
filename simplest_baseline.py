import os
from sklearn.metrics import accuracy_score, precision_score, recall_score


def count_images_in_directory(directory) -> int:
    total_images = 0
    for root, dirs, files in os.walk(directory):
        # סינון קבצים על פי סיומות של תמונות
        image_files = [f for f in files if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))]
        total_images += len(image_files)
    return total_images


# Generate ground truth labels based on the folder structure of the test dataset
def get_ground_truth_labels(test_dir) -> list:
    labels = []
    for class_name in os.listdir(test_dir):
        class_path = os.path.join(test_dir, class_name)
        if os.path.isdir(class_path):
            # Count the number of images in this class folder
            num_images = count_images_in_directory(class_path)
            # Add labels corresponding to the class name
            labels.extend([class_name] * num_images)
    return labels


def calculate_accuracy(ground_truth_labels, test_labels) -> float:
    correct_predictions = 0
    total_predictions = len(ground_truth_labels)

    for true, predicted in zip(ground_truth_labels, test_labels):
        if true == predicted:
            correct_predictions += 1

    accuracy = correct_predictions / total_predictions
    return accuracy


# def calculate_recall(ground_truth_labels, test_labels, average='weighted'):
#     # יצירת מילונים לספירת true positives, false positives ו-false negatives
#     class_counts = {}
#     true_positives = {}
#     false_positives = {}
#     false_negatives = {}
#
#     # סופר את המופעים של כל מחלקה
#     for true in ground_truth_labels:
#         if true not in class_counts:
#             class_counts[true] = 0
#         class_counts[true] += 1
#
#     print("Class counts:", class_counts)  # דיבאג: הצגת ספירת מחלקות
#
#     for true, predicted in zip(ground_truth_labels, test_labels):
#         # עדכון מניות ההתחזיות לכל מחלקה
#         if true == predicted:
#             if true not in true_positives:
#                 true_positives[true] = 0
#             true_positives[true] += 1
#         else:
#             if true not in false_negatives:
#                 false_negatives[true] = 0
#             false_negatives[true] += 1
#
#             if predicted not in false_positives:
#                 false_positives[predicted] = 0
#             false_positives[predicted] += 1
#
#     # הצגת הערכים של true positives, false positives ו-false negatives עבור דיבאג
#     print("True Positives:", true_positives)  # דיבאג: הצגת true positives
#     print("False Positives:", false_positives)  # דיבאג: הצגת false positives
#     print("False Negatives:", false_negatives)  # דיבאג: הצגת false negatives
#
#     # חישוב Recall לכל מחלקה
#     recall_per_class = {}
#     for class_name in true_positives:
#         tp = true_positives[class_name]
#         fn = false_negatives.get(class_name, 0)
#         recall_per_class[class_name] = tp / (tp + fn) if (tp + fn) != 0 else 0
#
#     print("Recall per class:", recall_per_class)  # דיבאג: הצגת recall per class
#
#     # חישוב Recall ממוצע לפי weighted
#     if average == 'weighted':
#         total_samples = len(ground_truth_labels)
#         weighted_recall = sum(recall_per_class[class_name] * class_counts.get(class_name, 0)
#                               for class_name in recall_per_class) / total_samples
#         print(f"Weighted Recall: {weighted_recall}")  # דיבאג: הצגת weighted recall
#         return weighted_recall
#     else:
#         return recall_per_class

def calculate_class_counts(train_dir) -> dict:
    # ספירת מספר התמונות בכל מחלקה
    class_counts = {}
    for class_name in os.listdir(train_dir):
        class_path = os.path.join(train_dir, class_name)
        if os.path.isdir(class_path):  # בדיקה אם זו תקייה
            class_counts[class_name] = count_images_in_directory(class_path)
            print(f"class_path:[{class_name}] {class_counts[class_name]}")
    return class_counts


def Calculate_metrics(ground_truth_labels, test_labels) -> None:
    accuracy = accuracy_score(ground_truth_labels, test_labels)
    precision = precision_score(ground_truth_labels, test_labels, average='weighted', zero_division=0)
    recall = recall_score(ground_truth_labels, test_labels, average='weighted')
    print(f"Test Accuracy: {accuracy:.2f}%")
    print(f"Precision: {precision:.2f}")
    print(f"Recall: {recall:.2f}")


def main():
    # נתיב לתקיית האימון
    train_dir = "data/Banana Images-Real Dataset/train"
    test_dir = "data/Banana Images-Real Dataset/test"
    validation_dit = "data/Banana Images-Real Dataset/validation"

    class_counts = calculate_class_counts(train_dir)
    print("Class counts:", class_counts)

    # מציאת המחלקה הדומיננטית
    majority_class = max(class_counts, key=class_counts.get)
    print("Majority class:", majority_class)

    test_images = count_images_in_directory(test_dir)
    test_labels = [majority_class] * test_images  # predicting the majority class for all test images

    print("test labels", test_labels)
    print("size test labels", len(test_labels))

    ground_truth_labels = get_ground_truth_labels(test_dir)
    print("Ground truth labels:")
    print(ground_truth_labels)
    print(len(ground_truth_labels))

    Calculate_metrics(ground_truth_labels, test_labels)


if __name__ == '__main__':
    main()
