import os
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


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


def calculate_class_counts(train_dir) -> dict:
    # ספירת מספר התמונות בכל מחלקה
    class_counts = {}
    for class_name in os.listdir(train_dir):
        class_path = os.path.join(train_dir, class_name)
        if os.path.isdir(class_path):  # בדיקה אם זו תקייה
            class_counts[class_name] = count_images_in_directory(class_path)
    return class_counts


label_to_name = {
    0: "A_raw",
    1: "B_almost_ripe",
    2: "C_ripe",
    3: "D_banana_honey"
}


def Calculate_metrics(truth_labels, test_labels) -> None:
    print("-----------------------------metrics-----------------------------")
    accuracy = accuracy_score(truth_labels, test_labels)
    print(f"\nTest Accuracy: {accuracy*100:.3f}%")

    precisions = precision_score(truth_labels, test_labels, average=None, zero_division=1)
    for target_class, precision in enumerate(precisions):
        print(f"Precision for {label_to_name[target_class]}: {precision*100:.3f}%")

    recall = recall_score(truth_labels, test_labels, average=None, zero_division=1)
    for target_class, recall_value in enumerate(recall):
        print(f"Recall for {label_to_name[target_class]}: {recall_value*100:.3f}%")

    f1_weighted = f1_score(truth_labels, test_labels, average='weighted', zero_division=1)
    print(f"Weighted F1 Score: {f1_weighted*100:.3f}%")


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
    # don't matter the order of the labels because we are always predicting the same class
    truth_labels = get_ground_truth_labels(test_dir)
    print("Ground truth labels:")
    print(truth_labels)
    Calculate_metrics(truth_labels, test_labels)


if __name__ == '__main__':
    main()
