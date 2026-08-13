import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models


def main():
    # ---------------------------------
    # Check command line arguments
    # ---------------------------------
    if len(sys.argv) != 2:
        print("Usage: python inference.py <dataset_folder>")
        sys.exit(1)

    data_dir = Path(sys.argv[1])

    if not data_dir.exists():
        print(f"Error: dataset folder does not exist -> {data_dir}")
        sys.exit(1)

    if not data_dir.is_dir():
        print(f"Error: path is not a folder -> {data_dir}")
        sys.exit(1)

    # ---------------------------------
    # Device
    # ---------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---------------------------------
    # Transform
    # Must match validation/test transform
    # ---------------------------------
    test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

    # ---------------------------------
    # Dataset
    # Expects structure like:
    # dataset_folder/
    #   00_Asparagus/
    #   01_Carrots/
    #   ...
    # ---------------------------------
    try:
        test_dataset = datasets.ImageFolder(data_dir, transform=test_transform)
    except Exception as e:
        print(f"Error loading dataset: {e}")
        sys.exit(1)

    if len(test_dataset.classes) == 0:
        print("Error: no class folders found in dataset folder.")
        sys.exit(1)

    # ---------------------------------
    # DataLoader
    # ---------------------------------
    test_loader = DataLoader(
    test_dataset,
    batch_size=64,
    shuffle=False,
    num_workers=0,
    pin_memory=torch.cuda.is_available()
    )
    # ---------------------------------
    # Model
    # Must match training architecture
    # ---------------------------------
    model = models.resnet50(weights=None)
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, len(test_dataset.classes))
    model = model.to(device)

    # ---------------------------------
    # Load trained weights
    # ---------------------------------
    weights_path = Path("best_resnet50.pth")

    if not weights_path.exists():
        print("Error: best_resnet18.pth not found in the current folder.")
        print("Make sure inference.py is in the same folder as best_resnet18.pth")
        sys.exit(1)

    try:
        model.load_state_dict(torch.load(weights_path, map_location=device))
    except Exception as e:
        print(f"Error loading model weights: {e}")
        sys.exit(1)

    model.eval()

    # ---------------------------------
    # Evaluation
    # ---------------------------------
    class_total = [0] * len(test_dataset.classes)
    class_correct = [0] * len(test_dataset.classes)
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            outputs = model(images)
            _, predicted = torch.max(outputs, 1)

            total_correct += (predicted == labels).sum().item()
            total_samples += labels.size(0)

            for label, pred in zip(labels, predicted):
                label_idx = label.item()
                pred_idx = pred.item()

                class_total[label_idx] += 1
                if label_idx == pred_idx:
                    class_correct[label_idx] += 1

    # ---------------------------------
    # Output in required format
    # ---------------------------------
    print(f"Dataset: {data_dir}")
    print("Class Samples Correct Accuracy")

    accuracies = []
    for i, class_name in enumerate(test_dataset.classes):
        samples = class_total[i]
        correct = class_correct[i]
        acc = (correct / samples * 100) if samples > 0 else 0.0
        accuracies.append(acc)

        print(f"{class_name} {samples} {correct} {acc:.2f}%")

    mean_class_acc = sum(accuracies) / len(accuracies)
    overall_acc = (total_correct / total_samples * 100) if total_samples > 0 else 0.0

    print(f"Mean Class Acc: {mean_class_acc:.2f}%")
    print(f"Overall Acc: {overall_acc:.2f}%")


if __name__ == "__main__":
    main()