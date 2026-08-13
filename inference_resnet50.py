import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import sys
import os
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import numpy as np

# Fixed class list for indexing and display
FULL_CLASS_LIST = [
    '00_Asparagus', '01_Carrots', '02_Oysters', '03_Pork', '04_Salmon',
    '05_Zuccini', '06_Strawberries', '07_Sausages', '08_Garlic', '09_Ginger',
    '10_Cauliflower', '11_Capsicum', '12_Pumpkin', '13_Rockmelon', '14_Watermelon',
    '15_Avocado', '16_Tomato', '17_Pineapple', '18_Pears', '19_Apples',
    '20_Peach', '21_Trout', '22_Snapper', '23_Barra', '24_Prawns',
    '25_TropicalFish', '26_Steak', '27_Chicken', '28_Lamb', '29_Mushrooms',
    '30_RedOnion', '31_Tortellini', '32_Blueberries', '33_Lettuce', '34_Milk',
    '35_Eggs', '36_Juice', '37_Kiwi', '38_Butter', '39_Cheese'
]

def get_model_architecture(path):
    """Matches the architecture to the saved model file name."""
    name = os.path.basename(path).lower()
    if "mobilenetv2" in name:
        model = models.mobilenet_v2(weights=None)
        model.classifier = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(model.last_channel, 40)
        )
    else:
        # Default to ResNet50
        model = models.resnet50(weights=None)
        model.fc = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(model.fc.in_features, 40)
        )
    return model

def run_inference(dataset_path, model_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load Model
    model = get_model_architecture(model_path)
    try:
        state_dict = torch.load(model_path, map_location=device)
        # Handle cases where model might be saved as a full dict or just state_dict
        if isinstance(state_dict, dict) and 'model_state_dict' in state_dict:
            model.load_state_dict(state_dict['model_state_dict'])
        else:
            model.load_state_dict(state_dict)
    except Exception as e:
        print(f"Error loading model at {model_path}: {e}")
        return

    model = model.to(device)
    model.eval()

    # Transforms (Matches validation/testing logic)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    dataset = datasets.ImageFolder(root=dataset_path, transform=transform)
    loader = DataLoader(dataset, batch_size=32, shuffle=False)

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    # Output Printing
    print(f"\nDataset: {dataset_path}")
    print(f"{'Class':<20} {'Samples':<8} {'Correct':<8} {'Accuracy':<10}")
    
    accuracies = []
    total_correct = 0

    for i in range(len(FULL_CLASS_LIST)):
        indices = np.where(all_labels == i)[0]
        samples = len(indices)
        if samples > 0:
            correct = np.sum(all_preds[indices] == i)
            acc = (correct / samples) * 100
        else:
            correct = 0
            acc = 0.0
        
        accuracies.append(acc)
        total_correct += correct
        print(f"{FULL_CLASS_LIST[i]:<20} {samples:<8} {correct:<8} {acc:>7.2f}%")

    mean_class_acc = np.mean(accuracies)
    overall_acc = (total_correct / len(all_labels)) * 100

    print("-" * 50)
    print(f"Mean Class Acc: {mean_class_acc:.2f}%")
    print(f"Overall Acc: {overall_acc:.2f}%")

    # Generate Confusion Matrix Visual
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(18, 14))
    sns.heatmap(cm, annot=False, cmap='Blues', 
                xticklabels=FULL_CLASS_LIST, yticklabels=FULL_CLASS_LIST)
    plt.title(f"Confusion Matrix - {os.path.basename(model_path)}")
    plt.tight_layout()
    plt.savefig("confusion_matrix.png")
    print("\nConfusion matrix saved as 'confusion_matrix.png'")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python inference.py <dataset_folder> [model_path]")
    else:
        target_dataset = sys.argv[1]
        # Default model path is 'food_model.pth' unless specified
        target_model = sys.argv[2] if len(sys.argv) > 2 else "food_model.pth"
        run_inference(target_dataset, target_model)
