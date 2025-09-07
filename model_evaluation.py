import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
from sklearn.metrics import (
    confusion_matrix, classification_report,
    accuracy_score, precision_score,
    recall_score, f1_score
)
import seaborn as sns
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image

# --- Dataset Definition ---
class ClothingDataset(Dataset):
    def __init__(self, labels_csv, images_npy, transform=None):
        self.labels_df = pd.read_csv(labels_csv)
        self.images = np.load(images_npy)
        self.transform = transform
        self.labels = self.labels_df.drop(columns=['image_name']).values.astype(np.float32)

    def __len__(self):
        return len(self.labels_df)

    def __getitem__(self, idx):
        image = self.images[idx]
        image = Image.fromarray((image * 255).astype(np.uint8))
        if self.transform:
            image = self.transform(image)
        label = self.labels[idx]
        return image, torch.tensor(label)

# --- Evaluation Utilities ---
def load_model(model_path, num_attrs, device):
    model = models.mobilenet_v2(weights=None)
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.2),
        nn.Linear(in_features, num_attrs)
    )
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model

def evaluate_model(model, dataloader, device, attr_names, output_dir):
    all_labels, all_preds = [], []
    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            outputs = model(images)
            preds = torch.sigmoid(outputs).cpu().numpy() > 0.5
            all_preds.append(preds)
            all_labels.append(labels.numpy())
    y_true = np.vstack(all_labels)
    y_pred = np.vstack(all_preds)

    # Compute and save per-attribute metrics and confusion matrices
    metrics = {}
    for i, attr in enumerate(attr_names):
        y_t, y_p = y_true[:, i], y_pred[:, i]
        metrics[attr] = {
            'accuracy': accuracy_score(y_t, y_p),
            'precision': precision_score(y_t, y_p, zero_division=0),
            'recall': recall_score(y_t, y_p, zero_division=0),
            'f1': f1_score(y_t, y_p, zero_division=0)
        }
        # Confusion matrix plot
        cm = confusion_matrix(y_t, y_p)
        plt.figure(figsize=(4,3))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f'Confusion Matrix: {attr}')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.savefig(os.path.join(output_dir, f'confusion_{attr}.png'))
        plt.close()

    # Save metrics table
    pd.DataFrame(metrics).T.to_csv(os.path.join(output_dir, 'attribute_metrics.csv'))

    # Overall classification report
    report = classification_report(
        y_true, y_pred,
        target_names=attr_names,
        zero_division=0,
        output_dict=True
    )
    pd.DataFrame(report).T.to_csv(os.path.join(output_dir, 'classification_report.csv'))

    return pd.DataFrame(metrics).T

def visualize_predictions(model, dataset, device, attr_names, output_dir, num_samples=10):
    indices = np.random.choice(len(dataset), num_samples, replace=False)
    imgs, lbls = zip(*[dataset[i] for i in indices])
    imgs_tensor = torch.stack(imgs).to(device)
    with torch.no_grad():
        outputs = model(imgs_tensor)
        preds = (torch.sigmoid(outputs).cpu().numpy() > 0.5)

    plt.figure(figsize=(num_samples*2, 4))
    for i in range(num_samples):
        # Predicted
        plt.subplot(2, num_samples, i+1)
        img = imgs[i].permute(1,2,0).cpu().numpy()
        plt.imshow(img); plt.axis('off')
        plt.title('Pred'); plt.xlabel(
            ','.join([attr_names[j] for j,v in enumerate(preds[i]) if v])
        )
        # True
        plt.subplot(2, num_samples, num_samples+i+1)
        plt.imshow(img); plt.axis('off')
        plt.title('True'); plt.xlabel(
            ','.join([attr_names[j] for j,v in enumerate(lbls[i].numpy()) if v])
        )
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'predictions_visualization.png'))
    plt.close()

# --- Main Execution ---
def main():
    data_dir = 'processed_data'
    output_dir = os.path.join(data_dir, 'evaluation')
    os.makedirs(output_dir, exist_ok=True)

    # Load attribute names
    with open(os.path.join(data_dir, 'attribute_names.txt')) as f:
        attr_names = [line.strip() for line in f]

    # Prepare DataLoader for validation set
    from model_training import get_data_transforms, ClothingDataset
    _, val_transform = get_data_transforms()
    val_dataset = ClothingDataset(
        labels_csv=os.path.join(data_dir, 'val_labels.csv'),
        images_npy=os.path.join(data_dir, 'processed_images.npy'),
        transform=val_transform
    )
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load trained model
    model_path = os.path.join(data_dir, 'mobilenet_v2_clothing_model.pth')
    model = load_model(model_path, len(attr_names), device)

    # Evaluate model
    metrics_df = evaluate_model(
        model, val_loader, device, attr_names, output_dir
    )

    # Visualize sample predictions
    visualize_predictions(
        model, val_dataset, device, attr_names, output_dir
    )

    print('Model evaluation complete. Check the evaluation directory for metrics and plots.')

if __name__ == '__main__':
    main()
