import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
import numpy as np
import pandas as pd
from PIL import Image
from torch.cuda.amp import GradScaler, autocast

class ClothingDataset(Dataset):
    def __init__(self, labels_csv, images_npy, transform=None):
        self.labels_df = pd.read_csv(labels_csv)
        self.images = np.load(images_npy) if images_npy is not None else None   
        self.transform = transform
        # Drop image_name column; keep only label columns
        self.labels = self.labels_df.drop(columns=['image_name']).values.astype(np.float32)

    def __len__(self):
        return len(self.labels_df)

    def __getitem__(self, idx):
        image = None
        if self.images is not None:
            image = self.images[idx]
            image = Image.fromarray((image * 255).astype(np.uint8))  # Converts the normalized image  back to 8-bit format (0-255) and creates a PIL Image
        label = self.labels[idx]       # Extracts the label vector for that index
        if self.transform and image is not None:
            image = self.transform(image)
        return image, torch.tensor(label)   # Returns a tuple containing the processed image and label as a PyTorch tensor


def get_data_transforms():
    """Data augmentations optimized for speed."""
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),   # Randomly crops and resizes images to 224x224 pixels, using 80-100% of the original image area to add variety
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),                    # Converts the PIL Image to a PyTorch tensor and scales pixel values to [0, 1]
        transforms.Normalize([0.485, 0.456, 0.406], 
                             [0.229, 0.224, 0.225])   # Normalizes the image using mean and std values typical for ImageNet
    ])
    val_transform = transforms.Compose([
        transforms.Resize(256),            # Resizes the shorter side of the image to 256 pixels
        transforms.CenterCrop(224),  # Crops the center 224x224 pixels from the resized image
        transforms.ToTensor(),             # Converts the PIL Image to a PyTorch tensor and scales pixel values to [0, 1]
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])  # Normalizes the image using mean and std values typical for ImageNet
    ])
    return train_transform, val_transform


def create_lightweight_model(num_attributes):
    """
    Creates a MobileNetV2-based classifier head.
    MobileNetV2 is fast and lightweight for CPU/GPU.
    """
    model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
    # Replace the classifier
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.2),
        nn.Linear(in_features, num_attributes) # Maps feature representations to the specified number of clothing attributes
    )
    return model


def train_model(model, dataloaders, criterion, optimizer, device, num_epochs=10, use_amp=True):
    """Training loop with optional mixed-precision (AMP) for speed & memory savings."""
    scaler = GradScaler() if use_amp and device.type == 'cuda' else None
    best_wts = model.state_dict()        # Keep track of best model weights
    best_loss = float('inf')           # Initialize best loss to infinity

    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')   # Epoch counter
        for phase in ['train', 'val']:           # Training and validation phases
            model.train() if phase == 'train' else model.eval()  # Set model mode
            running_loss = 0.0                                   # Initialize running loss
            total_samples = 0                             # Initialize sample counter

            for inputs, labels in dataloaders[phase]:       # Iterate over data
                inputs, labels = inputs.to(device), labels.to(device)    # Move data to device
                optimizer.zero_grad()                                  # Zero the parameter gradients

                with autocast(enabled=(scaler is not None and phase == 'train')):  # Mixed precision context
                    outputs = model(inputs)                                      # Forward pass
                    loss = criterion(outputs, labels)                  # Compute loss

                if phase == 'train':                 # Backpropagation and optimization only in training phase
                    if scaler:
                        scaler.scale(loss).backward()  # Scales the loss for numerical stability in mixed precision
                        scaler.step(optimizer)        # Updates model parameters
                        scaler.update()         # Updates the scale for next iteration
                    else:
                        loss.backward()        # Backpropagate the loss
                        optimizer.step()      # Update model parameters

                running_loss += loss.item() * inputs.size(0)      # Accumulate loss
                total_samples += inputs.size(0)          # Accumulate number of samples

            epoch_loss = running_loss / total_samples     # Average loss for the epoch
            print(f'{phase} Loss: {epoch_loss:.4f}')

            if phase == 'val' and epoch_loss < best_loss:       # Save best model weights based on validation loss
                best_loss = epoch_loss           
                best_wts = model.state_dict()

    print('Training complete')
    model.load_state_dict(best_wts)
    return model


def save_model(model, path):
    torch.save(model.state_dict(), path)
    print(f'Model saved to {path}')


def main_training(
    data_dir='../processed_data',
    model_name='mobilenet_v2',
    batch_size=16,
    num_epochs=8,
    learning_rate=1e-3,
    num_workers=0  # Set to 0 for Windows compatibility
):
    # Device setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Paths
    train_csv = os.path.join(data_dir, 'train_labels.csv')
    val_csv   = os.path.join(data_dir, 'val_labels.csv')
    images_npy = os.path.join(data_dir, 'processed_images.npy')

    # Data transforms
    train_transform, val_transform = get_data_transforms()

    # Datasets & loaders
    train_ds = ClothingDataset(train_csv, images_npy, transform=train_transform)
    val_ds   = ClothingDataset(val_csv, images_npy, transform=val_transform)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                            num_workers=0, pin_memory=False)  # Set to 0 for Windows
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                        num_workers=0, pin_memory=False)
    dataloaders = {'train': train_loader, 'val': val_loader}

    # Determine number of attributes
    num_attrs = pd.read_csv(train_csv).shape[1] - 1

    # Model, criterion, optimizer
    model = create_lightweight_model(num_attrs).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Train
    trained = train_model(
        model, dataloaders, criterion, optimizer,
        device, num_epochs=num_epochs, use_amp=True
    )

    # Save
    save_model(trained, os.path.join(data_dir, f'{model_name}_clothing_model.pth'))


if __name__ == '__main__':
    main_training()
