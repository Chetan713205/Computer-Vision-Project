import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import numpy as np

# Load attribute names from file
def load_attribute_names(filepath):
    with open(filepath, 'r') as f:
        return [line.strip() for line in f.readlines()]

# Create the model architecture matching before
def create_lightweight_model(num_attributes):
    model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.2),
        nn.Linear(in_features, num_attributes)
    )
    return model

# Preprocessing transform (matches validation transform in training)
def get_preprocess_transform():
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                           [0.229, 0.224, 0.225])
    ])

# Load and preprocess image
def preprocess_image(image_path, transform):
    image = Image.open(image_path).convert('RGB')
    return transform(image).unsqueeze(0)  # add batch dim

# Modified function to return real probability values
def predict_attributes_with_probabilities(model, input_tensor):
    model.eval()
    with torch.no_grad():
        outputs = model(input_tensor)
        probs = torch.sigmoid(outputs)
        return probs.cpu().numpy().flatten()

# Optional: Keep the original boolean prediction function
def predict_attributes(model, input_tensor, threshold=0.5):
    model.eval()
    with torch.no_grad():
        outputs = model(input_tensor)
        probs = torch.sigmoid(outputs)
        preds = (probs >= threshold).cpu().numpy().flatten()
        return preds, probs.cpu().numpy().flatten()

def main():
    model_path = 'processed_data/mobilenet_v2_clothing_model.pth'
    attributes_path = 'processed_data/attribute_names.txt'
    image_path = 'test_photo.jpg'

    # Load attribute names
    attribute_names = load_attribute_names(attributes_path)
    num_attributes = len(attribute_names)

    # Create model and load weights
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = create_lightweight_model(num_attributes)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)

    # Preprocess image
    preprocess = get_preprocess_transform()
    input_tensor = preprocess_image(image_path, preprocess).to(device)

    # Get probability values
    probabilities = predict_attributes_with_probabilities(model, input_tensor)
    
    # Print predicted probabilities
    print(f"Predicted probabilities for {image_path}:")
    print("-" * 50)
    
    for attr, prob in zip(attribute_names, probabilities):
        confidence = "High" if prob > 0.7 or prob < 0.3 else "Medium"
        print(f"{attr:20}: {prob:.4f} ({confidence} confidence)")
    
    print("\n" + "="*50)
    print("Attributes with probability > 0.5:")
    for attr, prob in zip(attribute_names, probabilities):
        if prob > 0.5:
            print(f"{attr:20}: {prob:.4f}")

if __name__ == '__main__':
    main()
