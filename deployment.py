import torch
import torch.nn as nn
from torchvision import models
from huggingface_hub import login, PyTorchModelHubMixin

# Extend your model class with the Hugging Face Hub mixin
class ClothingModel(nn.Module, PyTorchModelHubMixin):
    def __init__(self, num_attributes):
        super().__init__()
        # Load MobileNetV2 pretrained backbone
        backbone = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
        # Replace classifier head with one matching number of attributes
        in_features = backbone.classifier[1].in_features
        backbone.classifier = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(in_features, num_attributes)
        )
        # Copy attributes to self
        self.__dict__.update(backbone.__dict__)

    def forward(self, x):
        return super().forward(x)

# Number of attributes: should match your training labels count
num_attributes = 26  # replace with actual count if different

# Instantiate model and load weights
model = ClothingModel(num_attributes)
model_path = r"E:\clothing-attributes-classification\processed_data\mobilenet_v2_clothing_model.pth"
state_dict = torch.load(model_path, map_location='cpu')
model.load_state_dict(state_dict)

# Login with your token
token = "HF_TOKEN"
login(token=token)

# Push model to hub repo 'chetantiwari/clothing-attributes-model'
model.push_to_hub(repo_id="chetantiwari/clothing-attributes-model", token=token)

print("Model uploaded successfully!")