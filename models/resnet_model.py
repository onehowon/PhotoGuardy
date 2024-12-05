import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights

def initialize_resnet(num_classes=10):
    model = resnet50(weights=ResNet50_Weights.DEFAULT)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    return model
