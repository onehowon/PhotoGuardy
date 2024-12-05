import torch.nn as nn
from torchvision.models import vgg16, VGG16_Weights

def initialize_vgg(num_classes=10):
    model = vgg16(weights=VGG16_Weights.DEFAULT)
    num_ftrs = model.classifier[6].in_features
    model.classifier[6] = nn.Linear(num_ftrs, num_classes)
    return model
