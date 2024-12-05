from PIL import Image
import numpy as np
from torchvision import transforms

def preprocess_image_with_landmark_focus(image, landmarks, downscale_factor=0.5):
    mask = apply_focus_mask(image, landmarks)
    original_size = image.size
    low_res_size = (int(original_size[0] * downscale_factor), int(original_size[1] * downscale_factor))

    low_res_image = image.resize(low_res_size, Image.BILINEAR).resize(original_size, Image.BILINEAR)
    masked_image = Image.composite(low_res_image, image, mask)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return transform(masked_image).unsqueeze(0)
