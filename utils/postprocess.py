from PIL import Image
import numpy as np

def postprocess_image(tensor, original_size):
    adv_img_np = tensor.squeeze(0).cpu().numpy()
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    adv_img_np = (adv_img_np * std[:, None, None]) + mean[:, None, None]
    adv_img_np = np.clip(adv_img_np, 0, 1)
    adv_img_np = adv_img_np.transpose(1, 2, 0)
    adv_image_pil = Image.fromarray((adv_img_np * 255).astype(np.uint8)).resize(original_size)
    return adv_image_pil
