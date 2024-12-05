from blind_watermark import WaterMark
from PIL import Image
import tempfile
import os

def apply_watermark(image_pil, wm_text="텍스트 삽입", password_img=0, password_wm=0):
    try:
        bwm = WaterMark(password_img=password_img, password_wm=password_wm)
        temp_image_path = tempfile.mktemp(suffix=".png")
        image_pil.save(temp_image_path, format="PNG")

        bwm.read_img(temp_image_path)
        bwm.read_wm(wm_text, mode='str')
        output_path = tempfile.mktemp(suffix=".png")
        bwm.embed(output_path)

        result_image = Image.open(output_path).convert("RGB")
        os.remove(temp_image_path)
        os.remove(output_path)

        return result_image
    except Exception as e:
        print(f"Error in apply_watermark: {str(e)}")
        return image_pil

def extract_watermark(image_pil, password_img=0, password_wm=0):
    bwm = WaterMark(password_img=password_img, password_wm=password_wm)
    temp_image_path = tempfile.mktemp(suffix=".png")
    image_pil.save(temp_image_path, format="PNG")

    extracted_wm_text = bwm.extract(temp_image_path, wm_shape=(32, 32), mode='str')
    os.remove(temp_image_path)
    return extracted_wm_text
