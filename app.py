import gradio as gr
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, models
from art.attacks.evasion import (
    FastGradientMethod, CarliniL2Method, DeepFool, AutoAttack,
    ProjectedGradientDescent, BasicIterativeMethod, SpatialTransformation,
    MomentumIterativeMethod, SaliencyMapMethod, NewtonFool
)
from art.estimators.classification import PyTorchClassifier
from PIL import Image, ImageOps
import numpy as np
import os
from blind_watermark import WaterMark
from torchvision.models import resnet50, vgg16, ResNet50_Weights, VGG16_Weights
import tempfile

resnet_model = resnet50(weights=ResNet50_Weights.DEFAULT)
num_ftrs_resnet = resnet_model.fc.in_features
resnet_model.fc = nn.Linear(num_ftrs_resnet, 10)
resnet_model = resnet_model.to("cuda" if torch.cuda.is_available() else "cpu")

vgg_model = vgg16(weights=VGG16_Weights.DEFAULT)
num_ftrs_vgg = vgg_model.classifier[6].in_features
vgg_model.classifier[6] = nn.Linear(num_ftrs_vgg, 10)
vgg_model = vgg_model.to("cuda" if torch.cuda.is_available() else "cpu")

criterion = nn.CrossEntropyLoss()
optimizer_resnet = optim.Adam(resnet_model.parameters(), lr=0.001)
optimizer_vgg = optim.Adam(vgg_model.parameters(), lr=0.001)

resnet_classifier = PyTorchClassifier(
    model=resnet_model,
    loss=criterion,
    optimizer=optimizer_resnet,
    input_shape=(3, 224, 224),
    nb_classes=10,
)

vgg_classifier = PyTorchClassifier(
    model=vgg_model,
    loss=criterion,
    optimizer=optimizer_vgg,
    input_shape=(3, 224, 224),
    nb_classes=10,
)

models_dict = {
    "ResNet50": resnet_classifier,
    "VGG16": vgg_classifier
}

def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return transform(image).unsqueeze(0).to("cuda" if torch.cuda.is_available() else "cpu")

def postprocess_image(tensor, original_size):
    adv_img_np = tensor.squeeze(0).cpu().numpy()
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    adv_img_np = (adv_img_np * std[:, None, None]) + mean[:, None, None]
    adv_img_np = np.clip(adv_img_np, 0, 1)
    adv_img_np = adv_img_np.transpose(1, 2, 0)
    adv_image_pil = Image.fromarray((adv_img_np * 255).astype(np.uint8)).resize(original_size)
    return adv_image_pil

def generate_adversarial_image(image, model_name, attack_types, eps_value):
    original_size = image.size
    img_tensor = preprocess_image(image)

    classifier = models_dict[model_name]

    try:
        for attack_type in attack_types:
            if attack_type == "FGSM":
                attack = FastGradientMethod(estimator=classifier, eps=eps_value)
            elif attack_type == "C&W":
                attack = CarliniL2Method(classifier=classifier, confidence=0.05)
            elif attack_type == "DeepFool":
                attack = DeepFool(classifier=classifier, max_iter=20)
            elif attack_type == "AutoAttack":
                attack = AutoAttack(estimator=classifier, eps=eps_value, batch_size=1)
            elif attack_type == "PGD":
                attack = ProjectedGradientDescent(estimator=classifier, eps=eps_value, eps_step=eps_value / 10, max_iter=40)
            elif attack_type == "BIM":
                attack = BasicIterativeMethod(estimator=classifier, eps=eps_value, eps_step=eps_value / 10, max_iter=10)
            elif attack_type == "STA":
                attack = SpatialTransformation(estimator=classifier, max_translation=0.2)
            elif attack_type == "MIM":
                attack = MomentumIterativeMethod(estimator=classifier, eps=eps_value, eps_step=eps_value / 10, max_iter=10)
            elif attack_type == "JSMA":
                attack = SaliencyMapMethod(classifier=classifier, theta=0.1, gamma=0.1)
            elif attack_type == "NewtonFool":
                attack = NewtonFool(classifier=classifier, max_iter=20)

            adv_img_np = attack.generate(x=img_tensor.cpu().numpy())
            img_tensor = torch.tensor(adv_img_np).to("cuda" if torch.cuda.is_available() else "cpu")
    except Exception as e:
        print(f"Error in adversarial generation: {e}")
        return image

    adv_image_pil = postprocess_image(img_tensor, original_size)
    return adv_image_pil

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

def process_image(image, model_name, attack_types, eps_value, wm_text, password_img, password_wm):
    try:
        adv_image = generate_adversarial_image(image, model_name, attack_types, eps_value)
    except Exception as e:
        error_message = f"Error in adversarial generation: {str(e)}"
        return image, error_message, None, None, None

    try:
        watermarked_image = apply_watermark(adv_image, wm_text, int(password_img), int(password_wm))
    except Exception as e:
        error_message = f"Error in watermarking: {str(e)}"
        return image, adv_image, error_message, None, None

    try:
        extracted_wm_text = extract_watermark(watermarked_image, int(password_img), int(password_wm))
    except Exception as e:
        error_message = f"Error in watermark extraction: {str(e)}"
        return image, adv_image, watermarked_image, error_message, None

    output_path = tempfile.mktemp(suffix=".png")
    watermarked_image.save(output_path, format="PNG")
    return image, adv_image, watermarked_image, extracted_wm_text, output_path

def download_image_as_png(image_path):
    with open(image_path, "rb") as file:
        return file.read(), "image/png"

interface = gr.Interface(
    fn=process_image,
    inputs=[
        gr.Image(type="pil", label="이미지를 업로드하세요"),
        gr.Dropdown(choices=["ResNet50", "VGG16"], label="모델 선택"),
        gr.CheckboxGroup(choices=["FGSM", "C&W", "DeepFool", "AutoAttack", "PGD", "BIM", "STA", "MIM", "JSMA", "NewtonFool"], label="공격 유형 선택"),
        gr.Slider(0.001, 0.9, step=0.001, value=0.005, label="Epsilon 값 설정 (노이즈 강도)"),
        gr.Textbox(label="워터마크 텍스트 입력", value="텍스트 삽입"),
        gr.Number(label="이미지 비밀번호", value=0),
        gr.Number(label="워터마크 비밀번호", value=0)
    ],
    outputs=[
        gr.Image(type="numpy", label="원본 이미지"),
        gr.Image(type="numpy", label="적대적 이미지 생성 단계"),
        gr.Image(type="numpy", label="워터마크가 삽입된 최종 이미지"),
        gr.Textbox(label="추출된 워터마크 텍스트"),
        gr.File(label="PNG로 다운로드")
    ]
)

interface.launch(debug=True, share=True)
