---
title: Adversarial Attack Pgd
emoji: 🐨
colorFrom: red
colorTo: green
sdk: gradio
sdk_version: 4.44.1
app_file: app.py
pinned: false
license: apache-2.0
---

Check out the configuration reference at https://huggingface.co/docs/hub/spaces-config-reference

# Adversarial Image Generation and Watermarking Service

## 📜 프로젝트 소개
이 프로젝트는 딥러닝 모델(ResNet50, VGG16)을 활용해 적대적 이미지를 생성하고, 워터마크를 삽입 및 추출할 수 있는 **Gradio 웹 애플리케이션**입니다.

### 주요 기능:
1. **적대적 공격 생성**: 다양한 적대적 공격 기법(FGSM, PGD 등)을 통해 이미지를 변환.
2. **워터마크 삽입 및 추출**: 이미지에 워터마크를 삽입하고 이를 다시 추출.
3. **실시간 인터페이스**: Gradio를 통해 직관적인 UI 제공.

---

## 📂 프로젝트 구조
. ├── app.py # 메인 애플리케이션 파일 
  ├── requirements.txt # 필요한 Python 패키지 목록 
  └── README.md # 프로젝트 설명 파일

  
---

## ⚙️ 설치 및 실행 방법

# 1. 필요 조건
Python 3.8 이상
CUDA 지원 GPU (옵션, 가속화 목적)

# 2. 환경 설정
# 가상환경 생성 (권장)
python -m venv venv
source venv/bin/activate   # macOS/Linux
venv\Scripts\activate      # Windows

# 의존성 설치
pip install -r requirements.txt

# 3. 추가 파일 다운로드
# Dlib의 얼굴 랜드마크 탐지를 위해 shape_predictor_68_face_landmarks.dat 파일이 필요합니다.
# 아래 링크에서 다운로드한 후, app.py와 같은 디렉토리에 저장하세요:
# http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2

# 4. 앱 실행
python app.py

# 기본적으로 애플리케이션은 http://127.0.0.1:7860/ 에서 실행됩니다.


## 🖥️ 사용 방법
# 입력값:
- 이미지 업로드: 변환할 이미지를 업로드.
- 모델 선택: ResNet50 또는 VGG16.
- 공격 유형 선택: PGD, FGSM 등 다양한 적대적 공격 기법 중 선택.
- Epsilon 값: 적대적 노이즈 강도를 설정.
- 워터마크 텍스트: 삽입할 워터마크 텍스트.
- 비밀번호: 워터마크 삽입 및 추출 시 사용할 비밀번호.

# 출력값:
- 원본 이미지
- 적대적 이미지
- 워터마크가 삽입된 최종 이미지
- 추출된 워터마크 텍스트
- 최종 이미지 다운로드 링크

## 🧩 주요 라이브러리
- Torch & Torchvision: 모델 학습 및 적대적 공격 생성.
- ART (Adversarial Robustness Toolbox): 적대적 공격 기법 구현.
- Gradio: 직관적인 웹 애플리케이션 UI.
- Pillow: 이미지 처리.
- Blind Watermark: 워터마크 삽입 및 추출.


## 📊 주요 기능 설명
# 1. 적대적 이미지 생성
제공된 이미지에 다양한 적대적 공격(FGSM, PGD 등)을 적용하여 이미지의 시각적 특성을 유지하면서도 딥러닝 모델을 속이는 이미지를 생성.

# 2. 워터마킹
Blind Watermark 라이브러리를 사용하여 이미지에 워터마크를 삽입.
삽입된 워터마크는 비밀번호 기반으로 암호화되어, 추출 시 동일한 비밀번호가 필요.

## 🤖 API 구조 (Gradio)
# process_image
입력: 이미지, 모델 이름, 공격 유형, epsilon 값, 워터마크 텍스트, 비밀번호.
출력: 변환된 이미지들 및 워터마크 텍스트.

## 💻 테스트
# 기본 실행 테스트:
python app.py

# API 테스트 (cURL):
curl -X POST http://127.0.0.1:7860/api/predict \
     -H "Content-Type: application/json" \
     -d '{
         "data": [
             "/path/to/image.png", 
             "ResNet50", 
             ["FGSM", "PGD"], 
             0.01,
             "Watermark Text",
             123,
             456
         ]
     }'

## 📜 라이선스
이 프로젝트는 MIT 라이선스를 따릅니다.
