# -*- coding: utf-8 -*-
"""Final Deepfake Image detection

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/18iq_Zkj70qxSrLVyN_YDUMDvn4PjEBX4
"""

# 구글 드라이브 마운트
from google.colab import drive
drive.mount('/content/drive')

!pip install opencv-python mtcnn
!pip install facenet-pytorch

import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from torchvision import models
import torch.nn as nn
import torch.optim as optim
from torch.utils.data.dataset import random_split
from torch.utils.data import random_split
from PIL import Image
import os
import cv2
from torchvision import datasets
import glob
import random
import time  
from google.colab.patches import cv2_imshow
import numpy as np
from facenet_pytorch import MTCNN

# 클래스 매핑
idx_to_cls = {0: 'fake', 1: 'real'}

# GPU 설정
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
mtcnn = MTCNN(keep_all=True, device=device)  


# 비가시적 워터마크 추출 함수
def extract_invisible_watermark(image_path, output_path):
    """
    이미지에서 비가시적 워터마크를 추출하여 템플릿 생성.
    """
    try:
        watermarked_img = Image.open(image_path).convert("RGB")
    except FileNotFoundError:
        raise FileNotFoundError(f"Image file not found: {image_path}")

    # 워터마크 추출 로직
    width, height = watermarked_img.size
    extracted_template = Image.new("RGB", (width, height), (255, 255, 255))  
    for row in range(height):
        for col in range(width):
            r, g, b = watermarked_img.getpixel((col, row))

            # 삽입 로직 기반 추출: LSB를 확인해 템플릿 생성
            if r % 2 != 0:  # LSB가 1인 경우 텍스트 있음 (검정색)
                extracted_template.putpixel((col, row), (0, 0, 0))
            else:  # LSB가 0인 경우 텍스트 없음 (흰색)
                extracted_template.putpixel((col, row), (255, 255, 255))

    # 추출된 템플릿 저장
    extracted_template.save(output_path)
    print(f"Watermark template saved to {output_path}")
    return output_path


# 비가시적 워터마크 탐지 함수
def check_extracted_watermark(output_path):
    """
    추출된 템플릿에서 워터마크 존재 여부 탐지.
    """
    try:
        extracted_img = Image.open(output_path).convert("RGB")
        width, height = extracted_img.size

        # 워터마크 탐지 로직
        for row in range(height):
            for col in range(width):
                r, g, b = extracted_img.getpixel((col, row))
                if r == 0 and g == 0 and b == 0:  # 검정색 픽셀이 있으면 워터마크가 있음
                    print("Invisible watermark detected.")
                    return True

        print("No invisible watermark detected.")
        return False  # 검정색 픽셀이 없으면 워터마크가 없음
    except FileNotFoundError:
        raise FileNotFoundError(f"Extracted watermark file not found: {output_path}")


# 모델 로드 함수
def load_model(model_path, num_classes):
    model = models.efficientnet_v2_s(pretrained=False)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model


# 얼굴 탐지 및 크롭 함수
def detect_and_crop_face(image_path, normalize=True):
    img = Image.open(image_path).convert("RGB")
    faces = mtcnn(img)

    if faces is not None:
        face = faces[0] if faces.ndim == 4 else faces
        transform_steps = [transforms.Resize((224, 224))]
        if normalize:
            transform_steps += [
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        else:
            transform_steps += [transforms.ToTensor()]

        transform = transforms.Compose(transform_steps)

        face_np = face.permute(1, 2, 0).cpu().numpy()
        face_pil = Image.fromarray((face_np * 255).astype(np.uint8))
        face_tensor = transform(face_pil).unsqueeze(0)
        return face_tensor, face_pil
    else:
        print("얼굴을 찾지 못했습니다.")
        return None, None


# 이미지 탐지 함수
def predict_image(image_path, model):
    face_tensor, _ = detect_and_crop_face(image_path, normalize=True)  # 정규화된 이미지 사용
    if face_tensor is None:
        return None

    face_tensor = face_tensor.to(device)
    with torch.no_grad():
        outputs = model(face_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        max_prob, predicted = torch.max(probabilities, 1)

    predicted_label = idx_to_cls[predicted.item()]
    print(predicted_label)
    return predicted_label


# 통합 로직 실행
def main(image_path, model_path, watermark_output_path):
    # 모델 로드
    model = load_model(model_path, num_classes=2)

    # Step 1: 이미지 탐지 수행
    initial_prediction = predict_image(image_path, model)
    if initial_prediction == 'fake':
        print(f"Predicted class -> fake")
        # OpenCV 이미지 출력
        test_img = cv2.imread(image_path)
        test_img_resized = cv2.resize(test_img, (400, 400))
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(test_img_resized, "Predicted: fake", (50, 50), font, 1, (0, 0, 255), 2)
        cv2_imshow(test_img_resized)  
        return  # 탐지 결과가 fake이면 종료

    # Step 2: 이미지 탐지 결과가 'real'인 경우 비가시적 워터마크 탐지 수행
    _, cropped_face = detect_and_crop_face(image_path, normalize=False)  # 정규화되지 않은 이미지 사용
    cropped_face.save(watermark_output_path)
    watermark_detected = check_extracted_watermark(watermark_output_path)

    if watermark_detected:
        print(f"Predicted class -> fake (Invisible Watermark Detected)")
        test_img = cv2.imread(image_path)
        test_img_resized = cv2.resize(test_img, (400, 400))
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(test_img_resized, "Predicted: fake (Watermark)", (50, 50), font, 1, (0, 0, 255), 2)
        cv2_imshow(test_img_resized)  
    else:
        print(f"Predicted class -> real")
        test_img = cv2.imread(image_path)
        test_img_resized = cv2.resize(test_img, (400, 400))
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(test_img_resized, "Predicted: real", (50, 50), font, 1, (0, 255, 0), 2)
        cv2_imshow(test_img_resized)  


#경로 설정
image_path = '#탐지하고 싶은 이미지 경로 설정'
model_path = '#최고 정확도 모델 가중치 경로 설정'
watermark_output_path = '/content/drive/MyDrive/temp_extracted_watermark.png'

# 실행
main(image_path, model_path, watermark_output_path)


