import cv2
import torch
import fractions
import numpy as np
from PIL import Image
import torch.nn.functional as F
from torchvision import transforms
from models.models import create_model
from options.test_options import TestOptions
from insightface_func.face_detect_crop_single import Face_detect_crop
from util.videoswap import video_swap
import os

def lcm(a, b):
    """
    최소공배수 계산 함수.
    """
    return abs(a * b) // fractions.gcd(a, b) if a and b else 0

# Arcface 모델 입력을 위한 이미지 전처리
transformer_Arcface = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# 일반적인 이미지 전처리
transformer = transforms.Compose([
    transforms.ToTensor(),
])

if __name__ == '__main__':
    # 옵션 파싱
    opt = TestOptions().parse()
    crop_size = opt.crop_size

    # 모델 초기화 및 설정
    torch.nn.Module.dump_patches = True
    if crop_size == 512:
        opt.which_epoch = 550000
        opt.name = '512'
        mode = 'ffhq'
    else:
        mode = 'None'

    model = create_model(opt)
    model.eval()

    # 얼굴 탐지 및 크롭 객체 초기화
    app = Face_detect_crop(name='antelope', root='./insightface_func/models')
    app.prepare(ctx_id=0, det_thresh=0.6, det_size=(640, 640), mode=mode)

    with torch.no_grad():
        # 소스 이미지 로드 및 특징 추출
        pic_a = opt.pic_a_path
        img_a_whole = cv2.imread(pic_a)
        img_a_align_crop, _ = app.get(img_a_whole, crop_size)
        img_a_align_crop_pil = Image.fromarray(cv2.cvtColor(img_a_align_crop[0], cv2.COLOR_BGR2RGB))
        img_a = transformer_Arcface(img_a_align_crop_pil)
        img_id = img_a.view(-1, img_a.shape[0], img_a.shape[1], img_a.shape[2]).cuda()

        # Latent ID 생성 및 정규화
        img_id_downsample = F.interpolate(img_id, size=(112, 112))
        latend_id = model.netArc(img_id_downsample)
        latend_id = F.normalize(latend_id, p=2, dim=1)

        # 비디오 얼굴 스왑 실행
        video_swap(
            opt.video_path,
            latend_id,
            model,
            app,
            opt.output_path,
            temp_results_dir=opt.temp_path,
            no_simswaplogo=opt.no_simswaplogo,
            use_mask=opt.use_mask,
            crop_size=crop_size
        )
