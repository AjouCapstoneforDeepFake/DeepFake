import cv2
import torch
import fractions
import numpy as np
from PIL import Image
import torch.nn.functional as F
from torchvision import transforms
from models.models import create_model
from options.test_options import TestOptions


def lcm(a, b):
    """
    최소공배수 계산 함수.
    """
    return abs(a * b) / fractions.gcd(a, b) if a and b else 0


# 이미지 전처리 변환 정의
transformer = transforms.Compose([
    transforms.ToTensor(),
])

transformer_Arcface = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# 이미지 역변환 (복원) 정의
detransformer = transforms.Compose([
    transforms.Normalize([0, 0, 0], [1/0.229, 1/0.224, 1/0.225]),
    transforms.Normalize([-0.485, -0.456, -0.406], [1, 1, 1])
])

if __name__ == '__main__':
    # 옵션 파싱
    opt = TestOptions().parse()

    # 모델 초기화
    torch.nn.Module.dump_patches = True
    model = create_model(opt)
    model.eval()

    with torch.no_grad():
        # 소스 이미지 로드 및 전처리
        pic_a = opt.pic_a_path
        img_a = Image.open(pic_a).convert('RGB')
        img_a = transformer_Arcface(img_a)
        img_id = img_a.view(-1, img_a.shape[0], img_a.shape[1], img_a.shape[2])

        # 타겟 이미지 로드 및 전처리
        pic_b = opt.pic_b_path
        img_b = Image.open(pic_b).convert('RGB')
        img_b = transformer(img_b)
        img_att = img_b.view(-1, img_b.shape[0], img_b.shape[1], img_b.shape[2])

        # 텐서를 GPU로 이동
        img_id = img_id.cuda()
        img_att = img_att.cuda()

        # Latent ID 생성
        img_id_downsample = F.interpolate(img_id, size=(112, 112))
        latend_id = model.netArc(img_id_downsample)
        latend_id = latend_id.detach().to('cpu')
        latend_id = latend_id / np.linalg.norm(latend_id, axis=1, keepdims=True)
        latend_id = latend_id.to('cuda')

        ############## Forward Pass ######################
        img_fake = model(img_id, img_att, latend_id, latend_id, True)

        # 결과 이미지 병합
        for i in range(img_id.shape[0]):
            if i == 0:
                row1 = img_id[i]
                row2 = img_att[i]
                row3 = img_fake[i]
            else:
                row1 = torch.cat([row1, img_id[i]], dim=2)
                row2 = torch.cat([row2, img_att[i]], dim=2)
                row3 = torch.cat([row3, img_fake[i]], dim=2)

        # 최종 출력 생성
        full = row3.detach()
        full = full.permute(1, 2, 0)
        output = full.to('cpu')
        output = np.array(output)
        output = output[..., ::-1]  # RGB -> BGR 변환

        # 이미지 스케일 복원 및 저장
        output = output * 255
        cv2.imwrite(opt.output_path + 'result.jpg', output)
