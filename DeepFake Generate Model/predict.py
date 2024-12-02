import cog
import tempfile
from pathlib import Path
import argparse
import cv2
import torch
from PIL import Image
import torch.nn.functional as F
from torchvision import transforms
from models.models import create_model
from options.test_options import TestOptions
from util.reverse2original import reverse2wholeimage
from util.norm import SpecificNorm
from test_wholeimage_swapmulti import _totensor
from insightface_func.face_detect_crop_multi import Face_detect_crop as Face_detect_crop_multi
from insightface_func.face_detect_crop_single import Face_detect_crop as Face_detect_crop_single


class Predictor(cog.Predictor):
    """
    Face swapping 모델을 활용한 예측기 클래스.
    """

    def setup(self):
        """
        모델 준비 작업.
        Arcface 모델에 필요한 이미지 전처리 변환 정의.
        """
        self.transformer_Arcface = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    @cog.input("source", type=Path, help="source image")
    @cog.input("target", type=Path, help="target image")
    @cog.input("mode", type=str, options=['single', 'all'], default='all',
               help="swap a single face (the one with highest confidence by face detection) or all faces in the target image")
    def predict(self, source, target, mode='all'):
        """
        얼굴 스왑 예측 실행 함수.
        Args:
            source (Path): 소스 이미지 경로
            target (Path): 타겟 이미지 경로
            mode (str): 스왑 모드 ('single' 또는 'all')
        Returns:
            Path: 결과 이미지 경로
        """

        # 얼굴 탐지 및 크롭 객체 초기화
        app = Face_detect_crop_multi(name='antelope', root='./insightface_func/models')

        if mode == 'single':
            app = Face_detect_crop_single(name='antelope', root='./insightface_func/models')

        app.prepare(ctx_id=0, det_thresh=0.6, det_size=(640, 640))  # 탐지기 준비

        # 테스트 옵션 설정
        options = TestOptions()
        options.initialize()
        opt = options.parser.parse_args([
            "--Arc_path", 'arcface_model/arcface_checkpoint.tar',
            "--pic_a_path", str(source),
            "--pic_b_path", str(target),
            "--isTrain", False,
            "--no_simswaplogo"
        ])

        # GPU 설정
        str_ids = opt.gpu_ids.split(',')
        opt.gpu_ids = []
        for str_id in str_ids:
            id = int(str_id)
            if id >= 0:
                opt.gpu_ids.append(id)

        if len(opt.gpu_ids) > 0:
            torch.cuda.set_device(opt.gpu_ids[0])

        # 모델 생성 및 평가 모드로 설정
        torch.nn.Module.dump_patches = True
        model = create_model(opt)
        model.eval()

        crop_size = opt.crop_size
        spNorm = SpecificNorm()

        with torch.no_grad():
            # 소스 이미지 전처리 및 특징 추출
            pic_a = opt.pic_a_path
            img_a_whole = cv2.imread(pic_a)
            img_a_align_crop, _ = app.get(img_a_whole, crop_size)
            img_a_align_crop_pil = Image.fromarray(cv2.cvtColor(img_a_align_crop[0], cv2.COLOR_BGR2RGB))
            img_a = self.transformer_Arcface(img_a_align_crop_pil)
            img_id = img_a.view(-1, img_a.shape[0], img_a.shape[1], img_a.shape[2]).cuda()

            # Latent ID 생성
            img_id_downsample = F.interpolate(img_id, size=(112, 112))
            latend_id = model.netArc(img_id_downsample)
            latend_id = F.normalize(latend_id, p=2, dim=1)

            # 타겟 이미지 처리 및 얼굴 스왑
            pic_b = opt.pic_b_path
            img_b_whole = cv2.imread(pic_b)
            img_b_align_crop_list, b_mat_list = app.get(img_b_whole, crop_size)

            swap_result_list = []
            b_align_crop_tenor_list = []

            for b_align_crop in img_b_align_crop_list:
                b_align_crop_tenor = _totensor(cv2.cvtColor(b_align_crop, cv2.COLOR_BGR2RGB))[None, ...].cuda()
                swap_result = model(None, b_align_crop_tenor, latend_id, None, True)[0]
                swap_result_list.append(swap_result)
                b_align_crop_tenor_list.append(b_align_crop_tenor)

            # 결과 이미지를 원본 이미지에 합성
            net = None
            out_path = Path(tempfile.mkdtemp()) / "output.png"

            reverse2wholeimage(
                b_align_crop_tenor_list, swap_result_list, b_mat_list, crop_size,
                img_b_whole, None, str(out_path), opt.no_simswaplogo,
                pasring_model=net, use_mask=opt.use_mask, norm=spNorm
            )
            return out_path
