from __future__ import division
import collections
import numpy as np
import glob
import os
import os.path as osp
import cv2
from insightface.model_zoo import model_zoo
from insightface_func.utils import face_align_ffhqandnewarc as face_align

# __all__은 모듈에서 외부로 노출할 객체를 정의
__all__ = ['Face_detect_crop', 'Face']

# Face: 얼굴 정보를 담는 NamedTuple
# bbox: 바운딩 박스 정보
# kps: 얼굴 랜드마크 좌표
# det_score: 얼굴 탐지 점수
# embedding: 얼굴 특징 임베딩
# gender: 성별 정보
# age: 나이 정보
# embedding_norm: 임베딩 정규화 값
# normed_embedding: 정규화된 임베딩
# landmark: 얼굴 랜드마크 정보
Face = collections.namedtuple('Face', [
    'bbox', 'kps', 'det_score', 'embedding', 'gender', 'age',
    'embedding_norm', 'normed_embedding',
    'landmark'
])

# Face 객체의 기본값 설정
Face.__new__.__defaults__ = (None, ) * len(Face._fields)

# Face_detect_crop 클래스: 얼굴 탐지 및 정렬
class Face_detect_crop:
    def __init__(self, name, root='~/.insightface_func/models'):
        """
        모델 초기화 및 탐지 모델 로드
        name: 모델 이름
        root: 모델 파일이 저장된 기본 경로
        """
        self.models = {}  # 모델을 저장할 딕셔너리
        root = os.path.expanduser(root)  # 사용자 경로 확장
        onnx_files = glob.glob(osp.join(root, name, '*.onnx'))  # .onnx 모델 파일 검색
        onnx_files = sorted(onnx_files)  # 모델 파일 정렬
        for onnx_file in onnx_files:
            if onnx_file.find('_selfgen_') > 0:
                # '_selfgen_'가 포함된 파일은 무시
                continue
            model = model_zoo.get_model(onnx_file)  # 모델 로드
            if model.taskname not in self.models:
                print('find model:', onnx_file, model.taskname)  # 모델 로드 확인 출력
                self.models[model.taskname] = model  # 모델 추가
            else:
                print('duplicated model task type, ignore:', onnx_file, model.taskname)  # 중복된 모델 무시
                del model
        # 반드시 'detection' 모델이 포함되어야 함
        assert 'detection' in self.models
        self.det_model = self.models['detection']

    # 모델 준비
    def prepare(self, ctx_id, det_thresh=0.5, det_size=(640, 640), mode='None'):
        """
        얼굴 탐지 모델 준비
        ctx_id: 실행할 컨텍스트 ID (CPU/GPU)
        det_thresh: 탐지 임계값
        det_size: 탐지 크기
        mode: 얼굴 정렬 모드
        """
        self.det_thresh = det_thresh  # 탐지 임계값 설정
        self.mode = mode  # 얼굴 정렬 모드 설정
        assert det_size is not None
        print('set det-size:', det_size)  # 탐지 크기 출력
        self.det_size = det_size
        for taskname, model in self.models.items():
            if taskname == 'detection':
                model.prepare(ctx_id, input_size=det_size)  # 탐지 모델 준비
            else:
                model.prepare(ctx_id)  # 다른 모델 준비

    # 얼굴 탐지 및 정렬
    def get(self, img, crop_size, max_num=0):
        """
        얼굴 탐지 및 정렬
        img: 입력 이미지
        crop_size: 잘라낼 크기
        max_num: 탐지할 최대 얼굴 수
        """
        # 얼굴 탐지 수행
        bboxes, kpss = self.det_model.detect(img,
                                             threshold=self.det_thresh,
                                             max_num=max_num,
                                             metric='default')
        if bboxes.shape[0] == 0:  # 얼굴이 탐지되지 않으면 None 반환
            return None
        align_img_list = []  # 정렬된 이미지 리스트
        M_list = []  # 정렬 행렬 리스트
        for i in range(bboxes.shape[0]):  # 탐지된 얼굴 수만큼 반복
            kps = None
            if kpss is not None:
                kps = kpss[i]  # 랜드마크 좌표 가져오기
            M, _ = face_align.estimate_norm(kps, crop_size, mode=self.mode)  # 얼굴 정렬 행렬 계산
            align_img = cv2.warpAffine(img, M, (crop_size, crop_size), borderValue=0.0)  # 이미지 정렬
            align_img_list.append(align_img)  # 정렬된 이미지 추가
            M_list.append(M)  # 정렬 행렬 추가
        
        return align_img_list, M_list  # 정렬된 이미지와 정렬 행렬 반환
