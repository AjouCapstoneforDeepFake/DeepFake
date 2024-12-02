import os
import torch

# PyTorch에서 사용할 장치를 설정 (CUDA가 가능하면 GPU, 그렇지 않으면 CPU 사용)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 모델 매개변수
image_w = 112  # 입력 이미지의 너비
image_h = 112  # 입력 이미지의 높이
channel = 3  # 이미지의 채널 수 (RGB이므로 3)
emb_size = 512  # 임베딩 벡터 크기

# 학습 매개변수
num_workers = 1  # 데이터 로딩에 사용할 워커 수 (현재 h5py와의 호환성 문제로 1만 지원)
grad_clip = 5.  # 그래디언트를 절대값으로 클리핑
print_freq = 100  # 학습/검증 통계를 출력하는 빈도 (배치 단위)
checkpoint = None  # 체크포인트 경로, 체크포인트가 없으면 None

# 데이터 매개변수
num_classes = 93431  # 클래스 수 (얼굴 ID)
num_samples = 5179510  # 총 샘플 수 (얼굴 이미지 수)
DATA_DIR = 'data'  # 데이터 디렉토리 경로

# MS1M 데이터셋 폴더 및 파일 경로
faces_ms1m_folder = 'data/ms1m-retinaface-t1'  # MS1M 데이터셋 폴더 경로
path_imgidx = os.path.join(faces_ms1m_folder, 'train.idx')  # 데이터셋 인덱스 파일 경로
path_imgrec = os.path.join(faces_ms1m_folder, 'train.rec')  # 데이터셋 이미지 파일 경로

IMG_DIR = 'data/images'  # 이미지 저장 디렉토리 경로
pickle_file = 'data/faces_ms1m_112x112.pickle'  # 데이터셋 피클 파일 경로
