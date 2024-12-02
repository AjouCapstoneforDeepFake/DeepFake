import os
import glob
import torch
import random
from PIL import Image
from torch.utils import data
from torchvision import transforms as T

# data_prefetcher 클래스: 데이터 로딩을 최적화하기 위해 CUDA 스트림을 활용
class data_prefetcher():
    def __init__(self, loader):
        self.loader = loader  # 데이터 로더 초기화
        self.dataiter = iter(loader)  # 데이터 로더 반복자 생성
        self.stream = torch.cuda.Stream()  # CUDA 스트림 초기화
        self.mean = torch.tensor([0.485, 0.456, 0.406]).cuda().view(1,3,1,1)  # 데이터 정규화의 평균값
        self.std = torch.tensor([0.229, 0.224, 0.225]).cuda().view(1,3,1,1)  # 데이터 정규화의 표준편차
        self.num_images = len(loader)  # 총 이미지 개수
        self.preload()  # 데이터 미리 로드

    # 데이터를 미리 로드하는 메서드
    def preload(self):
        try:
            self.src_image1, self.src_image2 = next(self.dataiter)  # 다음 배치 데이터를 로드
        except StopIteration:  # 데이터가 끝났다면 다시 반복자 초기화
            self.dataiter = iter(self.loader)
            self.src_image1, self.src_image2 = next(self.dataiter)
            
        with torch.cuda.stream(self.stream):  # CUDA 스트림에서 비동기 로드
            self.src_image1 = self.src_image1.cuda(non_blocking=True)  # GPU로 데이터 전송
            self.src_image1 = self.src_image1.sub_(self.mean).div_(self.std)  # 정규화
            self.src_image2 = self.src_image2.cuda(non_blocking=True)  # GPU로 데이터 전송
            self.src_image2 = self.src_image2.sub_(self.mean).div_(self.std)  # 정규화

    # 다음 데이터를 반환
    def next(self):
        torch.cuda.current_stream().wait_stream(self.stream)  # CUDA 스트림 대기
        src_image1 = self.src_image1  # 미리 로드된 이미지 반환
        src_image2 = self.src_image2
        self.preload()  # 다음 배치를 미리 로드
        return src_image1, src_image2
    
    def __len__(self):
        """총 이미지 개수 반환"""
        return self.num_images


# SwappingDataset 클래스: 데이터를 불러오고 전처리
class SwappingDataset(data.Dataset):
    """Artworks 데이터셋 및 content 데이터셋을 위한 데이터셋 클래스"""

    def __init__(self,
                    image_dir,  # 데이터 디렉토리 경로
                    img_transform,  # 이미지 변환(transform) 파이프라인
                    subffix='jpg',  # 파일 확장자
                    random_seed=1234):  # 랜덤 시드 설정
        """Swapping 데이터셋 초기화 및 전처리"""
        self.image_dir = image_dir
        self.img_transform = img_transform
        self.subffix = subffix
        self.dataset = []  # 데이터셋 리스트 초기화
        self.random_seed = random_seed
        self.preprocess()  # 데이터셋 전처리
        self.num_images = len(self.dataset)  # 총 이미지 개수

    # 데이터셋 전처리 함수
    def preprocess(self):
        """Swapping 데이터셋 전처리"""
        print("processing Swapping dataset images...")

        temp_path = os.path.join(self.image_dir,'*/')  # 데이터 디렉토리 경로
        pathes = glob.glob(temp_path)  # 하위 디렉토리 경로 수집
        self.dataset = []
        for dir_item in pathes:
            join_path = glob.glob(os.path.join(dir_item,'*.jpg'))  # 하위 디렉토리의 jpg 파일 수집
            print("processing %s"%dir_item,end='\r')  # 진행 상황 출력
            temp_list = []
            for item in join_path:
                temp_list.append(item)  # 파일 경로 추가
            self.dataset.append(temp_list)  # 디렉토리별 파일 리스트 추가
        random.seed(self.random_seed)
        random.shuffle(self.dataset)  # 데이터셋 셔플
        print('Finished preprocessing the Swapping dataset, total dirs number: %d...'%len(self.dataset))
             
    def __getitem__(self, index):
        """두 개의 src 도메인 이미지를 반환"""
        dir_tmp1 = self.dataset[index]
        dir_tmp1_len = len(dir_tmp1)

        filename1 = dir_tmp1[random.randint(0,dir_tmp1_len-1)]  # 랜덤 파일 선택
        filename2 = dir_tmp1[random.randint(0,dir_tmp1_len-1)]
        image1 = self.img_transform(Image.open(filename1))  # 이미지 변환 적용
        image2 = self.img_transform(Image.open(filename2))
        return image1, image2
    
    def __len__(self):
        """총 데이터 개수 반환"""
        return self.num_images


# 데이터 로더 생성 함수
def GetLoader(  dataset_roots,
                batch_size=16,
                dataloader_workers=8,
                random_seed = 1234):
    """데이터 로더 생성 및 반환"""
        
    num_workers = dataloader_workers
    data_root = dataset_roots
    random_seed = random_seed
    
    c_transforms = []
    c_transforms.append(T.ToTensor())  # 이미지를 텐서로 변환
    c_transforms = T.Compose(c_transforms)  # 변환 파이프라인 생성

    content_dataset = SwappingDataset(
                            data_root, 
                            c_transforms,
                            "jpg",
                            random_seed)  # 데이터셋 초기화
    content_data_loader = data.DataLoader(dataset=content_dataset,
                                          batch_size=batch_size,
                                          drop_last=True,
                                          shuffle=True,
                                          num_workers=num_workers,
                                          pin_memory=True)  # 데이터 로더 생성
    prefetcher = data_prefetcher(content_data_loader)  # Prefetcher로 감싸기
    return prefetcher

# 텐서를 디노멀라이즈하는 함수
def denorm(x):
    out = (x + 1) / 2  # 데이터 정규화 해제
    return out.clamp_(0, 1)  # 값 범위 제한
