# 필요한 라이브러리 import
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as modelzoo

# ResNet-18 모델의 사전 학습 가중치 URL
resnet18_url = 'https://download.pytorch.org/models/resnet18-5c106cde.pth'


# 3x3 컨볼루션 레이어 생성 함수
def conv3x3(in_planes, out_planes, stride=1):
    """3x3 컨볼루션 레이어 (패딩 포함)"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


# BasicBlock 정의: ResNet의 기본 구성 블록
class BasicBlock(nn.Module):
    def __init__(self, in_chan, out_chan, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(in_chan, out_chan, stride)  # 첫 번째 컨볼루션
        self.bn1 = nn.BatchNorm2d(out_chan)  # 배치 정규화
        self.conv2 = conv3x3(out_chan, out_chan)  # 두 번째 컨볼루션
        self.bn2 = nn.BatchNorm2d(out_chan)  # 배치 정규화
        self.relu = nn.ReLU(inplace=True)  # 활성화 함수
        self.downsample = None  # 다운샘플링 레이어
        if in_chan != out_chan or stride != 1:  # 입력과 출력 채널이 다르거나 stride가 1이 아닐 때
            self.downsample = nn.Sequential(
                nn.Conv2d(in_chan, out_chan,
                          kernel_size=1, stride=stride, bias=False),  # 1x1 컨볼루션
                nn.BatchNorm2d(out_chan),  # 배치 정규화
            )

    def forward(self, x):
        residual = self.conv1(x)
        residual = F.relu(self.bn1(residual))  # 첫 번째 컨볼루션과 활성화
        residual = self.conv2(residual)
        residual = self.bn2(residual)  # 두 번째 컨볼루션

        shortcut = x
        if self.downsample is not None:  # 다운샘플링이 필요한 경우
            shortcut = self.downsample(x)

        out = shortcut + residual  # 스킵 연결
        out = self.relu(out)  # 활성화 함수 적용
        return out


# BasicBlock 계층 생성 함수
def create_layer_basic(in_chan, out_chan, bnum, stride=1):
    """BasicBlock 레이어 생성"""
    layers = [BasicBlock(in_chan, out_chan, stride=stride)]
    for i in range(bnum-1):  # 추가 블록 생성
        layers.append(BasicBlock(out_chan, out_chan, stride=1))
    return nn.Sequential(*layers)


# ResNet-18 클래스 정의
class Resnet18(nn.Module):
    def __init__(self):
        super(Resnet18, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)  # 초기 컨볼루션 레이어
        self.bn1 = nn.BatchNorm2d(64)  # 배치 정규화
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)  # 최대 풀링 레이어
        self.layer1 = create_layer_basic(64, 64, bnum=2, stride=1)  # 첫 번째 계층
        self.layer2 = create_layer_basic(64, 128, bnum=2, stride=2)  # 두 번째 계층
        self.layer3 = create_layer_basic(128, 256, bnum=2, stride=2)  # 세 번째 계층
        self.layer4 = create_layer_basic(256, 512, bnum=2, stride=2)  # 네 번째 계층
        self.init_weight()  # 가중치 초기화

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(self.bn1(x))  # 초기 컨볼루션과 활성화
        x = self.maxpool(x)  # 최대 풀링

        x = self.layer1(x)
        feat8 = self.layer2(x)  # 1/8 크기 특징 맵
        feat16 = self.layer3(feat8)  # 1/16 크기 특징 맵
        feat32 = self.layer4(feat16)  # 1/32 크기 특징 맵
        return feat8, feat16, feat32  # 세 가지 특징 맵 반환

    def init_weight(self):
        """사전 학습된 가중치 초기화"""
        state_dict = modelzoo.load_url(resnet18_url)  # ResNet-18 가중치 다운로드
        self_state_dict = self.state_dict()
        for k, v in state_dict.items():
            if 'fc' in k:  # Fully Connected 레이어 제외
                continue
            self_state_dict.update({k: v})
        self.load_state_dict(self_state_dict)

    def get_params(self):
        """모델의 가중치와 편향 파라미터 반환"""
        wd_params, nowd_params = [], []
        for name, module in self.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):  # Conv2d 또는 Linear 파라미터
                wd_params.append(module.weight)
                if not module.bias is None:
                    nowd_params.append(module.bias)
            elif isinstance(module, nn.BatchNorm2d):  # 배치 정규화 파라미터
                nowd_params += list(module.parameters())
        return wd_params, nowd_params


if __name__ == "__main__":
    # ResNet-18 네트워크 초기화
    net = Resnet18()
    x = torch.randn(16, 3, 224, 224)  # 예제 입력 텐서
    out = net(x)
    # 출력 특징 맵 크기 출력
    print(out[0].size())  # 1/8 크기
    print(out[1].size())  # 1/16 크기
    print(out[2].size())  # 1/32 크기
    net.get_params()  # 파라미터 가져오기
