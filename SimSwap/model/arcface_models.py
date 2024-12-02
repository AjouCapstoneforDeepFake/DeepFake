import math
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import Parameter
from .config import device, num_classes  # 설정 파일에서 device와 num_classes 불러오기


# SEBlock 클래스: Squeeze-and-Excitation 블록 정의
class SEBlock(nn.Module):
    """
    SEBlock은 채널별로 중요한 특성을 학습하여 강조하는 역할을 수행.
    """
    def __init__(self, channel, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)  # 입력 텐서의 평균 풀링 수행
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction),  # 채널 축소
            nn.PReLU(),  # 활성화 함수
            nn.Linear(channel // reduction, channel),  # 채널 복원
            nn.Sigmoid()  # 중요도를 계산하는 활성화 함수
        )

    def forward(self, x):
        b, c, _, _ = x.size()  # 배치, 채널, 높이, 너비
        y = self.avg_pool(x).view(b, c)  # 평균 풀링 후 1D로 변환
        y = self.fc(y).view(b, c, 1, 1)  # 중요도 계산 후 차원 복원
        return x * y  # 입력 텐서와 중요도를 곱함


# IRBlock 클래스: Residual Block 구현
class IRBlock(nn.Module):
    """
    IRBlock은 ResNet에서 사용되는 잔차 블록을 확장하여 SEBlock을 추가.
    """
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, use_se=True):
        super(IRBlock, self).__init__()
        self.bn0 = nn.BatchNorm2d(inplanes)  # 입력에 배치 정규화 적용
        self.conv1 = conv3x3(inplanes, inplanes)  # 첫 번째 3x3 컨볼루션
        self.bn1 = nn.BatchNorm2d(inplanes)  # 첫 번째 배치 정규화
        self.prelu = nn.PReLU()  # 활성화 함수
        self.conv2 = conv3x3(inplanes, planes, stride)  # 두 번째 3x3 컨볼루션
        self.bn2 = nn.BatchNorm2d(planes)  # 두 번째 배치 정규화
        self.downsample = downsample  # 다운샘플링 모듈
        self.stride = stride
        self.use_se = use_se
        if self.use_se:
            self.se = SEBlock(planes)  # SEBlock 추가

    def forward(self, x):
        residual = x  # 입력 잔차 저장
        out = self.bn0(x)  # 첫 번째 배치 정규화
        out = self.conv1(out)  # 첫 번째 컨볼루션
        out = self.bn1(out)  # 배치 정규화
        out = self.prelu(out)  # 활성화

        out = self.conv2(out)  # 두 번째 컨볼루션
        out = self.bn2(out)  # 배치 정규화
        if self.use_se:
            out = self.se(out)  # SEBlock 적용

        if self.downsample is not None:
            residual = self.downsample(x)  # 다운샘플링 수행

        out += residual  # 입력과 출력 더하기
        out = self.prelu(out)  # 활성화

        return out


# ResNet 클래스: Residual Network 구현
class ResNet(nn.Module):
    """
    ResNet은 IRBlock과 SEBlock을 사용하여 심층 네트워크 구조를 생성.
    """
    def __init__(self, block, layers, use_se=True):
        self.inplanes = 64  # 초기 입력 채널 수
        self.use_se = use_se
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, bias=False)  # 첫 번째 컨볼루션
        self.bn1 = nn.BatchNorm2d(64)  # 배치 정규화
        self.prelu = nn.PReLU()  # 활성화
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)  # 최대 풀링
        self.layer1 = self._make_layer(block, 64, layers[0])  # 첫 번째 레이어
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)  # 두 번째 레이어
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)  # 세 번째 레이어
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)  # 네 번째 레이어
        self.bn2 = nn.BatchNorm2d(512)  # 배치 정규화
        self.dropout = nn.Dropout()  # 드롭아웃
        self.fc = nn.Linear(512 * 7 * 7, 512)  # 완전 연결 계층
        self.bn3 = nn.BatchNorm1d(512)  # 배치 정규화

        # 가중치 초기화
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight)
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        """
        ResNet의 계층 생성
        block: IRBlock
        planes: 출력 채널 수
        blocks: 블록 수
        stride: 스트라이드
        """
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, use_se=self.use_se))
        self.inplanes = planes
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, use_se=self.use_se))

        return nn.Sequential(*layers)

    def forward(self, x):
        """
        ResNet의 순전파
        """
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.prelu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.bn2(x)
        x = self.dropout(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        x = self.bn3(x)

        return x


# ArcMarginModel 클래스: ArcFace 손실을 구현한 모델
class ArcMarginModel(nn.Module):
    """
    ArcFace 손실을 사용하여 얼굴 인식 성능을 향상.
    """
    def __init__(self, args):
        super(ArcMarginModel, self).__init__()

        self.weight = Parameter(torch.FloatTensor(num_classes, args.emb_size))  # 가중치 초기화
        nn.init.xavier_uniform_(self.weight)

        self.easy_margin = args.easy_margin
        self.m = args.margin_m  # ArcFace의 각도 마진
        self.s = args.margin_s  # 스케일링 인자

        self.cos_m = math.cos(self.m)  # cos(m)
        self.sin_m = math.sin(self.m)  # sin(m)
        self.th = math.cos(math.pi - self.m)  # 임계값
        self.mm = math.sin(math.pi - self.m) * self.m  # 보정 값

    def forward(self, input, label):
        """
        ArcFace 손실 계산
        input: 입력 특징 벡터
        label: 클래스 레이블
        """
        x = F.normalize(input)  # 입력 정규화
        W = F.normalize(self.weight)  # 가중치 정규화
        cosine = F.linear(x, W)  # cos(theta)
        sine = torch.sqrt(1.0 - torch.pow(cosine, 2))  # sin(theta)
        phi = cosine * self.cos_m - sine * self.sin_m  # cos(theta + m)
        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)
        one_hot = torch.zeros(cosine.size(), device=device)
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)  # 레이블에 해당하는 위치에 1 할당
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)  # ArcFace 손실 적용
        output *= self.s  # 스케일링
        return output
