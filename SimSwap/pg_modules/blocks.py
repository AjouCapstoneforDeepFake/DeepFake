import functools
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm


### Single Layers (단일 레이어)


def conv2d(*args, **kwargs):
    """Spectral Normalization이 적용된 2D 컨볼루션 레이어 생성"""
    return spectral_norm(nn.Conv2d(*args, **kwargs))


def convTranspose2d(*args, **kwargs):
    """Spectral Normalization이 적용된 Transposed 2D 컨볼루션 레이어 생성"""
    return spectral_norm(nn.ConvTranspose2d(*args, **kwargs))


def embedding(*args, **kwargs):
    """Spectral Normalization이 적용된 Embedding 레이어 생성"""
    return spectral_norm(nn.Embedding(*args, **kwargs))


def linear(*args, **kwargs):
    """Spectral Normalization이 적용된 Linear 레이어 생성"""
    return spectral_norm(nn.Linear(*args, **kwargs))


def NormLayer(c, mode='batch'):
    """정규화 레이어 생성 (BatchNorm2D 또는 GroupNorm)"""
    if mode == 'group':
        return nn.GroupNorm(c//2, c)  # 그룹 노말라이제이션
    elif mode == 'batch':
        return nn.BatchNorm2d(c)  # 배치 노말라이제이션


### Activations (활성화 함수)


class GLU(nn.Module):
    """Gated Linear Unit 활성화"""
    def forward(self, x):
        nc = x.size(1)
        assert nc % 2 == 0, '채널 수는 2로 나누어 떨어져야 합니다!'
        nc = int(nc/2)
        return x[:, :nc] * torch.sigmoid(x[:, nc:])


class Swish(nn.Module):
    """Swish 활성화 함수"""
    def forward(self, feat):
        return feat * torch.sigmoid(feat)


### Upblocks (업샘플링 블록)


class InitLayer(nn.Module):
    """초기화 레이어: 노이즈 벡터를 입력받아 초기 특징 맵 생성"""
    def __init__(self, nz, channel, sz=4):
        super().__init__()

        self.init = nn.Sequential(
            convTranspose2d(nz, channel*2, sz, 1, 0, bias=False),  # Transposed Conv2D
            NormLayer(channel*2),  # 정규화
            GLU(),  # Gated Linear Unit 활성화
        )

    def forward(self, noise):
        noise = noise.view(noise.shape[0], -1, 1, 1)  # 노이즈를 4D 텐서로 변환
        return self.init(noise)


def UpBlockSmall(in_planes, out_planes):
    """업샘플링 블록 (작은 모델용)"""
    block = nn.Sequential(
        nn.Upsample(scale_factor=2, mode='nearest'),  # 업샘플링
        conv2d(in_planes, out_planes*2, 3, 1, 1, bias=False),  # Conv2D
        NormLayer(out_planes*2),  # 정규화
        GLU()  # Gated Linear Unit 활성화
    )
    return block


class UpBlockSmallCond(nn.Module):
    """업샘플링 블록 (조건부 작은 모델용)"""
    def __init__(self, in_planes, out_planes, z_dim):
        super().__init__()
        self.in_planes = in_planes
        self.out_planes = out_planes
        self.up = nn.Upsample(scale_factor=2, mode='nearest')  # 업샘플링
        self.conv = conv2d(in_planes, out_planes*2, 3, 1, 1, bias=False)  # Conv2D

        # 조건부 BatchNorm 생성
        which_bn = functools.partial(CCBN, which_linear=linear, input_size=z_dim)
        self.bn = which_bn(2*out_planes)
        self.act = GLU()

    def forward(self, x, c):
        x = self.up(x)
        x = self.conv(x)
        x = self.bn(x, c)  # 조건부 정규화 적용
        x = self.act(x)
        return x


def UpBlockBig(in_planes, out_planes):
    """업샘플링 블록 (큰 모델용)"""
    block = nn.Sequential(
        nn.Upsample(scale_factor=2, mode='nearest'),
        conv2d(in_planes, out_planes*2, 3, 1, 1, bias=False),
        NoiseInjection(),
        NormLayer(out_planes*2), GLU(),
        conv2d(out_planes, out_planes*2, 3, 1, 1, bias=False),
        NoiseInjection(),
        NormLayer(out_planes*2), GLU()
    )
    return block


class UpBlockBigCond(nn.Module):
    """업샘플링 블록 (조건부 큰 모델용)"""
    def __init__(self, in_planes, out_planes, z_dim):
        super().__init__()
        self.in_planes = in_planes
        self.out_planes = out_planes
        self.up = nn.Upsample(scale_factor=2, mode='nearest')
        self.conv1 = conv2d(in_planes, out_planes*2, 3, 1, 1, bias=False)
        self.conv2 = conv2d(out_planes, out_planes*2, 3, 1, 1, bias=False)

        which_bn = functools.partial(CCBN, which_linear=linear, input_size=z_dim)
        self.bn1 = which_bn(2*out_planes)
        self.bn2 = which_bn(2*out_planes)
        self.act = GLU()
        self.noise = NoiseInjection()

    def forward(self, x, c):
        # 첫 번째 블록
        x = self.up(x)
        x = self.conv1(x)
        x = self.noise(x)
        x = self.bn1(x, c)
        x = self.act(x)

        # 두 번째 블록
        x = self.conv2(x)
        x = self.noise(x)
        x = self.bn2(x, c)
        x = self.act(x)

        return x


def UpBlockBig(in_planes, out_planes):
    """
    큰 업샘플링 블록 정의.
    두 개의 컨볼루션과 노이즈 주입 및 Gated Linear Unit 활성화를 포함.
    """
    block = nn.Sequential(
        nn.Upsample(scale_factor=2, mode='nearest'),  # 업샘플링
        conv2d(in_planes, out_planes*2, 3, 1, 1, bias=False),  # 첫 번째 컨볼루션
        NoiseInjection(),  # 노이즈 주입
        NormLayer(out_planes*2),  # 정규화
        GLU(),  # Gated Linear Unit 활성화
        conv2d(out_planes, out_planes*2, 3, 1, 1, bias=False),  # 두 번째 컨볼루션
        NoiseInjection(),  # 노이즈 주입
        NormLayer(out_planes*2),  # 정규화
        GLU()  # Gated Linear Unit 활성화
    )
    return block


class UpBlockBigCond(nn.Module):
    """
    조건부 큰 업샘플링 블록.
    조건부 배치 정규화를 사용하여 z_dim 입력에 따라 조정 가능.
    """
    def __init__(self, in_planes, out_planes, z_dim):
        super().__init__()
        self.in_planes = in_planes
        self.out_planes = out_planes
        self.up = nn.Upsample(scale_factor=2, mode='nearest')  # 업샘플링
        self.conv1 = conv2d(in_planes, out_planes*2, 3, 1, 1, bias=False)  # 첫 번째 컨볼루션
        self.conv2 = conv2d(out_planes, out_planes*2, 3, 1, 1, bias=False)  # 두 번째 컨볼루션

        # 조건부 BatchNorm 생성
        which_bn = functools.partial(CCBN, which_linear=linear, input_size=z_dim)
        self.bn1 = which_bn(2*out_planes)  # 첫 번째 배치 정규화
        self.bn2 = which_bn(2*out_planes)  # 두 번째 배치 정규화
        self.act = GLU()  # Gated Linear Unit 활성화
        self.noise = NoiseInjection()  # 노이즈 주입

    def forward(self, x, c):
        # 첫 번째 블록
        x = self.up(x)
        x = self.conv1(x)
        x = self.noise(x)
        x = self.bn1(x, c)  # 조건부 배치 정규화
        x = self.act(x)

        # 두 번째 블록
        x = self.conv2(x)
        x = self.noise(x)
        x = self.bn2(x, c)  # 조건부 배치 정규화
        x = self.act(x)

        return x


class SEBlock(nn.Module):
    """
    Squeeze-and-Excitation Block.
    작은 특징 맵을 사용하여 큰 특징 맵의 채널을 조정.
    """
    def __init__(self, ch_in, ch_out):
        super().__init__()
        self.main = nn.Sequential(
            nn.AdaptiveAvgPool2d(4),  # 적응형 평균 풀링
            conv2d(ch_in, ch_out, 4, 1, 0, bias=False),  # 채널 축소
            Swish(),  # Swish 활성화
            conv2d(ch_out, ch_out, 1, 1, 0, bias=False),  # 채널 복원
            nn.Sigmoid(),  # 시그모이드 활성화
        )

    def forward(self, feat_small, feat_big):
        return feat_big * self.main(feat_small)  # 작은 특징 맵으로 큰 특징 맵 조정


### Downblocks (다운샘플링 블록)


class SeparableConv2d(nn.Module):
    """
    깊이별 분리 컨볼루션.
    깊이별 컨볼루션과 점별 컨볼루션으로 구성.
    """
    def __init__(self, in_channels, out_channels, kernel_size, bias=False):
        super(SeparableConv2d, self).__init__()
        self.depthwise = conv2d(in_channels, in_channels, kernel_size=kernel_size,
                                groups=in_channels, bias=bias, padding=1)  # 깊이별 컨볼루션
        self.pointwise = conv2d(in_channels, out_channels,
                                kernel_size=1, bias=bias)  # 점별 컨볼루션

    def forward(self, x):
        out = self.depthwise(x)  # 깊이별 컨볼루션
        out = self.pointwise(out)  # 점별 컨볼루션
        return out


class DownBlock(nn.Module):
    """
    일반적인 다운샘플링 블록.
    선택적으로 깊이별 분리 컨볼루션을 사용.
    """
    def __init__(self, in_planes, out_planes, separable=False):
        super().__init__()
        if not separable:
            self.main = nn.Sequential(
                conv2d(in_planes, out_planes, 4, 2, 1),  # Conv2D
                NormLayer(out_planes),  # 정규화
                nn.LeakyReLU(0.2, inplace=True),  # LeakyReLU 활성화
            )
        else:
            self.main = nn.Sequential(
                SeparableConv2d(in_planes, out_planes, 3),  # 깊이별 분리 컨볼루션
                NormLayer(out_planes),  # 정규화
                nn.LeakyReLU(0.2, inplace=True),  # LeakyReLU 활성화
                nn.AvgPool2d(2, 2),  # 평균 풀링
            )

    def forward(self, feat):
        return self.main(feat)


class DownBlockPatch(nn.Module):
    """
    패치 기반 다운샘플링 블록.
    """
    def __init__(self, in_planes, out_planes, separable=False):
        super().__init__()
        self.main = nn.Sequential(
            DownBlock(in_planes, out_planes, separable),  # 다운블록
            conv2d(out_planes, out_planes, 1, 1, 0, bias=False),  # Conv2D
            NormLayer(out_planes),  # 정규화
            nn.LeakyReLU(0.2, inplace=True),  # LeakyReLU 활성화
        )

    def forward(self, feat):
        return self.main(feat)
