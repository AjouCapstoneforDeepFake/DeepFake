import torch
import torch.nn as nn

# InstanceNorm 클래스: 사용자 정의 인스턴스 정규화
class InstanceNorm(nn.Module):
    """
    채널별 평균과 표준 편차를 사용해 텐서를 정규화.
    """
    def __init__(self, epsilon=1e-8):
        super(InstanceNorm, self).__init__()
        self.epsilon = epsilon  # 분모가 0이 되는 것을 방지하기 위한 작은 값

    def forward(self, x):
        """
        입력 텐서 x를 정규화하여 반환.
        """
        x = x - torch.mean(x, (2, 3), True)  # 채널별 평균 제거
        tmp = torch.mul(x, x)  # 입력 값을 제곱
        tmp = torch.rsqrt(torch.mean(tmp, (2, 3), True) + self.epsilon)  # 제곱 평균 후 역제곱근
        return x * tmp  # 정규화된 텐서 반환


# ApplyStyle 클래스: 스타일 벡터를 입력 텐서에 적용
class ApplyStyle(nn.Module):
    """
    스타일 벡터를 사용해 입력 텐서의 스타일을 조정.
    """
    def __init__(self, latent_size, channels):
        super(ApplyStyle, self).__init__()
        self.linear = nn.Linear(latent_size, channels * 2)  # 스타일 벡터 생성

    def forward(self, x, latent):
        """
        스타일 변환 수행.
        """
        style = self.linear(latent)  # 스타일 벡터 계산
        shape = [-1, 2, x.size(1), 1, 1]  # 스타일 벡터 크기 조정
        style = style.view(shape)  # [batch_size, 2, n_channels, ...]
        x = x * (style[:, 0] + 1.) + style[:, 1]  # 스타일 적용
        return x


# ResnetBlock_Adain 클래스: ResNet 블록과 Adain(Adaptive Instance Normalization)을 결합
class ResnetBlock_Adain(nn.Module):
    """
    ResNet 블록에 Adain과 스타일 변환을 결합.
    """
    def __init__(self, dim, latent_size, padding_type, activation=nn.ReLU(True)):
        super(ResnetBlock_Adain, self).__init__()

        # 첫 번째 컨볼루션과 인스턴스 정규화
        conv1 = []
        if padding_type == 'reflect':
            conv1 += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv1 += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError(f"패딩 유형 [{padding_type}]은 구현되지 않았습니다.")
        conv1 += [nn.Conv2d(dim, dim, kernel_size=3, padding=p), InstanceNorm()]
        self.conv1 = nn.Sequential(*conv1)
        self.style1 = ApplyStyle(latent_size, dim)
        self.act1 = activation

        # 두 번째 컨볼루션과 인스턴스 정규화
        conv2 = []
        if padding_type == 'reflect':
            conv2 += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv2 += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError(f"패딩 유형 [{padding_type}]은 구현되지 않았습니다.")
        conv2 += [nn.Conv2d(dim, dim, kernel_size=3, padding=p), InstanceNorm()]
        self.conv2 = nn.Sequential(*conv2)
        self.style2 = ApplyStyle(latent_size, dim)

    def forward(self, x, dlatents_in_slice):
        """
        ResNet 블록을 통과하며 Adain 적용.
        """
        y = self.conv1(x)
        y = self.style1(y, dlatents_in_slice)
        y = self.act1(y)
        y = self.conv2(y)
        y = self.style2(y, dlatents_in_slice)
        out = x + y  # Residual 연결
        return out


# Generator_Adain_Upsample 클래스: Adain과 업샘플링을 사용하는 Generator
class Generator_Adain_Upsample(nn.Module):
    """
    ResNet 블록과 업샘플링을 결합하여 이미지 생성.
    """
    def __init__(self, input_nc, output_nc, latent_size, n_blocks=6, deep=False,
                 norm_layer=nn.BatchNorm2d, padding_type='reflect'):
        super(Generator_Adain_Upsample, self).__init__()

        activation = nn.ReLU(True)
        self.deep = deep  # 네트워크 깊이

        # 초기 컨볼루션 레이어
        self.first_layer = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(input_nc, 64, kernel_size=7, padding=0),
            norm_layer(64), activation
        )

        # 다운샘플링 레이어
        self.down1 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            norm_layer(128), activation
        )
        self.down2 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            norm_layer(256), activation
        )
        self.down3 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),
            norm_layer(512), activation
        )

        if self.deep:
            self.down4 = nn.Sequential(
                nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=1),
                norm_layer(512), activation
            )

        # ResNet 블록 (BottleNeck)
        self.BottleNeck = nn.Sequential(
            *[ResnetBlock_Adain(512, latent_size=latent_size, padding_type=padding_type, activation=activation)
              for _ in range(n_blocks)]
        )

        # 업샘플링 레이어
        if self.deep:
            self.up4 = nn.Sequential(
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(512), activation
            )
        self.up3 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256), activation
        )
        self.up2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128), activation
        )
        self.up1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64), activation
        )
        self.last_layer = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(64, output_nc, kernel_size=7, padding=0)
        )

    def forward(self, input, dlatents):
        """
        Generator 순전파.
        """
        x = input  # 입력 이미지
        skip1 = self.first_layer(x)
        skip2 = self.down1(skip1)
        skip3 = self.down2(skip2)
        if self.deep:
            skip4 = self.down3(skip3)
            x = self.down4(skip4)
        else:
            x = self.down3(skip3)

        bot = [x]  # ResNet 블록 출력 저장
        for block in self.BottleNeck:
            x = block(x, dlatents)
            bot.append(x)

        features = []  # 업샘플링 출력 저장
        if self.deep:
            x = self.up4(x)
            features.append(x)
        x = self.up3(x)
        features.append(x)
        x = self.up2(x)
        features.append(x)
        x = self.up1(x)
        features.append(x)
        x = self.last_layer(x)

        return x
