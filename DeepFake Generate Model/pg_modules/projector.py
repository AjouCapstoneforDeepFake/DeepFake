import torch
import torch.nn as nn
import timm
from pg_modules.blocks import FeatureFusionBlock


def _make_scratch_ccm(scratch, in_channels, cout, expand=False):
    """
    Cross Channel Mixing (CCM) 모듈 생성.
    입력 채널을 압축하거나 확장하여 새로운 출력 채널 생성.
    """
    # 출력 채널 설정 (확장 여부에 따라 다름)
    out_channels = [cout, cout*2, cout*4, cout*8] if expand else [cout]*4

    # 각 레이어에 1x1 컨볼루션 적용
    scratch.layer0_ccm = nn.Conv2d(in_channels[0], out_channels[0], kernel_size=1, stride=1, padding=0, bias=True)
    scratch.layer1_ccm = nn.Conv2d(in_channels[1], out_channels[1], kernel_size=1, stride=1, padding=0, bias=True)
    scratch.layer2_ccm = nn.Conv2d(in_channels[2], out_channels[2], kernel_size=1, stride=1, padding=0, bias=True)
    scratch.layer3_ccm = nn.Conv2d(in_channels[3], out_channels[3], kernel_size=1, stride=1, padding=0, bias=True)

    scratch.CHANNELS = out_channels

    return scratch


def _make_scratch_csm(scratch, in_channels, cout, expand):
    """
    Cross Scale Mixing (CSM) 모듈 생성.
    여러 해상도의 특징 맵을 결합하여 상위 스케일로 혼합.
    """
    scratch.layer3_csm = FeatureFusionBlock(in_channels[3], nn.ReLU(False), expand=expand, lowest=True)
    scratch.layer2_csm = FeatureFusionBlock(in_channels[2], nn.ReLU(False), expand=expand)
    scratch.layer1_csm = FeatureFusionBlock(in_channels[1], nn.ReLU(False), expand=expand)
    scratch.layer0_csm = FeatureFusionBlock(in_channels[0], nn.ReLU(False))

    # 최상위 레이어는 채널 확장을 제한
    scratch.CHANNELS = [cout, cout, cout*2, cout*4] if expand else [cout]*4

    return scratch


def _make_efficientnet(model):
    """
    EfficientNet 모델에서 특정 레이어를 추출하여 새로운 구조 생성.
    """
    pretrained = nn.Module()
    pretrained.layer0 = nn.Sequential(model.conv_stem, model.bn1, model.act1, *model.blocks[0:2])
    pretrained.layer1 = nn.Sequential(*model.blocks[2:3])
    pretrained.layer2 = nn.Sequential(*model.blocks[3:5])
    pretrained.layer3 = nn.Sequential(*model.blocks[5:9])
    return pretrained


def calc_channels(pretrained, inp_res=224):
    """
    입력 해상도에 따른 각 레이어의 채널 수 계산.
    """
    channels = []
    tmp = torch.zeros(1, 3, inp_res, inp_res)

    # 순전파 실행하여 채널 계산
    tmp = pretrained.layer0(tmp)
    channels.append(tmp.shape[1])
    tmp = pretrained.layer1(tmp)
    channels.append(tmp.shape[1])
    tmp = pretrained.layer2(tmp)
    channels.append(tmp.shape[1])
    tmp = pretrained.layer3(tmp)
    channels.append(tmp.shape[1])

    return channels


def _make_projector(im_res, cout, proj_type, expand=False):
    """
    프로젝트 네트워크 생성.
    proj_type에 따라 CCM 또는 CSM 모듈 포함.
    """
    assert proj_type in [0, 1, 2], "유효하지 않은 proj_type 값입니다."

    # 사전 학습된 EfficientNet 모델 생성
    model = timm.create_model('tf_efficientnet_lite0', pretrained=True)
    pretrained = _make_efficientnet(model)

    # 특징 맵 해상도 계산
    im_res = 256
    pretrained.RESOLUTIONS = [im_res//4, im_res//8, im_res//16, im_res//32]
    pretrained.CHANNELS = calc_channels(pretrained)

    if proj_type == 0: 
        return pretrained, None

    # CCM 모듈 추가
    scratch = nn.Module()
    scratch = _make_scratch_ccm(scratch, in_channels=pretrained.CHANNELS, cout=cout, expand=expand)
    pretrained.CHANNELS = scratch.CHANNELS

    if proj_type == 1: 
        return pretrained, scratch

    # CSM 모듈 추가
    scratch = _make_scratch_csm(scratch, in_channels=scratch.CHANNELS, cout=cout, expand=expand)

    # CSM은 x2 업샘플링을 포함하여 해상도 증가
    pretrained.RESOLUTIONS = [res*2 for res in pretrained.RESOLUTIONS]
    pretrained.CHANNELS = scratch.CHANNELS

    return pretrained, scratch


class F_RandomProj(nn.Module):
    """
    랜덤 프로젝션 네트워크.
    EfficientNet을 백본으로 사용하며, CCM 또는 CSM을 통해 특징을 혼합.
    """
    def __init__(
        self,
        im_res=256,
        cout=64,
        expand=True,
        proj_type=2,  # 0 = 프로젝션 없음, 1 = CCM, 2 = CSM
        **kwargs,
    ):
        super().__init__()
        self.proj_type = proj_type
        self.cout = cout
        self.expand = expand

        # 사전 학습된 네트워크와 프로젝션 모듈 생성
        self.pretrained, self.scratch = _make_projector(im_res=im_res, cout=self.cout, proj_type=self.proj_type, expand=self.expand)
        self.CHANNELS = self.pretrained.CHANNELS
        self.RESOLUTIONS = self.pretrained.RESOLUTIONS

    def forward(self, x, get_features=False):
        """
        특징 추출 및 프로젝션 수행.
        get_features가 True일 경우 원본 백본 특징 반환.
        """
        # 백본 네트워크에서 특징 추출
        out0 = self.pretrained.layer0(x)
        out1 = self.pretrained.layer1(out0)
        out2 = self.pretrained.layer2(out1)
        out3 = self.pretrained.layer3(out2)

        # 백본 특징 저장
        backbone_features = {
            '0': out0,
            '1': out1,
            '2': out2,
            '3': out3,
        }
        if get_features:
            return backbone_features

        if self.proj_type == 0: 
            return backbone_features

        # CCM 적용
        out0_channel_mixed = self.scratch.layer0_ccm(backbone_features['0'])
        out1_channel_mixed = self.scratch.layer1_ccm(backbone_features['1'])
        out2_channel_mixed = self.scratch.layer2_ccm(backbone_features['2'])
        out3_channel_mixed = self.scratch.layer3_ccm(backbone_features['3'])

        out = {
            '0': out0_channel_mixed,
            '1': out1_channel_mixed,
            '2': out2_channel_mixed,
            '3': out3_channel_mixed,
        }

        if self.proj_type == 1: 
            return out

        # CSM 적용
        out3_scale_mixed = self.scratch.layer3_csm(out3_channel_mixed)
        out2_scale_mixed = self.scratch.layer2_csm(out3_scale_mixed, out2_channel_mixed)
        out1_scale_mixed = self.scratch.layer1_csm(out2_scale_mixed, out1_channel_mixed)
        out0_scale_mixed = self.scratch.layer0_csm(out1_scale_mixed, out0_channel_mixed)

        out = {
            '0': out0_scale_mixed,
            '1': out1_scale_mixed,
            '2': out2_scale_mixed,
            '3': out3_scale_mixed,
        }

        return out, backbone_features
