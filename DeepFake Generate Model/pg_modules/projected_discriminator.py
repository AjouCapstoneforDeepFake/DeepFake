from functools import partial
import numpy as np
import torch
import torch.nn as nn

from pg_modules.blocks import DownBlock, DownBlockPatch, conv2d
from pg_modules.projector import F_RandomProj
from pg_modules.diffaug import DiffAugment


class SingleDisc(nn.Module):
    """
    단일 스케일 판별기 클래스.
    다양한 입력 크기에서 작동하며, 입력 크기를 점진적으로 줄여가며 특징을 추출.
    """
    def __init__(self, nc=None, ndf=None, start_sz=256, end_sz=8, head=None, separable=False, patch=False):
        super().__init__()
        # 채널 설정
        channel_dict = {4: 512, 8: 512, 16: 256, 32: 128, 64: 64, 128: 64,
                        256: 32, 512: 16, 1024: 8}

        # start_sz가 정확한 2의 제곱이 아닐 경우 가장 가까운 값으로 보정
        if start_sz not in channel_dict.keys():
            sizes = np.array(list(channel_dict.keys()))
            start_sz = sizes[np.argmin(abs(sizes - start_sz))]
        self.start_sz = start_sz

        # ndf 값을 지정하지 않을 경우 기본 채널 설정 사용
        if ndf is None:
            nfc = channel_dict
        else:
            nfc = {k: ndf for k, v in channel_dict.items()}

        # nc 값이 지정되었고 head가 없는 경우
        if nc is not None and head is None:
            nfc[start_sz] = nc

        layers = []

        # 입력이 전체 modality인 경우 head 추가
        if head:
            layers += [conv2d(nc, nfc[256], 3, 1, 1, bias=False),
                       nn.LeakyReLU(0.2, inplace=True)]

        # Down Blocks 추가
        DB = partial(DownBlockPatch, separable=separable) if patch else partial(DownBlock, separable=separable)
        while start_sz > end_sz:
            layers.append(DB(nfc[start_sz],  nfc[start_sz//2]))
            start_sz = start_sz // 2

        layers.append(conv2d(nfc[end_sz], 1, 4, 1, 0, bias=False))  # 마지막 출력층
        self.main = nn.Sequential(*layers)

    def forward(self, x, c):
        return self.main(x)


class SingleDiscCond(nn.Module):
    """
    조건부 단일 스케일 판별기 클래스.
    클래스 정보를 활용하여 조건부 판별 가능.
    """
    def __init__(self, nc=None, ndf=None, start_sz=256, end_sz=8, head=None, separable=False, patch=False, c_dim=1000, cmap_dim=64, embedding_dim=128):
        super().__init__()
        self.cmap_dim = cmap_dim

        # 채널 설정
        channel_dict = {4: 512, 8: 512, 16: 256, 32: 128, 64: 64, 128: 64,
                        256: 32, 512: 16, 1024: 8}

        # start_sz가 정확한 2의 제곱이 아닐 경우 가장 가까운 값으로 보정
        if start_sz not in channel_dict.keys():
            sizes = np.array(list(channel_dict.keys()))
            start_sz = sizes[np.argmin(abs(sizes - start_sz))]
        self.start_sz = start_sz

        # ndf 값을 지정하지 않을 경우 기본 채널 설정 사용
        if ndf is None:
            nfc = channel_dict
        else:
            nfc = {k: ndf for k, v in channel_dict.items()}

        # nc 값이 지정되었고 head가 없는 경우
        if nc is not None and head is None:
            nfc[start_sz] = nc

        layers = []

        # 입력이 전체 modality인 경우 head 추가
        if head:
            layers += [conv2d(nc, nfc[256], 3, 1, 1, bias=False),
                       nn.LeakyReLU(0.2, inplace=True)]

        # Down Blocks 추가
        DB = partial(DownBlockPatch, separable=separable) if patch else partial(DownBlock, separable=separable)
        while start_sz > end_sz:
            layers.append(DB(nfc[start_sz],  nfc[start_sz//2]))
            start_sz = start_sz // 2
        self.main = nn.Sequential(*layers)

        # 클래스 조건 추가
        self.cls = conv2d(nfc[end_sz], self.cmap_dim, 4, 1, 0, bias=False)
        self.embed = nn.Embedding(num_embeddings=c_dim, embedding_dim=embedding_dim)  # 클래스 임베딩
        self.embed_proj = nn.Sequential(
            nn.Linear(self.embed.embedding_dim, self.cmap_dim),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x, c):
        h = self.main(x)
        out = self.cls(h)

        # 클래스 조건 기반 프로젝션
        cmap = self.embed_proj(self.embed(c.argmax(1))).unsqueeze(-1).unsqueeze(-1)
        out = (out * cmap).sum(dim=1, keepdim=True) * (1 / np.sqrt(self.cmap_dim))

        return out


class MultiScaleD(nn.Module):
    """
    멀티 스케일 판별기 클래스.
    여러 해상도의 입력에서 판별 수행.
    """
    def __init__(
        self,
        channels,
        resolutions,
        num_discs=4,
        proj_type=2,
        cond=0,
        separable=False,
        patch=False,
        **kwargs,
    ):
        super().__init__()

        assert num_discs in [1, 2, 3, 4]

        # 판별기 설정
        self.disc_in_channels = channels[:num_discs]
        self.disc_in_res = resolutions[:num_discs]
        Disc = SingleDiscCond if cond else SingleDisc

        mini_discs = []
        for i, (cin, res) in enumerate(zip(self.disc_in_channels, self.disc_in_res)):
            start_sz = res if not patch else 16
            mini_discs += [str(i), Disc(nc=cin, start_sz=start_sz, end_sz=8, separable=separable, patch=patch)],
        self.mini_discs = nn.ModuleDict(mini_discs)

    def forward(self, features, c):
        all_logits = []
        for k, disc in self.mini_discs.items():
            res = disc(features[k], c).view(features[k].size(0), -1)
            all_logits.append(res)

        all_logits = torch.cat(all_logits, dim=1)
        return all_logits


class ProjectedDiscriminator(torch.nn.Module):
    """
    프로젝션 기반 판별기.
    """
    def __init__(
        self,
        diffaug=True,
        interp224=True,
        backbone_kwargs={},
        **kwargs
    ):
        super().__init__()
        self.diffaug = diffaug
        self.interp224 = interp224
        self.feature_network = F_RandomProj(**backbone_kwargs)
        self.discriminator = MultiScaleD(
            channels=self.feature_network.CHANNELS,
            resolutions=self.feature_network.RESOLUTIONS,
            **backbone_kwargs,
        )

    def train(self, mode=True):
        self.feature_network = self.feature_network.train(False)
        self.discriminator = self.discriminator.train(mode)
        return self

    def eval(self):
        return self.train(False)
    
    def get_feature(self, x):
        """입력에서 특징 추출"""
        features = self.feature_network(x, get_features=True)
        return features

    def forward(self, x, c):
        """
        판별기 전방 전달 함수.
        입력 특징과 조건 정보를 기반으로 출력.
        """
        features, backbone_features = self.feature_network(x)
        logits = self.discriminator(features, c)

        return logits, backbone_features
