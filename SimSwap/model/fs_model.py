import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from torch.autograd import Variable
from .base_model import BaseModel  # BaseModel 클래스를 가져옴
from . import networks  # 네트워크 모듈

# 특정 정규화 클래스
class SpecificNorm(nn.Module):
    """
    특정 평균(mean)과 표준 편차(std)로 입력 텐서를 정규화.
    """
    def __init__(self, epsilon=1e-8):
        super(SpecificNorm, self).__init__()
        # RGB 채널별 평균과 표준 편차를 설정
        self.mean = np.array([0.485, 0.456, 0.406])
        self.mean = torch.from_numpy(self.mean).float().cuda().view([1, 3, 1, 1])

        self.std = np.array([0.229, 0.224, 0.225])
        self.std = torch.from_numpy(self.std).float().cuda().view([1, 3, 1, 1])

    def forward(self, x):
        """
        입력 텐서를 정규화
        """
        mean = self.mean.expand([1, 3, x.shape[2], x.shape[3]])
        std = self.std.expand([1, 3, x.shape[2], x.shape[3]])
        x = (x - mean) / std
        return x

# 얼굴 합성을 위한 모델 클래스
class fsModel(BaseModel):
    def name(self):
        """
        모델 이름 반환
        """
        return 'fsModel'

    def init_loss_filter(self, use_gan_feat_loss, use_vgg_loss):
        """
        손실 필터 초기화
        """
        flags = (True, use_gan_feat_loss, use_vgg_loss, True, True, True, True, True)

        def loss_filter(g_gan, g_gan_feat, g_vgg, g_id, g_rec, g_mask, d_real, d_fake):
            return [l for (l, f) in zip((g_gan, g_gan_feat, g_vgg, g_id, g_rec, g_mask, d_real, d_fake), flags) if f]

        return loss_filter

    def initialize(self, opt):
        """
        모델 초기화 및 네트워크 설정
        """
        BaseModel.initialize(self, opt)
        if opt.resize_or_crop != 'none' or not opt.isTrain:
            torch.backends.cudnn.benchmark = True
        self.isTrain = opt.isTrain

        device = torch.device("cuda:0")  # GPU 사용 설정

        # 입력 크기에 따라 네트워크 선택
        if opt.crop_size == 224:
            from .fs_networks import Generator_Adain_Upsample, Discriminator
        elif opt.crop_size == 512:
            from .fs_networks_512 import Generator_Adain_Upsample, Discriminator

        # Generator 초기화
        self.netG = Generator_Adain_Upsample(input_nc=3, output_nc=3, latent_size=512, n_blocks=9, deep=False)
        self.netG.to(device)

        # ID 네트워크 초기화
        netArc_checkpoint = opt.Arc_path
        netArc_checkpoint = torch.load(netArc_checkpoint, map_location=torch.device("cpu"))
        self.netArc = netArc_checkpoint.to(device)
        self.netArc.eval()  # 평가 모드로 설정

        # Training 모드에서만 Discriminator를 초기화
        if self.isTrain:
            use_sigmoid = opt.gan_mode == 'original'
            self.netD1 = Discriminator(input_nc=3, use_sigmoid=use_sigmoid).to(device)
            self.netD2 = Discriminator(input_nc=3, use_sigmoid=use_sigmoid).to(device)

            self.spNorm = SpecificNorm()
            self.downsample = nn.AvgPool2d(3, stride=2, padding=[1, 1], count_include_pad=False)

            if opt.continue_train or opt.load_pretrain:
                pretrained_path = opt.load_pretrain if self.isTrain else ''
                self.load_network(self.netG, 'G', opt.which_epoch, pretrained_path)
                self.load_network(self.netD1, 'D1', opt.which_epoch, pretrained_path)
                self.load_network(self.netD2, 'D2', opt.which_epoch, pretrained_path)

    def forward(self, img_id, img_att, latent_id, latent_att, for_G=False):
        """
        순전파 및 손실 계산
        """
        img_fake = self.netG.forward(img_att, latent_id)  # 생성된 이미지
        if not self.isTrain:
            return img_fake  # 학습 모드가 아니면 생성된 이미지만 반환

        img_fake_downsample = self.downsample(img_fake)  # 생성 이미지 다운샘플링
        img_att_downsample = self.downsample(img_att)  # 원본 이미지 다운샘플링

        # Discriminator의 가짜 이미지 손실 계산
        fea1_fake = self.netD1.forward(img_fake.detach())
        fea2_fake = self.netD2.forward(img_fake_downsample.detach())
        pred_fake = [fea1_fake, fea2_fake]
        loss_D_fake = self.criterionGAN(pred_fake, False, for_discriminator=True)

        # Discriminator의 진짜 이미지 손실 계산
        fea1_real = self.netD1.forward(img_att)
        fea2_real = self.netD2.forward(img_att_downsample)
        pred_real = [fea1_real, fea2_real]
        loss_D_real = self.criterionGAN(pred_real, True, for_discriminator=True)

        # Generator의 손실 계산
        loss_G_GAN = self.criterionGAN([fea1_fake, fea2_fake], True, for_discriminator=False)
        loss_G_ID = (1 - self.cosin_metric(self.netArc(self.spNorm(F.interpolate(img_fake, size=(112, 112)))), latent_id))
        loss_G_Rec = self.criterionRec(img_fake, img_att) * self.opt.lambda_rec

        return [self.loss_filter(loss_G_GAN, None, None, loss_G_ID, loss_G_Rec, None, loss_D_real, loss_D_fake),
                img_fake]

    def save(self, which_epoch):
        """
        네트워크 저장
        """
        self.save_network(self.netG, 'G', which_epoch, self.gpu_ids)
        self.save_network(self.netD1, 'D1', which_epoch, self.gpu_ids)
        self.save_network(self.netD2, 'D2', which_epoch, self.gpu_ids)

    def update_learning_rate(self):
        """
        학습률 업데이트
        """
        lrd = self.opt.lr / self.opt.niter_decay
        lr = self.old_lr - lrd
        for param_group in self.optimizer_D.param_groups:
            param_group['lr'] = lr
        for param_group in self.optimizer_G.param_groups:
            param_group['lr'] = lr
        self.old_lr = lr
