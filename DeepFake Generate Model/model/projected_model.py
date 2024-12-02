import torch
import torch.nn as nn

from .base_model import BaseModel
from .fs_networks_fix import Generator_Adain_Upsample

from pg_modules.projected_discriminator import ProjectedDiscriminator

# Gradient penalty 계산 함수
def compute_grad2(d_out, x_in):
    batch_size = x_in.size(0)  # 입력 텐서의 배치 크기
    grad_dout = torch.autograd.grad(
        outputs=d_out.sum(), inputs=x_in,
        create_graph=True, retain_graph=True, only_inputs=True
    )[0]  # d_out의 gradient를 계산
    grad_dout2 = grad_dout.pow(2)  # gradient를 제곱
    assert(grad_dout2.size() == x_in.size())  # gradient 크기와 입력 크기가 동일해야 함
    reg = grad_dout2.view(batch_size, -1).sum(1)  # 배치별로 gradient 제곱 값을 합산
    return reg  # 정규화 값 반환

class fsModel(BaseModel):
    # 모델 이름 반환 함수
    def name(self):
        return 'fsModel'

    # 모델 초기화 함수
    def initialize(self, opt):
        BaseModel.initialize(self, opt)  # 부모 클래스 초기화
        self.isTrain = opt.isTrain  # 훈련 모드 설정

        # 생성기 네트워크 초기화
        self.netG = Generator_Adain_Upsample(input_nc=3, output_nc=3, latent_size=512, n_blocks=9, deep=opt.Gdeep)
        self.netG.cuda()

        # ID 네트워크 초기화
        netArc_checkpoint = opt.Arc_path
        netArc_checkpoint = torch.load(netArc_checkpoint, map_location=torch.device("cpu"))  # ID 네트워크 체크포인트 로드
        self.netArc = netArc_checkpoint
        self.netArc = self.netArc.cuda()
        self.netArc.eval()  # 평가 모드로 설정
        self.netArc.requires_grad_(False)  # gradient 계산 비활성화
        if not self.isTrain:  # 훈련 모드가 아니면
            pretrained_path = opt.checkpoints_dir
            self.load_network(self.netG, 'G', opt.which_epoch, pretrained_path)  # 생성기 네트워크 로드
            return

        # 판별기 네트워크 초기화
        self.netD = ProjectedDiscriminator(diffaug=False, interp224=False, **{})
        self.netD.cuda()

        if self.isTrain:
            # 손실 함수 정의
            self.criterionFeat = nn.L1Loss()  # L1 손실
            self.criterionRec = nn.L1Loss()  # L1 손실

            # 옵티마이저 초기화
            # 생성기 옵티마이저
            params = list(self.netG.parameters())
            self.optimizer_G = torch.optim.Adam(params, lr=opt.lr, betas=(opt.beta1, 0.99), eps=1e-8)

            # 판별기 옵티마이저
            params = list(self.netD.parameters())
            self.optimizer_D = torch.optim.Adam(params, lr=opt.lr, betas=(opt.beta1, 0.99), eps=1e-8)

        # 네트워크 로드
        if opt.continue_train:  # 훈련을 이어서 할 경우
            pretrained_path = '' if not self.isTrain else opt.load_pretrain
            self.load_network(self.netG, 'G', opt.which_epoch, pretrained_path)
            self.load_network(self.netD, 'D', opt.which_epoch, pretrained_path)
            self.load_optim(self.optimizer_G, 'G', opt.which_epoch, pretrained_path)
            self.load_optim(self.optimizer_D, 'D', opt.which_epoch, pretrained_path)
        torch.cuda.empty_cache()  # GPU 캐시 정리

    # 코사인 유사도 계산 함수
    def cosin_metric(self, x1, x2):
        return torch.sum(x1 * x2, dim=1) / (torch.norm(x1, dim=1) * torch.norm(x2, dim=1))

    # 모델 저장 함수
    def save(self, which_epoch):
        self.save_network(self.netG, 'G', which_epoch)
        self.save_network(self.netD, 'D', which_epoch)
        self.save_optim(self.optimizer_G, 'G', which_epoch)
        self.save_optim(self.optimizer_D, 'D', which_epoch)

    # 고정된 파라미터 업데이트 함수
    def update_fixed_params(self):
        params = list(self.netG.parameters())
        if self.gen_features:
            params += list(self.netE.parameters())
        self.optimizer_G = torch.optim.Adam(params, lr=self.opt.lr, betas=(self.opt.beta1, 0.999))
        if self.opt.verbose:
            print('------------ Now also finetuning global generator -----------')

    # 학습률 업데이트 함수
    def update_learning_rate(self):
        lrd = self.opt.lr / self.opt.niter_decay  # 학습률 감소 비율
        lr = self.old_lr - lrd  # 새로운 학습률 계산
        for param_group in self.optimizer_D.param_groups:
            param_group['lr'] = lr  # 판별기 학습률 업데이트
        for param_group in self.optimizer_G.param_groups:
            param_group['lr'] = lr  # 생성기 학습률 업데이트
        if self.opt.verbose:
            print('update learning rate: %f -> %f' % (self.old_lr, lr))
        self.old_lr = lr  # 기존 학습률 갱신
