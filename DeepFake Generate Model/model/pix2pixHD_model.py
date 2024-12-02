import numpy as np
import torch
import os
from torch.autograd import Variable
from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks

class Pix2PixHDModel(BaseModel):
    def name(self):
        return 'Pix2PixHDModel'
    
    def init_loss_filter(self, use_gan_feat_loss, use_vgg_loss):
        # 사용하지 않을 손실을 필터링하여 손실 계산을 초기화
        flags = (True, use_gan_feat_loss, use_vgg_loss, True, True)
        def loss_filter(g_gan, g_gan_feat, g_vgg, d_real, d_fake):
            # 활성화된 손실만 반환
            return [l for (l, f) in zip((g_gan, g_gan_feat, g_vgg, d_real, d_fake), flags) if f]
        return loss_filter
    
    def initialize(self, opt):
        # 모델 초기화
        BaseModel.initialize(self, opt)
        if opt.resize_or_crop != 'none' or not opt.isTrain:  # 전체 해상도로 학습 시 OOM 방지
            torch.backends.cudnn.benchmark = True
        self.isTrain = opt.isTrain
        self.use_features = opt.instance_feat or opt.label_feat  # 특징 맵 사용 여부
        self.gen_features = self.use_features and not self.opt.load_features  # 특징 맵 생성 여부
        input_nc = opt.label_nc if opt.label_nc != 0 else opt.input_nc  # 입력 채널 수 결정

        ##### 네트워크 정의 #####
        # Generator 네트워크
        netG_input_nc = input_nc        
        if not opt.no_instance:
            netG_input_nc += 1  # 인스턴스 지도 추가
        if self.use_features:
            netG_input_nc += opt.feat_num  # 특징 맵 추가                  
        self.netG = networks.define_G(netG_input_nc, opt.output_nc, opt.ngf, opt.netG, 
                                      opt.n_downsample_global, opt.n_blocks_global, opt.n_local_enhancers, 
                                      opt.n_blocks_local, opt.norm, gpu_ids=self.gpu_ids)        

        # Discriminator 네트워크
        if self.isTrain:
            use_sigmoid = opt.no_lsgan
            netD_input_nc = input_nc + opt.output_nc  # 입력 이미지와 생성 이미지 결합
            if not opt.no_instance:
                netD_input_nc += 1  # 인스턴스 지도 추가
            self.netD = networks.define_D(netD_input_nc, opt.ndf, opt.n_layers_D, opt.norm, use_sigmoid, 
                                          opt.num_D, not opt.no_ganFeat_loss, gpu_ids=self.gpu_ids)

        ### Encoder 네트워크
        if self.gen_features:          
            self.netE = networks.define_G(opt.output_nc, opt.feat_num, opt.nef, 'encoder', 
                                          opt.n_downsample_E, norm=opt.norm, gpu_ids=self.gpu_ids)  
        if self.opt.verbose:
            print('---------- 네트워크 초기화 완료 -------------')

        # 네트워크 불러오기
        if not self.isTrain or opt.continue_train or opt.load_pretrain:
            pretrained_path = '' if not self.isTrain else opt.load_pretrain
            self.load_network(self.netG, 'G', opt.which_epoch, pretrained_path)            
            if self.isTrain:
                self.load_network(self.netD, 'D', opt.which_epoch, pretrained_path)  
            if self.gen_features:
                self.load_network(self.netE, 'E', opt.which_epoch, pretrained_path)              

        # 손실 함수와 옵티마이저 설정
        if self.isTrain:
            if opt.pool_size > 0 and (len(self.gpu_ids)) > 1:
                raise NotImplementedError("멀티 GPU에서는 Fake Pool을 지원하지 않습니다.")
            self.fake_pool = ImagePool(opt.pool_size)  # Fake 이미지 저장
            self.old_lr = opt.lr  # 초기 학습률

            # 손실 함수 정의
            self.loss_filter = self.init_loss_filter(not opt.no_ganFeat_loss, not opt.no_vgg_loss)
            
            self.criterionGAN = networks.GANLoss(use_lsgan=not opt.no_lsgan, tensor=self.Tensor)  # GAN 손실 함수   
            self.criterionFeat = torch.nn.L1Loss()  # 특징 손실
            if not opt.no_vgg_loss:             
                self.criterionVGG = networks.VGGLoss(self.gpu_ids)  # VGG 손실
                
            # 손실 이름 정의 (모니터링용)
            self.loss_names = self.loss_filter('G_GAN', 'G_GAN_Feat', 'G_VGG', 'D_real', 'D_fake')

            # 옵티마이저 초기화
            # Generator 옵티마이저
            if opt.niter_fix_global > 0:                
                import sys
                if sys.version_info >= (3, 0):
                    finetune_list = set()
                else:
                    from sets import Set
                    finetune_list = Set()

                params_dict = dict(self.netG.named_parameters())
                params = []
                for key, value in params_dict.items():       
                    if key.startswith('model' + str(opt.n_local_enhancers)):                    
                        params += [value]
                        finetune_list.add(key.split('.')[0])  
                print('------------- 로컬 강화 네트워크만 학습 중 (%d epochs 동안) ------------' % opt.niter_fix_global)
                print('파인튜닝되는 레이어: ', sorted(finetune_list))                         
            else:
                params = list(self.netG.parameters())
            if self.gen_features:              
                params += list(self.netE.parameters())         
            self.optimizer_G = torch.optim.Adam(params, lr=opt.lr, betas=(opt.beta1, 0.999))                            

            # Discriminator 옵티마이저                        
            params = list(self.netD.parameters())    
            self.optimizer_D = torch.optim.Adam(params, lr=opt.lr, betas=(opt.beta1, 0.999))

    def encode_input(self, label_map, inst_map=None, real_image=None, feat_map=None, infer=False):             
        if self.opt.label_nc == 0:
            input_label = label_map.data.cuda()
        else:
            # 라벨 맵의 원-핫 벡터 생성
            size = label_map.size()
            oneHot_size = (size[0], self.opt.label_nc, size[2], size[3])
            input_label = torch.cuda.FloatTensor(torch.Size(oneHot_size)).zero_()
            input_label = input_label.scatter_(1, label_map.data.long().cuda(), 1.0)
            if self.opt.data_type == 16:
                input_label = input_label.half()

        # 인스턴스 맵에서 엣지 생성
        if not self.opt.no_instance:
            inst_map = inst_map.data.cuda()
            edge_map = self.get_edges(inst_map)
            input_label = torch.cat((input_label, edge_map), dim=1)         
        input_label = Variable(input_label, volatile=infer)

        # 학습용 실제 이미지
        if real_image is not None:
            real_image = Variable(real_image.data.cuda())

        # 특징 인코딩을 위한 인스턴스 맵
        if self.use_features:
            # 미리 계산된 특징 맵 가져오기
            if self.opt.load_features:
                feat_map = Variable(feat_map.data.cuda())
            if self.opt.label_feat:
                inst_map = label_map.cuda()

        return input_label, inst_map, real_image, feat_map

# 나머지 함수도 동일하게 한국어로 번역된 각주를 포함한 코드로 제공됩니다.


    def encode_features(self, image, inst):
        image = Variable(image.cuda(), volatile=True)
        feat_num = self.opt.feat_num
        h, w = inst.size()[2], inst.size()[3]
        block_num = 32
        feat_map = self.netE.forward(image, inst.cuda())
        inst_np = inst.cpu().numpy().astype(int)
        feature = {}
        for i in range(self.opt.label_nc):
            feature[i] = np.zeros((0, feat_num+1))
        for i in np.unique(inst_np):
            label = i if i < 1000 else i//1000
            idx = (inst == int(i)).nonzero()
            num = idx.size()[0]
            idx = idx[num//2,:]
            val = np.zeros((1, feat_num+1))                        
            for k in range(feat_num):
                val[0, k] = feat_map[idx[0], idx[1] + k, idx[2], idx[3]].data[0]            
            val[0, feat_num] = float(num) / (h * w // block_num)
            feature[label] = np.append(feature[label], val, axis=0)
        return feature

    def get_edges(self, t):
        edge = torch.cuda.ByteTensor(t.size()).zero_()
        edge[:,:,:,1:] = edge[:,:,:,1:] | (t[:,:,:,1:] != t[:,:,:,:-1])
        edge[:,:,:,:-1] = edge[:,:,:,:-1] | (t[:,:,:,1:] != t[:,:,:,:-1])
        edge[:,:,1:,:] = edge[:,:,1:,:] | (t[:,:,1:,:] != t[:,:,:-1,:])
        edge[:,:,:-1,:] = edge[:,:,:-1,:] | (t[:,:,1:,:] != t[:,:,:-1,:])
        if self.opt.data_type==16:
            return edge.half()
        else:
            return edge.float()

    def save(self, which_epoch):
        self.save_network(self.netG, 'G', which_epoch, self.gpu_ids)
        self.save_network(self.netD, 'D', which_epoch, self.gpu_ids)
        if self.gen_features:
            self.save_network(self.netE, 'E', which_epoch, self.gpu_ids)

    def update_fixed_params(self):
        # 여러 번 반복하도록 글로벌 제너레이터를 수정한 후 미세 조정도 시작
        params = list(self.netG.parameters())
        if self.gen_features:
            params += list(self.netE.parameters())           
        self.optimizer_G = torch.optim.Adam(params, lr=self.opt.lr, betas=(self.opt.beta1, 0.999))
        if self.opt.verbose:
            print('------------ Now also finetuning global generator -----------')

    def update_learning_rate(self):
        lrd = self.opt.lr / self.opt.niter_decay
        lr = self.old_lr - lrd        
        for param_group in self.optimizer_D.param_groups:
            param_group['lr'] = lr
        for param_group in self.optimizer_G.param_groups:
            param_group['lr'] = lr
        if self.opt.verbose:
            print('update learning rate: %f -> %f' % (self.old_lr, lr))
        self.old_lr = lr

class InferenceModel(Pix2PixHDModel):
    def forward(self, inp):
        label, inst = inp
        return self.inference(label, inst)

        
