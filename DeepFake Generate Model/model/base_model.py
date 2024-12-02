import os
import torch
import sys

# BaseModel 클래스: 모델의 기본 구조를 정의하는 부모 클래스
class BaseModel(torch.nn.Module):
    def name(self):
        """
        모델 이름을 반환하는 함수
        """
        return 'BaseModel'

    def initialize(self, opt):
        """
        모델 초기화 함수
        opt: 모델 설정 정보
        """
        self.opt = opt
        self.gpu_ids = opt.gpu_ids  # 사용할 GPU ID
        self.isTrain = opt.isTrain  # 학습 모드 여부
        self.Tensor = torch.cuda.FloatTensor if self.gpu_ids else torch.Tensor  # Tensor 타입 설정
        self.save_dir = os.path.join(opt.checkpoints_dir, opt.name)  # 체크포인트 저장 경로

    def set_input(self, input):
        """
        입력 데이터를 설정
        """
        self.input = input

    def forward(self):
        """
        순전파를 정의 (추상 메서드, 서브클래스에서 구현 필요)
        """
        pass

    def test(self):
        """
        테스트 시 호출 (역전파를 사용하지 않음)
        """
        pass

    def get_image_paths(self):
        """
        이미지 경로를 반환
        """
        pass

    def optimize_parameters(self):
        """
        모델의 파라미터 최적화 (서브클래스에서 구현 필요)
        """
        pass

    def get_current_visuals(self):
        """
        현재 시각화 가능한 데이터 반환
        """
        return self.input

    def get_current_errors(self):
        """
        현재 에러 정보를 반환
        """
        return {}

    def save(self, label):
        """
        모델 저장
        """
        pass

    # 네트워크 저장 함수 (서브클래스에서 사용 가능)
    def save_network(self, network, network_label, epoch_label, gpu_ids=None):
        """
        네트워크 상태 저장
        network: 저장할 네트워크
        network_label: 네트워크 이름
        epoch_label: 에포크 정보
        """
        save_filename = '{}_net_{}.pth'.format(epoch_label, network_label)
        save_path = os.path.join(self.save_dir, save_filename)
        torch.save(network.cpu().state_dict(), save_path)  # CPU로 상태 저장
        if torch.cuda.is_available():
            network.cuda()  # GPU로 이동

    # 옵티마이저 저장 함수
    def save_optim(self, network, network_label, epoch_label, gpu_ids=None):
        """
        옵티마이저 상태 저장
        """
        save_filename = '{}_optim_{}.pth'.format(epoch_label, network_label)
        save_path = os.path.join(self.save_dir, save_filename)
        torch.save(network.state_dict(), save_path)

    # 네트워크 로드 함수 (서브클래스에서 사용 가능)
    def load_network(self, network, network_label, epoch_label, save_dir=''):
        """
        저장된 네트워크 상태 로드
        """
        save_filename = '%s_net_%s.pth' % (epoch_label, network_label)
        if not save_dir:
            save_dir = self.save_dir
        save_path = os.path.join(save_dir, save_filename)
        if not os.path.isfile(save_path):
            print('%s not exists yet!' % save_path)
            if network_label == 'G':
                raise('Generator must exist!')
        else:
            try:
                network.load_state_dict(torch.load(save_path))  # 상태 로드
            except:
                pretrained_dict = torch.load(save_path)
                model_dict = network.state_dict()
                try:
                    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
                    network.load_state_dict(pretrained_dict)
                    if self.opt.verbose:
                        print('Pretrained network %s has excessive layers; Only loading layers that are used' % network_label)
                except:
                    print('Pretrained network %s has fewer layers; The following are not initialized:' % network_label)
                    for k, v in pretrained_dict.items():
                        if v.size() == model_dict[k].size():
                            model_dict[k] = v

                    not_initialized = set()
                    for k, v in model_dict.items():
                        if k not in pretrained_dict or v.size() != pretrained_dict[k].size():
                            not_initialized.add(k.split('.')[0])

                    print(sorted(not_initialized))
                    network.load_state_dict(model_dict)

    # 옵티마이저 로드 함수
    def load_optim(self, network, network_label, epoch_label, save_dir=''):
        """
        저장된 옵티마이저 상태 로드
        """
        save_filename = '%s_optim_%s.pth' % (epoch_label, network_label)
        if not save_dir:
            save_dir = self.save_dir
        save_path = os.path.join(save_dir, save_filename)
        if not os.path.isfile(save_path):
            print('%s not exists yet!' % save_path)
            if network_label == 'G':
                raise('Generator must exist!')
        else:
            try:
                network.load_state_dict(torch.load(save_path, map_location=torch.device("cpu")))
            except:
                pretrained_dict = torch.load(save_path, map_location=torch.device("cpu"))
                model_dict = network.state_dict()
                try:
                    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
                    network.load_state_dict(pretrained_dict)
                    if self.opt.verbose:
                        print('Pretrained network %s has excessive layers; Only loading layers that are used' % network_label)
                except:
                    print('Pretrained network %s has fewer layers; The following are not initialized:' % network_label)
                    for k, v in pretrained_dict.items():
                        if v.size() == model_dict[k].size():
                            model_dict[k] = v

                    not_initialized = set()
                    for k, v in model_dict.items():
                        if k not in pretrained_dict or v.size() != pretrained_dict[k].size():
                            not_initialized.add(k.split('.')[0])

                    print(sorted(not_initialized))
                    network.load_state_dict(model_dict)

    def update_learning_rate():
        """
        학습률 업데이트 (서브클래스에서 구현 필요)
        """
        pass
