# ml/model.py
from dataclasses import dataclass, field
from typing import List, OrderedDict, Any
from torchvision import models

@dataclass
class PyTorchModel:
    """PyTorch 모델의 상태(버전, 가중치 등)를 담는 데이터 클래스"""
    version: int
    model_state_dict: OrderedDict
    trained_by: List[int] = field(default_factory=list)

def create_mobilenet():
    """CIFAR10 데이터셋(10개 클래스)에 맞게 사전 학습 없이 초기화된 MobileNetV3-Small 모델을 생성"""
    model = models.mobilenet_v3_small(weights=None, num_classes=10)
    return model
