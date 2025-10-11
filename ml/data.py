# ml/data.py
import torch
from torchvision import transforms
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import CIFAR10

def get_cifar10_loaders(batch_size=128, val_split=5000):
    """
    CIFAR10 데이터셋을 불러와 훈련, 검증, 테스트용 데이터로더로 분리하여 반환합니다.
    - 훈련 데이터셋 (45,000개)
    - 검증 데이터셋 (5,000개)
    - 테스트 데이터셋 (10,000개)
    """
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    # 훈련/검증 데이터셋 분리
    full_train_dataset = CIFAR10(root='./data', train=True, download=True, transform=transform)
    num_train = len(full_train_dataset)
    indices = list(range(num_train))
    train_indices, val_indices = indices[val_split:], indices[:val_split]
    
    train_subset = Subset(full_train_dataset, train_indices)
    val_subset = Subset(full_train_dataset, val_indices)

    # 테스트 데이터셋 로드
    test_dataset = CIFAR10(root='./data', train=False, download=True, transform=transform)
    
    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_subset, batch_size=batch_size, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=2, pin_memory=True)
    
    return train_loader, val_loader, test_loader
