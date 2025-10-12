# ml/data.py
import torch
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader, Subset, random_split
from torchvision.datasets import CIFAR10

def get_cifar10_loaders(batch_size=128, val_split=0.1, data_root='./data'):
    """
    CIFAR10 데이터셋을 불러와 훈련, 검증, 테스트용 데이터로더로 분리하여 반환합니다.
    - 훈련 데이터셋 (45,000개)
    - 검증 데이터셋 (5,000개)
    - 테스트 데이터셋 (10,000개)
    """
    transform_train = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5), # 50% 확률로 좌우 반전
        transforms.RandomRotation(degrees=10),   # -10도 ~ 10도 사이로 랜덤 회전
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    transform_test = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])


    # 훈련/검증 데이터셋 분리
    full_train_dataset = CIFAR10(root=data_root, train=True, download=True, transform=transform_train)

    # num_train = len(full_train_dataset)
    # indices = list(range(num_train))
    # train_indices, val_indices = indices[val_split:], indices[:val_split]
    # train_subset = Subset(full_train_dataset, train_indices)
    # val_subset = Subset(full_train_dataset, val_indices)
    # # 테스트 데이터셋 로드
    # test_dataset = CIFAR10(root='./data', train=False, download=True, transform=transform)
    
    # train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    # val_loader = DataLoader(val_subset, batch_size=batch_size, num_workers=2, pin_memory=True)
    # test_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=2, pin_memory=True)

    num_train = len(full_train_dataset)
    val_size = int(num_train * val_split)
    train_size = num_train - val_size

    generator = torch.Generator().manual_seed(42)
    train_dataset, val_dataset = random_split(full_train_dataset, [train_size, val_size], generator=generator)
    
    # 참고: val_dataset은 full_train_dataset에서 분리되었기 때문에 transform_train (증강 포함)을 상속받습니다.
    # 더 엄밀한 검증을 원한다면, 검증셋을 위한 별도의 데이터셋 객체에 transform_test를 적용할 수 있으나,
    # 현재 구조에서는 학습 과정의 일부로 간주하여 그대로 사용합니다.

    # CIFAR10 테스트 데이터셋 다운로드 및 변환 적용
    test_dataset = torchvision.datasets.CIFAR10(root=data_root, train=False,
                                                    download=True, transform=transform_test)

    # 데이터 로더 생성
    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                              shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size,
                            shuffle=False, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size,
                             shuffle=False, num_workers=2, pin_memory=True)
    
    print("CIFAR10 DataLoaders created with Data Augmentation.")
    print(f" - Image size: 224x224")
    print(f" - Batch size: {batch_size}")
    print(f" - Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}, Test samples: {len(test_dataset)}")

    return train_loader, val_loader, test_loader
