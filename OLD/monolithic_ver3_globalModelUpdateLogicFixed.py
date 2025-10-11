import asyncio
import time
import random
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import List, Dict, Any
from dataclasses import dataclass, field
from collections import OrderedDict
import logging

# PyTorch 라이브러리 import
import torch
import torch.nn as nn
from torchvision import models, transforms
from torch.utils.data import DataLoader, Subset, random_split
from torchvision.datasets import CIFAR10

# Skyfield 라이브러리 import
from skyfield.api import load, EarthSatellite, Topos

# --- 상수 정의 ---
MAX_ISL_DISTANCE_KM = 2000 # 위성 간 통신 최대 거리 (예시)
IOT_FLYOVER_THRESHOLD_DEG = 30.0 # IoT 클러스터 상공 통과 기준 고도각
MIN_MODELS_FOR_AGGREGATION = 2 # GS Aggregation을 위한 최소 클러스터 모델 수 (조건 완화)
AGGREGATION_STALENESS_THRESHOLD = 1 # Aggregation 시 허용하는 모델 버전 차이
LOCAL_EPOCHS = 1 # 로컬 학습 에포크 수

# --- 로깅 설정 ---
def setup_loggers():
    # 일반 시뮬레이션 로거
    sim_logger = logging.getLogger("simulation")
    sim_logger.setLevel(logging.INFO)
    sim_handler = logging.FileHandler("federated_simulation.log", mode='w')
    sim_formatter = logging.Formatter('%(asctime)s - %(message)s')
    sim_handler.setFormatter(sim_formatter)
    sim_logger.addHandler(sim_handler)
    # 콘솔에도 출력
    sim_logger.addHandler(logging.StreamHandler())

    # 성능 평가 결과 로거 (CSV)
    perf_logger = logging.getLogger("performance")
    perf_logger.setLevel(logging.INFO)
    perf_handler = logging.FileHandler("performance_results.csv", mode='w')
    # CSV 헤더 작성
    perf_handler.stream.write("timestamp,event_type,owner_id,model_version,cluster_version,accuracy,loss\n")
    perf_logger.addHandler(perf_handler)
    
    return sim_logger, perf_logger

# --- 데이터 클래스 정의 ---
@dataclass
class PyTorchModel:
    """PyTorch 모델의 상태를 담는 클래스"""
    version: int
    model_state_dict: OrderedDict
    trained_by: List[int] = field(default_factory=list)

# --- Skyfield 및 시간 관련 헬퍼 함수 ---
ts = load.timescale()

def to_ts(dt: datetime):
    return ts.from_datetime(dt)

# --- 데이터셋 및 평가 함수 ---
def get_cifar10_loaders(batch_size=128, val_split=5000):
    """CIFAR10 훈련/검증/테스트 데이터로더를 반환"""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    full_train_dataset = CIFAR10(root='./data', train=True, download=True, transform=transform)
    num_train = len(full_train_dataset)
    indices = list(range(num_train))
    train_indices, val_indices = indices[val_split:], indices[:val_split]
    
    train_subset = Subset(full_train_dataset, train_indices)
    val_subset = Subset(full_train_dataset, val_indices)

    test_dataset = CIFAR10(root='./data', train=False, download=True, transform=transform)
    
    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    
    return train_loader, val_loader, test_loader

def evaluate_model(model_state_dict, data_loader, device):
    """주어진 모델과 데이터로더로 성능을 평가"""
    model = create_mobilenet()
    model.load_state_dict(model_state_dict)
    model.to(device)
    model.eval()
    
    criterion = nn.CrossEntropyLoss()
    correct = 0
    total = 0
    total_loss = 0.0
    
    with torch.no_grad():
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
    accuracy = 100 * correct / total
    avg_loss = total_loss / len(data_loader)
    return accuracy, avg_loss

# --- 유틸리티 함수 ---
def create_mobilenet():
    """CIFAR10에 맞는 MobileNetV3-Small 모델 생성"""
    model = models.mobilenet_v3_small(weights=None, num_classes=10)
    return model

def fed_avg(models_to_average: List[OrderedDict]) -> OrderedDict:
    """Federated Averaging을 수행하는 함수"""
    if not models_to_average: return OrderedDict()
    avg_state_dict = OrderedDict()
    for key in models_to_average[0].keys():
        tensors = [model[key].float() for model in models_to_average]
        avg_tensor = torch.stack(tensors).mean(dim=0)
        avg_state_dict[key] = avg_tensor
    return avg_state_dict

class SimulationClock:
    """시뮬레이션의 전역 시간을 관리하는 클래스"""
    def __init__(self, start_dt: datetime, time_step: timedelta, real_interval: float, sim_logger=None):
        self._current_dt = start_dt
        self.time_step = time_step
        self.real_interval = real_interval
        self.logger = sim_logger
        speed = time_step.total_seconds() / real_interval
        self.logger.info(f"시뮬레이션 시계 생성. 시작: {start_dt}, 1초당 {time_step.total_seconds()}초 진행 (x{speed:.0f} 배속)")

    async def run(self):
        while True:
            self._current_dt += self.time_step
            await asyncio.sleep(self.real_interval)

    def get_time_datetime(self) -> datetime:
        return self._current_dt
        
    def get_time_ts(self):
        return to_ts(self._current_dt)

class IoTCluster:
    """데이터 소스가 되는 IoT 클러스터를 나타내는 클래스"""
    def __init__(self, name: str, latitude: float, longitude: float, elevation: int, sim_logger=None):
        self.name = name
        self.topos = Topos(latitude_degrees=latitude, longitude_degrees=longitude, elevation_m=elevation)
        self.logger = sim_logger
        self.logger.info(f"IoT 클러스터 '{self.name}' 생성 완료.")

class GroundStation:
    """지상국 클래스"""
    def __init__(self, name: str, latitude: float, longitude: float, elevation: int, initial_model: PyTorchModel, 
                 eval_infra: dict, threshold_deg: float = 10.0,
                 min_models_agg: int = MIN_MODELS_FOR_AGGREGATION, 
                 staleness_th: int = AGGREGATION_STALENESS_THRESHOLD):
        self.name = name
        self.topos = Topos(latitude_degrees=latitude, longitude_degrees=longitude, elevation_m=elevation)
        self.threshold_deg = threshold_deg
        self.global_model = initial_model
        self.received_models_buffer: List[PyTorchModel] = []
        self._comm_status: Dict[int, bool] = {}
        self.min_models_for_aggregation = min_models_agg
        self.staleness_threshold = staleness_th
        self.logger = eval_infra['sim_logger']
        self.perf_logger = eval_infra['perf_logger']
        self.test_loader = eval_infra['test_loader']
        self.device = eval_infra['device']
        self.logger.info(f"지상국 '{self.name}' 생성 완료. 글로벌 모델 버전: {self.global_model.version}")
        self.logger.info(f"  - Aggregation 정책: 최소 모델 {self.min_models_for_aggregation}개, 버전 허용치 {self.staleness_threshold}")

    async def run(self, clock: SimulationClock, satellites: Dict[int, 'Satellite']):
        self.logger.info(f"지상국 '{self.name}' 운영 시작.")
        asyncio.create_task(self.aggregate_models_periodically())
        while True:
            current_ts = clock.get_time_ts()
            for sat_id, sat in satellites.items():
                if not isinstance(sat, MasterSatellite): continue
                elevation = (sat.satellite_obj - self.topos).at(current_ts).altaz()[0].degrees
                prev_visible = self._comm_status.get(sat_id, False)
                visible_now = elevation >= self.threshold_deg

                if visible_now:
                    if not prev_visible: # First moment of contact (AOS)
                        self.logger.info(f"📡 [AOS] {self.name} <-> MasterSAT {sat_id} 통신 시작 (고도각: {elevation:.2f}°)")
                        sat.state = 'COMMUNICATING_GS'
                    
                    # 1. 수신 먼저 시도
                    if sat.model_ready_to_upload:
                        await self.receive_model_from_satellite(sat)
                    
                    # 2. 그 다음 송신
                    await self.send_model_to_satellite(sat)

                elif prev_visible and not visible_now: # LOS
                    self.logger.info(f"📡 [LOS] {self.name} <-> MasterSAT {sat_id} 통신 종료 (고도각: {elevation:.2f}°)")
                    sat.state = 'IDLE'
                
                self._comm_status[sat_id] = visible_now

            await asyncio.sleep(clock.real_interval) # 시계의 실제 대기 시간과 동기화

    async def send_model_to_satellite(self, satellite: 'MasterSatellite'):
        self.logger.info(f"  📤 {self.name} -> MasterSAT {satellite.sat_id}: 글로벌 모델 전송 (버전 {self.global_model.version})")
        await satellite.receive_global_model(self.global_model)

    async def receive_model_from_satellite(self, satellite: 'MasterSatellite'):
        cluster_model = await satellite.send_local_model()
        if cluster_model:
            self.logger.info(f"  📥 {self.name} <- MasterSAT {satellite.sat_id}: 클러스터 모델 수신 완료 (버전 {cluster_model.version}, 학습자: {cluster_model.trained_by})")
            self.received_models_buffer.append(cluster_model)

    async def aggregate_models_periodically(self):
        while True:
            await asyncio.sleep(30)
            if len(self.received_models_buffer) < self.min_models_for_aggregation: continue
            max_version_in_buffer = max(model.version for model in self.received_models_buffer)
            if max_version_in_buffer < self.global_model.version: continue

            version_lower_bound = max_version_in_buffer - self.staleness_threshold
            models_to_aggregate = [m for m in self.received_models_buffer if m.version >= version_lower_bound]
            if len(models_to_aggregate) < self.min_models_for_aggregation: continue

            self.logger.info(f"✨ [{self.name} Aggregation] {len(models_to_aggregate)}개 모델(v >= {version_lower_bound})과 기존 글로벌 모델(v{self.global_model.version}) 취합 시작...")
            
            state_dicts_to_avg = [self.global_model.model_state_dict] + [m.model_state_dict for m in models_to_aggregate]
            new_state_dict = fed_avg(state_dicts_to_avg)
            
            new_version = self.global_model.version + 1 # 버전업
            all_contributors = list(set(self.global_model.trained_by + [p for model in models_to_aggregate for p in model.trained_by]))
            self.global_model = PyTorchModel(version=new_version, model_state_dict=new_state_dict, trained_by=all_contributors)
            self.logger.info(f"✨ [{self.name} Aggregation] 새로운 글로벌 모델 생성 완료! (버전 {self.global_model.version})")

            accuracy, loss = evaluate_model(self.global_model.model_state_dict, self.test_loader, self.device)
            self.logger.info(f"  🧪 [Global Test] Owner: {self.name}, Version: {self.global_model.version}, Accuracy: {accuracy:.2f}%, Loss: {loss:.4f}")
            self.perf_logger.info(f"{datetime.now(timezone.utc).isoformat()},GLOBAL_TEST,{self.name},{self.global_model.version},N/A,{accuracy:.4f},{loss:.6f}")

            self.received_models_buffer = [m for m in self.received_models_buffer if m not in models_to_aggregate]
            
            if self.received_models_buffer:
                try:
                    current_max_version = max(m.version for m in self.received_models_buffer)
                    cleanup_lower_bound = current_max_version - self.staleness_threshold
                    models_to_discard = [m for m in self.received_models_buffer if m.version < cleanup_lower_bound]
                    if models_to_discard:
                        discard_versions = {m.version for m in models_to_discard}
                        self.logger.info(f"  🗑️  [{self.name}] {len(models_to_discard)}개의 오래된 모델 정리 (버전: {discard_versions})")
                        self.received_models_buffer = [m for m in self.received_models_buffer if m not in models_to_discard]
                except ValueError: pass

class Satellite:
    """모든 위성의 기본 클래스"""
    def __init__(self, sat_id: int, satellite_obj: EarthSatellite, clock: SimulationClock, initial_model: PyTorchModel, 
                 iot_clusters: List[IoTCluster], eval_infra: dict):
        self.sat_id = sat_id
        self.satellite_obj = satellite_obj
        self.clock = clock
        self.local_model: PyTorchModel = initial_model
        self.iot_clusters = iot_clusters
        self.position = {"lat": 0.0, "lon": 0.0, "alt": 0.0}
        self.state = 'IDLE'
        self.model_ready_to_upload = False
        self.logger = eval_infra['sim_logger']
        self.perf_logger = eval_infra['perf_logger']
        self.train_loader = eval_infra['train_loader']
        self.val_loader = eval_infra['val_loader']
        self.test_loader = eval_infra['test_loader']
        self.device = eval_infra['device']

    async def run(self):
        raise NotImplementedError("Subclasses should implement this method")

    async def _propagate_orbit(self):
        while True:
            await asyncio.sleep(self.clock.real_interval) # 시계에 맞춰 대기
            current_ts = self.clock.get_time_ts()
            geocentric = self.satellite_obj.at(current_ts)
            subpoint = geocentric.subpoint()
            self.position["lat"], self.position["lon"], self.position["alt"] = subpoint.latitude.degrees, subpoint.longitude.degrees, subpoint.elevation.km
            
    async def train_local_model(self):
        self.state = 'TRAINING'
        self.logger.info(f"  🧠 SAT {self.sat_id}: 로컬 학습 시작 (기반 모델 버전: {self.local_model.version}).")
        
        try:
            temp_model = create_mobilenet()
            temp_model.load_state_dict(self.local_model.model_state_dict)
            temp_model.to(self.device)
            temp_model.train()
            criterion = nn.CrossEntropyLoss()
            optimizer = torch.optim.SGD(temp_model.parameters(), lr=0.01, momentum=0.9)
            
            for epoch in range(LOCAL_EPOCHS):
                for images, labels in self.train_loader:
                    images, labels = images.to(self.device), labels.to(self.device)
                    optimizer.zero_grad()
                    outputs = temp_model(images)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()
            
            self.local_model.model_state_dict = temp_model.state_dict()
        except Exception as e:
            self.logger.error(f"  💀 SAT {self.sat_id}: 학습 중 에러 발생 - {e}", exc_info=True)
            self.state = 'IDLE'
            return

        if self.state == 'TRAINING':
            self.logger.info(f"  🧠 SAT {self.sat_id}: 로컬 학습 완료 ({LOCAL_EPOCHS} 에포크).")
            accuracy, loss = evaluate_model(self.local_model.model_state_dict, self.val_loader, self.device)
            self.logger.info(f"  📊 [Local Validation] SAT: {self.sat_id}, Version: {self.local_model.version}, Accuracy: {accuracy:.2f}%, Loss: {loss:.4f}")
            self.perf_logger.info(f"{datetime.now(timezone.utc).isoformat()},LOCAL_VALIDATION,{self.sat_id},{self.local_model.version},N/A,{accuracy:.4f},{loss:.6f}")

            self.local_model.trained_by = [self.sat_id]
            self.model_ready_to_upload = True
            self.state = 'IDLE'
        else:
            self.logger.warning(f"  ⚠️ SAT {self.sat_id}: 학습 중단 (상태 변경: {self.state})")
    
    def get_distance_to(self, other_sat: 'Satellite') -> float:
        current_ts = self.clock.get_time_ts()
        return (self.satellite_obj - other_sat.satellite_obj).at(current_ts).distance().km

class WorkerSatellite(Satellite):
    """클러스터 내에서 실제 학습을 담당하는 워커 위성"""
    def __init__(self, *args, master: 'MasterSatellite', **kwargs):
        super().__init__(*args, **kwargs)
        self.master = master
        self.logger.info(f"WorkerSAT {self.sat_id} 생성, Master: {master.sat_id}. 초기 모델 버전: {self.local_model.version}")

    async def run(self):
        self.logger.info(f"WorkerSAT {self.sat_id} 임무 시작.")
        asyncio.create_task(self._propagate_orbit())
        await self.monitor_iot_and_train()

    async def monitor_iot_and_train(self):
        while True:
            if self.state == 'IDLE' and not self.model_ready_to_upload:
                current_ts = self.clock.get_time_ts()
                for iot in self.iot_clusters:
                    elevation = (self.satellite_obj - iot.topos).at(current_ts).altaz()[0].degrees
                    if elevation >= IOT_FLYOVER_THRESHOLD_DEG:
                        self.logger.info(f"🛰️  WorkerSAT {self.sat_id}가 IoT 클러스터 '{iot.name}' 상공 통과 (고도각: {elevation:.2f}°). 학습 시작.")
                        asyncio.create_task(self.train_local_model())
                        break 
            await asyncio.sleep(10)

class MasterSatellite(Satellite):
    """클러스터를 관리하고 지상국과 통신하는 마스터 위성"""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cluster_members: Dict[int, WorkerSatellite] = {}
        self.cluster_model_buffer: List[PyTorchModel] = []
        self.state = 'IDLE'
        self.cluster_version_counter = 0 # 클러스터 자체 버전 카운터
        self.logger.info(f"MasterSAT {self.sat_id} 생성. 초기 모델 버전: {self.local_model.version}")

    def add_member(self, worker: WorkerSatellite):
        self.cluster_members[worker.sat_id] = worker

    async def run(self):
        self.logger.info(f"MasterSAT {self.sat_id} 임무 시작.")
        asyncio.create_task(self._propagate_orbit())
        asyncio.create_task(self.manage_cluster_isl())
        asyncio.create_task(self.aggregate_models_periodically())

    async def manage_cluster_isl(self):
        while True:
            for worker in self.cluster_members.values():
                distance = self.get_distance_to(worker)
                if distance <= MAX_ISL_DISTANCE_KM:
                    if self.local_model.version > worker.local_model.version or \
                       (self.local_model.version == worker.local_model.version and self.local_model.model_state_dict is not worker.local_model.model_state_dict):
                        await self.send_model_to_worker(worker)
                    if worker.model_ready_to_upload:
                        await self.receive_model_from_worker(worker)
            await asyncio.sleep(10)

    async def receive_global_model(self, model: PyTorchModel):
        if model.version >= self.local_model.version:
            # 새 글로벌 모델을 받으면 클러스터 버전은 리셋
            if model.version > self.local_model.version:
                self.logger.info(f"  🛰️  MasterSAT {self.sat_id}: 새로운 글로벌 모델 수신 (v{model.version}). 클러스터 버전 리셋.")
                self.cluster_version_counter = 0
            self.local_model = model
            self.model_ready_to_upload = False
        else:
             self.logger.info(f"  🛰️  MasterSAT {self.sat_id}: 더 오래된 버전의 모델({model.version}) 수신. 스킵.")

    async def send_model_to_worker(self, worker: WorkerSatellite):
        self.logger.info(f"  🛰️ -> 🛰️  Master {self.sat_id} -> Worker {worker.sat_id}: 모델 전송 (버전 {self.local_model.version})")
        worker.local_model = self.local_model
    
    async def receive_model_from_worker(self, worker: WorkerSatellite):
        self.cluster_model_buffer.append(worker.local_model)
        worker.model_ready_to_upload = False
        self.logger.info(f"  📥 MasterSAT {self.sat_id}: Worker {worker.sat_id} 모델 수신. (버퍼 크기: {len(self.cluster_model_buffer)})")

    async def aggregate_models_periodically(self):
        while True:
            await asyncio.sleep(30)
            if not self.cluster_model_buffer:
                continue
            await self._aggregate_and_evaluate_cluster_models()

    async def _aggregate_and_evaluate_cluster_models(self):
        self.logger.info(f"  ✨ [Cluster Aggregation] Master {self.sat_id}: {len(self.cluster_model_buffer)}개 워커 모델과 기존 클러스터 모델 취합 시작")
        
        state_dicts_to_avg = [self.local_model.model_state_dict] + [m.model_state_dict for m in self.cluster_model_buffer]
        new_state_dict = fed_avg(state_dicts_to_avg)
        all_contributors = list(set(self.local_model.trained_by + [p for model in self.cluster_model_buffer for p in model.trained_by]))
        
        self.local_model.model_state_dict = new_state_dict
        self.local_model.trained_by = all_contributors
        self.model_ready_to_upload = True
        self.cluster_version_counter += 1
        self.logger.info(f"  ✨ [Cluster Aggregation] Master {self.sat_id}: 클러스터 모델 업데이트 완료. 학습자: {self.local_model.trained_by}")

        accuracy, loss = evaluate_model(self.local_model.model_state_dict, self.test_loader, self.device)
        self.logger.info(f"  🧪 [Cluster Test] Owner: SAT_{self.sat_id}, Global Ver: {self.local_model.version}, Cluster Ver: {self.cluster_version_counter}, Accuracy: {accuracy:.2f}%, Loss: {loss:.4f}")
        self.perf_logger.info(f"{datetime.now(timezone.utc).isoformat()},CLUSTER_TEST,SAT_{self.sat_id},{self.local_model.version},{self.cluster_version_counter},{accuracy:.4f},{loss:.6f}")

        self.cluster_model_buffer.clear()

    async def send_local_model(self) -> PyTorchModel | None:
        if self.model_ready_to_upload:
            self.model_ready_to_upload = False
            return self.local_model
        return None

def load_constellation(tle_path: str, sim_logger) -> Dict[int, EarthSatellite]:
    if not Path(tle_path).exists(): raise FileNotFoundError(f"'{tle_path}' 파일을 찾을 수 없습니다.")
    satellites = {}
    with open(tle_path, "r") as f:
        lines = [line.strip() for line in f.readlines()]; i = 0
        while i < len(lines):
            name, line1, line2 = lines[i:i+3]; sat_id = int(name.replace("SAT", ""))
            satellites[sat_id] = EarthSatellite(line1, line2, name, ts)
            i += 3
    sim_logger.info(f"총 {len(satellites)}개의 위성을 TLE 파일에서 불러왔습니다.")
    return satellites

async def main():
    sim_logger, perf_logger = setup_loggers()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    sim_logger.info(f"Using device: {device}")

    sim_logger.info("Loading CIFAR10 dataset...")
    train_loader, val_loader, test_loader = get_cifar10_loaders()
    sim_logger.info("Dataset loaded.")

    eval_infra = {
        "sim_logger": sim_logger,
        "perf_logger": perf_logger,
        "train_loader": train_loader,
        "val_loader": val_loader,
        "test_loader": test_loader,
        "device": device
    }

    start_time = datetime.now(timezone.utc)
    simulation_clock = SimulationClock(
        start_dt=start_time, 
        time_step=timedelta(minutes=10), # 시뮬레이션 시간 점프 크기
        real_interval=1.0, # 실제 시간 대기
        sim_logger=sim_logger
    )

    initial_pytorch_model = create_mobilenet()
    initial_global_model = PyTorchModel(version=0, model_state_dict=initial_pytorch_model.state_dict())
    
    ground_stations = [
        GroundStation("Seoul-GS", 37.5665, 126.9780, 34, initial_model=initial_global_model, eval_infra=eval_infra),
        GroundStation("Houston-GS", 29.7604, -95.3698, 12, initial_model=initial_global_model, eval_infra=eval_infra)
    ]
    
    iot_clusters = [
        IoTCluster("Amazon_Forest", -3.47, -62.37, 100, sim_logger=sim_logger),
        IoTCluster("Great_Barrier_Reef", -18.29, 147.77, 0, sim_logger=sim_logger),
        IoTCluster("Siberian_Tundra", 68.35, 18.79, 420, sim_logger=sim_logger)
    ]

    all_sats_skyfield = load_constellation("constellation.tle", sim_logger)
    satellites_in_sim: Dict[int, Satellite] = {}
    sat_ids = sorted(list(all_sats_skyfield.keys())) # ID 순서대로 정렬
    
    num_masters = 10
    sats_per_plane = 20 # TLE 파일 구조에 따라 궤도면당 위성 수 정의
    
    if len(sat_ids) < num_masters * sats_per_plane:
        raise ValueError(f"시뮬레이션을 위해 최소 {num_masters * sats_per_plane}개의 위성 TLE가 필요합니다.")
        
    master_ids = [sat_ids[i * sats_per_plane] for i in range(num_masters)]
    worker_ids = [sid for sid in sat_ids if sid not in master_ids]
    
    sim_logger.info(f"마스터 위성으로 {master_ids}가 선정되었습니다.")

    masters = []
    for m_id in master_ids:
        master_sat = MasterSatellite(
            m_id, all_sats_skyfield[m_id], simulation_clock,
            initial_model=initial_global_model,
            iot_clusters=iot_clusters, eval_infra=eval_infra
        )
        satellites_in_sim[m_id] = master_sat
        masters.append(master_sat)

    for i, w_id in enumerate(worker_ids):
        assigned_master = masters[i % num_masters]
        worker_sat = WorkerSatellite(
            w_id, all_sats_skyfield[w_id], simulation_clock,
            initial_model=initial_global_model,
            iot_clusters=iot_clusters,
            master=assigned_master, eval_infra=eval_infra
        )
        assigned_master.add_member(worker_sat)
        satellites_in_sim[w_id] = worker_sat

    tasks = [
        asyncio.create_task(simulation_clock.run()),
        *[asyncio.create_task(gs.run(simulation_clock, satellites_in_sim)) for gs in ground_stations],
        *[asyncio.create_task(sat.run()) for sat in satellites_in_sim.values()]
    ]
    await asyncio.gather(*tasks)

if __name__ == "__main__":
    try:
        torch.multiprocessing.set_start_method('spawn', force=True)
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n시뮬레이션을 종료합니다.")
    except FileNotFoundError as e:
        print(e)
    except Exception as e:
        sim_logger, _ = setup_loggers()
        sim_logger.error(f"\n시뮬레이션 중 치명적인 에러 발생: {e}", exc_info=True)

