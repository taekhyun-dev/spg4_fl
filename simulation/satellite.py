import asyncio
from typing import List, Dict, Tuple

import torch
import torch.nn as nn
import torch.optim.lr_scheduler as lr_scheduler

from config import LOCAL_EPOCHS, MAX_ISL_DISTANCE_KM, IOT_FLYOVER_THRESHOLD_DEG
from ml.model import PyTorchModel, create_mobilenet
from ml.training import evaluate_model, fed_avg
from utils.skyfield_utils import EarthSatellite
from utils.logging_setup import KST
from datetime import datetime

class Satellite:
    """
    모든 위성의 기본 클래스.
    공통 기능(궤도 전파, 로컬 학습, 상태 관리 등)을 정의합니다.
    """
    def __init__(self, sat_id: int, satellite_obj: EarthSatellite, clock: 'SimulationClock', initial_model: PyTorchModel, 
                 iot_clusters: List['IoTCluster'], eval_infra: dict):
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
        self.training_queue = eval_infra['training_queue']

    async def run(self):
        raise NotImplementedError("Subclasses should implement this method")

    async def _propagate_orbit(self):
        """시뮬레이션 시간에 맞춰 위성의 위치를 계속 업데이트"""
        while True:
            await asyncio.sleep(self.clock.real_interval)
            current_ts = self.clock.get_time_ts()
            geocentric = self.satellite_obj.at(current_ts)
            subpoint = geocentric.subpoint()
            self.position["lat"], self.position["lon"], self.position["alt"] = subpoint.latitude.degrees, subpoint.longitude.degrees, subpoint.elevation.km
            
    def _train_and_eval(self) -> Tuple[Dict, float, float]:
        """
        실제 PyTorch 모델 학습을 수행하는 블로킹(동기) 함수.
        asyncio 이벤트 루프를 막지 않기 위해 별도의 스레드에서 실행됩니다.
        """
        # --- 학습 파트 ---
        temp_model = create_mobilenet()
        temp_model.load_state_dict(self.local_model.model_state_dict)
        temp_model.to(self.device)
        temp_model.train()
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(temp_model.parameters(), lr=3e-4, weight_decay=1e-4)
        scheduler = lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)
    
        self.logger.info(f"  🧠 SAT {self.sat_id}: 로컬 학습 시작 ({LOCAL_EPOCHS} 에포크).")
        for epoch in range(LOCAL_EPOCHS):
            for images, labels in self.train_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                optimizer.zero_grad()
                outputs = temp_model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
            scheduler.step()
            
        print(f"    - SAT {self.sat_id}: Training complete.")
        new_state_dict = temp_model.cpu().state_dict()
        self.logger.info(f"  🧠 SAT {self.sat_id}: 로컬 학습 완료 ({LOCAL_EPOCHS} 에포크). 검증 시작...")
            
        # --- 검증 파트 ---
        accuracy, loss = evaluate_model(new_state_dict, self.val_loader, self.device)
            
        return new_state_dict, accuracy, loss

    async def train_local_model(self):
        """CIFAR10 데이터셋으로 로컬 모델을 학습하고 검증"""
        # self.state = 'TRAINING'
        # self.logger.info(f"  🧠 SAT {self.sat_id}: 로컬 학습 시작 (기반 모델 버전: {self.local_model.version}).")
        """
        로컬 모델 학습을 비동기적으로 실행합니다.
        세마포를 사용하여 동시 학습 수를 제어합니다.
        """
        # 세마포 획득을 시도합니다. 획득할 수 없으면 다른 위성이 끝날 때까지 여기서 대기합니다.
        # self.logger.info(f"  🕒 SAT {self.sat_id}: GPU 학습 슬롯 대기 중...")
        self.state = 'TRAINING'
        self.logger.info(f"  ✅ SAT {self.sat_id}: 학습 워커에 의해 로컬 학습 시작 (v{self.local_model.version}).")
        
        new_state_dict = None
        try:
            # 현재 실행중인 이벤트 루프를 가져옵니다.
            loop = asyncio.get_running_loop()
            new_state_dict, accuracy, loss = await loop.run_in_executor(None, self._train_and_eval)
            # new_state_dict = await loop.run_in_executor(None, self._blocking_train)
            # self.local_model.model_state_dict = new_state_dict
            # self.logger.info(f"  🧠 SAT {self.sat_id}: 로컬 학습 완료 ({LOCAL_EPOCHS} 에포크). 검증 시작...")

            # # 2. 검증 또한 별도 스레드에서 실행하여 이벤트 루프 블로킹 방지
            # accuracy, loss = await loop.run_in_executor(
            #     None, 
            #     evaluate_model, 
            #     new_state_dict, 
            #     self.val_loader, 
            #     self.device
            # )
            self.local_model.model_state_dict = new_state_dict
            self.logger.info(f"  📊 [Local Validation] SAT: {self.sat_id}, Version: {self.local_model.version}, Accuracy: {accuracy:.2f}%, Loss: {loss:.4f}")
            self.perf_logger.info(f"{datetime.now(KST).isoformat()},LOCAL_VALIDATION,{self.sat_id},{self.local_model.version},N/A,{accuracy:.4f},{loss:.6f}")

            self.local_model.trained_by = [self.sat_id]
            self.model_ready_to_upload = True

        except Exception as e:
            self.logger.error(f"  💀 SAT {self.sat_id}: 학습 또는 검증 중 에러 발생 - {e}", exc_info=True)

        finally:
            # 성공하든 실패하든 상태를 IDLE로 되돌립니다.
            self.state = 'IDLE'
            self.logger.info(f"  🏁 SAT {self.sat_id}: 학습 절차 완료.")

        # self.logger.info(f"  🏁 SAT {self.sat_id}: GPU 슬롯 반납.")

        #     if self.state == 'TRAINING':
        #         self.logger.info(f"  🧠 SAT {self.sat_id}: 로컬 학습 완료 ({LOCAL_EPOCHS} 에포크).")
        #         accuracy, loss = evaluate_model(self.local_model.model_state_dict, self.val_loader, self.device)
        #         self.logger.info(f"  📊 [Local Validation] SAT: {self.sat_id}, Version: {self.local_model.version}, Accuracy: {accuracy:.2f}%, Loss: {loss:.4f}")
        #         self.perf_logger.info(f"{datetime.now(KST).isoformat()},LOCAL_VALIDATION,{self.sat_id},{self.local_model.version},N/A,{accuracy:.4f},{loss:.6f}")

        #         self.local_model.trained_by = [self.sat_id]
        #         self.model_ready_to_upload = True
        #         self.state = 'IDLE'
        #     else:
        #         self.logger.warning(f"  ⚠️ SAT {self.sat_id}: 학습 중단 (상태 변경: {self.state})")
        # # `async with` 블록이 끝나면 세마포는 자동으로 해제됩니다.
        # self.logger.info(f"  🏁 SAT {self.sat_id}: GPU 슬롯 반납.")

        #     temp_model = create_mobilenet()
        #     temp_model.load_state_dict(self.local_model.model_state_dict)
        #     temp_model.to(self.device)
        #     temp_model.train()
        #     criterion = nn.CrossEntropyLoss()
        #     # 개선 제안: 옵티마이저를 SGD에서 Adam으로 변경
        #     optimizer = torch.optim.Adam(temp_model.parameters(), lr=3e-4, weight_decay=1e-4)
            
        #     for epoch in range(LOCAL_EPOCHS):
        #         for images, labels in self.train_loader:
        #             images, labels = images.to(self.device), labels.to(self.device)
        #             optimizer.zero_grad()
        #             outputs = temp_model(images)
        #             loss = criterion(outputs, labels)
        #             loss.backward()
        #             optimizer.step()
            
        #     self.local_model.model_state_dict = temp_model.state_dict()
        # except Exception as e:
        #     self.logger.error(f"  💀 SAT {self.sat_id}: 학습 중 에러 발생 - {e}", exc_info=True)
        #     self.state = 'IDLE'
        #     return

        # if self.state == 'TRAINING':
        #     self.logger.info(f"  🧠 SAT {self.sat_id}: 로컬 학습 완료 ({LOCAL_EPOCHS} 에포크).")
        #     accuracy, loss = evaluate_model(self.local_model.model_state_dict, self.val_loader, self.device)
        #     self.logger.info(f"  📊 [Local Validation] SAT: {self.sat_id}, Version: {self.local_model.version}, Accuracy: {accuracy:.2f}%, Loss: {loss:.4f}")
        #     self.perf_logger.info(f"{datetime.now(KST).isoformat()},LOCAL_VALIDATION,{self.sat_id},{self.local_model.version},N/A,{accuracy:.4f},{loss:.6f}")

        #     self.local_model.trained_by = [self.sat_id]
        #     self.model_ready_to_upload = True
        #     self.state = 'IDLE'
        # else:
        #     self.logger.warning(f"  ⚠️ SAT {self.sat_id}: 학습 중단 (상태 변경: {self.state})")
    
    def get_distance_to(self, other_sat: 'Satellite') -> float:
        """다른 위성과의 거리를 계산"""
        current_ts = self.clock.get_time_ts()
        return (self.satellite_obj - other_sat.satellite_obj).at(current_ts).distance().km

class WorkerSatellite(Satellite):
    """
    실제 로컬 학습을 수행하는 워커 위성.
    자율적으로 IoT 클러스터 상공을 감지하고 학습을 시작합니다.
    """
    def __init__(self, *args, master: 'MasterSatellite', **kwargs):
        super().__init__(*args, **kwargs)
        self.master = master
        self.logger.info(f"WorkerSAT {self.sat_id} 생성, Master: {master.sat_id}. 초기 모델 버전: {self.local_model.version}")

    async def run(self):
        self.logger.info(f"WorkerSAT {self.sat_id} 임무 시작.")
        asyncio.create_task(self._propagate_orbit())
        # await self.monitor_iot_and_train()
        await self.monitor_and_queue_training()

    async def monitor_and_queue_training(self):
        """
        주기적으로 IoT 클러스터 상공을 감시하고, 조건 충족 시 학습 '요청'을 큐에 넣습니다.
        """
        while True:
            if self.state == 'IDLE' and not self.model_ready_to_upload:
                current_ts = self.clock.get_time_ts()
                for iot in self.iot_clusters:
                    elevation = (self.satellite_obj - iot.topos).at(current_ts).altaz()[0].degrees
                    if elevation >= IOT_FLYOVER_THRESHOLD_DEG:
                        try:
                            # 큐에 작업을 넣습니다. 큐가 꽉 찼으면 QueueFull 예외가 발생합니다.
                            self.training_queue.put_nowait(self.sat_id)
                            self.logger.info(f"🛰️  WorkerSAT {self.sat_id}: '{iot.name}' 상공 통과. 학습 요청을 큐에 추가. (현재 큐 크기: {self.training_queue.qsize()})")
                            # 한번 큐에 넣으면 상태를 변경하여 중복 요청을 방지합니다.
                            self.state = 'WAITING_TRAINING'
                        except asyncio.QueueFull:
                            self.logger.warning(f"🛰️  WorkerSAT {self.sat_id}: '{iot.name}' 상공 통과. 학습 큐가 꽉 차 요청을 무시합니다.")
                        break 
            await asyncio.sleep(1)

    # async def monitor_iot_and_train(self):
    #     """주기적으로 IoT 클러스터 상공을 감시하고, 조건 충족 시 학습 시작"""
    #     while True:
    #         if self.state == 'IDLE' and not self.model_ready_to_upload:
    #             current_ts = self.clock.get_time_ts()
    #             for iot in self.iot_clusters:
    #                 elevation = (self.satellite_obj - iot.topos).at(current_ts).altaz()[0].degrees
    #                 if elevation >= IOT_FLYOVER_THRESHOLD_DEG:
    #                     self.logger.info(f"🛰️  WorkerSAT {self.sat_id}가 IoT 클러스터 '{iot.name}' 상공 통과 (고도각: {elevation:.2f}°). 학습 시작.")
    #                     asyncio.create_task(self.train_local_model())
    #                     break 
    #         # await asyncio.sleep(10)
    #         await asyncio.sleep(0.5)

class MasterSatellite(Satellite):
    """
    클러스터를 관리하는 마스터 위성.
    - 지상국과 통신하며 글로벌 모델을 수신/전송
    - ISL을 통해 워커 위성들에게 모델을 전파하고, 학습된 모델을 수집
    - 수집된 모델들을 주기적으로 취합하여 클러스터 모델 생성
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cluster_members: Dict[int, WorkerSatellite] = {}
        self.cluster_model_buffer: List[PyTorchModel] = []
        self.cluster_version_counter = 0 
        self.logger.info(f"MasterSAT {self.sat_id} 생성. 초기 모델 버전: {self.local_model.version}")

    def add_member(self, worker: WorkerSatellite):
        self.cluster_members[worker.sat_id] = worker

    async def run(self):
        self.logger.info(f"MasterSAT {self.sat_id} 임무 시작.")
        asyncio.create_task(self._propagate_orbit())
        asyncio.create_task(self.manage_cluster_isl())
        asyncio.create_task(self.aggregate_models_periodically())

    async def manage_cluster_isl(self):
        """ISL을 통해 워커 위성들과 통신하고 모델을 교환"""
        while True:
            for worker in self.cluster_members.values():
                distance = self.get_distance_to(worker)
                if distance <= MAX_ISL_DISTANCE_KM:
                    if self.local_model.version > worker.local_model.version or \
                       (self.local_model.version == worker.local_model.version and self.local_model.model_state_dict is not worker.local_model.model_state_dict):
                        await self.send_model_to_worker(worker)
                    if worker.model_ready_to_upload:
                        await self.receive_model_from_worker(worker)
            # await asyncio.sleep(10)
            await asyncio.sleep(1)

    async def receive_global_model(self, model: PyTorchModel):
        """지상국으로부터 글로벌 모델을 수신"""
        if model.version > self.local_model.version:
            self.logger.info(f"  🛰️  MasterSAT {self.sat_id}: 새로운 글로벌 모델 수신 (v{model.version}). 클러스터 버전 리셋.")
            self.cluster_version_counter = 0
            self.local_model = model
            self.model_ready_to_upload = False
        # elif model.version == self.local_model.version and self.local_model.model_state_dict is not model.model_state_dict:
        #      self.logger.info(f"  🛰️  MasterSAT {self.sat_id}: 같은 버전의 업데이트된 글로벌 모델 수신 (v{model.version}).")
        #      self.local_model = model
        #      self.model_ready_to_upload = False


    # async def send_model_to_worker(self, worker: WorkerSatellite):
    #     self.logger.info(f"  🛰️ -> 🛰️  Master {self.sat_id} -> Worker {worker.sat_id}: 모델 전송 (버전 {self.local_model.version})")
    #     worker.local_model = self.local_model

    async def send_model_to_worker(self, worker: WorkerSatellite):
        self.logger.info(f"  🛰️ -> 🛰️  Master {self.sat_id} -> Worker {worker.sat_id}: 모델 전송 (버전 {self.local_model.version})")
        worker.local_model = self.local_model
        # 모델을 받은 워커는 다시 학습할 준비가 된 것이므로 IDLE 상태로 변경
        if worker.state == 'WAITING_TRAINING':
            worker.state = 'IDLE'
    
    async def receive_model_from_worker(self, worker: WorkerSatellite):
        self.cluster_model_buffer.append(worker.local_model)
        worker.model_ready_to_upload = False
        self.logger.info(f"  📥 MasterSAT {self.sat_id}: Worker {worker.sat_id} 모델 수신. (버퍼 크기: {len(self.cluster_model_buffer)})")

    async def aggregate_models_periodically(self):
        """주기적으로 버퍼에 쌓인 워커 모델들을 취합"""
        while True:
            # await asyncio.sleep(30)
            await asyncio.sleep(2)
            if not self.cluster_model_buffer:
                continue
            await self._aggregate_and_evaluate_cluster_models()

    async def _aggregate_and_evaluate_cluster_models(self):
        """실제 모델 취합 및 평가 로직"""
        self.logger.info(f"  ✨ [Cluster Aggregation] Master {self.sat_id}: {len(self.cluster_model_buffer)}개 워커 모델과 기존 클러스터 모델 취합 시작")
        
        state_dicts_to_avg = [self.local_model.model_state_dict] + [m.model_state_dict for m in self.cluster_model_buffer]
        new_state_dict = fed_avg(state_dicts_to_avg)
        all_contributors = list(set(self.local_model.trained_by + [p for model in self.cluster_model_buffer for p in model.trained_by]))
        
        self.local_model.model_state_dict = new_state_dict
        self.local_model.trained_by = all_contributors
        self.model_ready_to_upload = True
        self.cluster_version_counter += 1
        self.logger.info(f"  ✨ [Cluster Aggregation] Master {self.sat_id}: 클러스터 모델 업데이트 완료. 학습자: {self.local_model.trained_by}")

        # 평가도 블로킹 작업이므로 executor에서 실행
        accuracy, loss = await asyncio.get_running_loop().run_in_executor(
            None, evaluate_model, new_state_dict, self.test_loader, self.device
        )
        self.logger.info(f"  🧪 [Cluster Test] Owner: SAT_{self.sat_id}, Global Ver: {self.local_model.version}, Cluster Ver: {self.cluster_version_counter}, Accuracy: {accuracy:.2f}%, Loss: {loss:.4f}")
        self.perf_logger.info(f"{datetime.now(KST).isoformat()},CLUSTER_TEST,SAT_{self.sat_id},{self.local_model.version},{self.cluster_version_counter},{accuracy:.4f},{loss:.6f}")

        self.cluster_model_buffer.clear()

    async def send_local_model(self) -> PyTorchModel | None:
        if self.model_ready_to_upload:
            self.model_ready_to_upload = False
            return self.local_model
        return None
