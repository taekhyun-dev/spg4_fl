import asyncio
import torch
import torch.nn as nn
import torch.optim.lr_scheduler as lr_scheduler
from datetime import datetime
from typing import List, Tuple, Dict
from ml.model import PyTorchModel, create_mobilenet
from ml.training import evaluate_model
from utils.skyfield_utils import EarthSatellite
from utils.logging_setup import KST
from config import LOCAL_EPOCHS, IOT_FLYOVER_THRESHOLD_DEG

# ----- CLASS DEFINITION ----- #
class Satellite:
    def __init__ (self, sat_id: int, satellite_obj: EarthSatellite, clock: 'SimulationClock', sim_logger, perf_logger,
                   iot_clusters: List['IoTCluster'], initial_model: PyTorchModel, train_loader, val_loader):
        self.sat_id = sat_id
        self.satellite_obj = satellite_obj
        self.clock = clock
        self.logger = sim_logger
        self.perf_logger = perf_logger
        self.position = {"lat": 0.0, "lon": 0.0, "alt": 0.0}
        self.state = "IDLE"
        self.iot_clusters = iot_clusters
        self.local_model = initial_model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.global_model = initial_model
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.logger.info(f"SAT {self.sat_id} 생성. 초기 모델 버전: {self.local_model.version}")
    
    async def run(self):
        self.logger.info(f"SAT {self.sat_id} 임무 시작.")
        asyncio.create_task(self._propagate_orbit())

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
            self.logger.info(f"    - SAT {self.sat_id}: 에포크 {epoch+1}/{LOCAL_EPOCHS} 진행 중...")
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

    async def train_and_eval(self):
        """CIFAR10 데이터셋으로 로컬 모델을 학습하고 검증"""
        self.state = 'TRAINING'
        self.logger.info(f"  ✅ SAT {self.sat_id}: 로컬 학습 시작 (v{self.local_model.version}).")
        new_state_dict = None
        try:
            # 현재 실행중인 이벤트 루프를 가져옵니다.
            loop = asyncio.get_running_loop()
            new_state_dict, accuracy, loss = await loop.run_in_executor(None, self._train_and_eval)
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

    async def monitoring(self):
        """
        기능
            - IoT와 통신 가능 여부 확인
            - Local Update 진행
            - IoT에게 모델 전송
        """
        while True:
            await asyncio.sleep(0.1)  # 코루틴 양보 (tight loop 방지)
            if self.state == 'IDLE' and not self.model_ready_to_upload:
                current_ts = self.clock.get_time_ts()
                for iot in self.iot_clusters:
                    elevation = (self.satellite_obj - iot.topos).at(current_ts).altaz()[0].degrees
                    #  통신 가능 시점
                    if elevation >= IOT_FLYOVER_THRESHOLD_DEG:
                        # IoT에게 모델 전송 - I/O이므로 코루틴 태스크로 비동기 발사
                        asyncio.create_task(self.send_model_to_iot(iot))
                        # Local Update 진행 - CPU 작업이므로 프로세스 풀로 오프로딩
                        if not self._local_update_in_flight:
                            asyncio.create_task(self.train_and_eval())

    async def send_model_to_iot(self, iot: 'IoT'):
        if self.global_model.version > iot.global_model.version:
            self.logger.info(f"  🛰️ SAT {self.sat_id} -> IoT {iot.name}: 글로벌 모델 전송 (버전 {self.global_model.version})")
            await iot.receive_global_model(self.global_model)
                        