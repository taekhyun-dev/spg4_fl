# simulation/environment.py
import asyncio
from datetime import datetime
from typing import List, Dict, Tuple

from skyfield.api import Topos

from config import MIN_MODELS_FOR_AGGREGATION, AGGREGATION_STALENESS_THRESHOLD, NUM_MASTERS, SATS_PER_PLANE
from ml.model import PyTorchModel, create_mobilenet
from ml.training import evaluate_model, fed_avg
from utils.logging_setup import KST
from utils.skyfield_utils import EarthSatellite
from simulation.satellite import Satellite, WorkerSatellite, MasterSatellite

class IoTCluster:
    """데이터 소스가 되는 IoT 클러스터를 나타내는 클래스"""
    def __init__(self, name: str, latitude: float, longitude: float, elevation: int, sim_logger=None):
        self.name = name
        self.topos = Topos(latitude_degrees=latitude, longitude_degrees=longitude, elevation_m=elevation)
        self.logger = sim_logger
        self.logger.info(f"IoT 클러스터 '{self.name}' 생성 완료.")

class GroundStation:
    """
    지상국 클래스.
    - 위성과의 통신(AOS/LOS)을 관리
    - 클러스터 모델을 수신하고 글로벌 모델을 취합(Aggregation)
    - 업데이트된 글로벌 모델을 위성에 전파
    """
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

    async def run(self, clock: 'SimulationClock', satellites: Dict[int, 'Satellite']):
        self.logger.info(f"지상국 '{self.name}' 운영 시작.")
        asyncio.create_task(self.periodic_aggregation_task())
        while True:
            current_ts = clock.get_time_ts()
            for sat_id, sat in satellites.items():
                if not hasattr(sat, 'cluster_members'): continue # MasterSatellite만 상대
                
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
                        # 수신 직후 바로 집계 시도하여 모델 즉시 업데이트
                        # await self.try_aggregate_and_update()
                    
                    # 2. 그 다음 송신
                    await self.send_model_to_satellite(sat)

                elif prev_visible and not visible_now: # LOS
                    self.logger.info(f"📡 [LOS] {self.name} <-> MasterSAT {sat_id} 통신 종료 (고도각: {elevation:.2f}°)")
                    sat.state = 'IDLE'
                
                self._comm_status[sat_id] = visible_now

            await asyncio.sleep(clock.real_interval)

    async def send_model_to_satellite(self, satellite: 'MasterSatellite'):
        if self.global_model.version > satellite.local_model.version:
            self.logger.info(f"  📤 {self.name} -> MasterSAT {satellite.sat_id}: 글로벌 모델 전송 (버전 {self.global_model.version})")
            await satellite.receive_global_model(self.global_model)

    async def receive_model_from_satellite(self, satellite: 'MasterSatellite'):
        cluster_model = await satellite.send_local_model()
        if cluster_model:
            self.logger.info(f"  📥 {self.name} <- MasterSAT {satellite.sat_id}: 클러스터 모델 수신 완료 (버전 {cluster_model.version}, 학습자: {cluster_model.trained_by})")
            self.received_models_buffer.append(cluster_model)

    async def periodic_aggregation_task(self):
        """주기적으로 Aggregation을 시도하는 백그라운드 작업"""
        while True:
            # self.logger.info(f"🕒 [{self.name}] Aggregation 조건 확인 중...")
            await self.try_aggregate_and_update()
            await asyncio.sleep(5) # 30초 -> 5초로 단축하여 더 자주 확인
            
    async def try_aggregate_and_update(self):
        """Aggregation 조건 확인 및 수행"""
        if len(self.received_models_buffer) < self.min_models_for_aggregation: return
        
        try:
            max_version_in_buffer = max(model.version for model in self.received_models_buffer)
        except ValueError:
            return # 버퍼가 비었을 경우

        if max_version_in_buffer < self.global_model.version: return

        version_lower_bound = max_version_in_buffer - self.staleness_threshold
        models_to_aggregate = [m for m in self.received_models_buffer if m.version >= version_lower_bound]
        
        if len(models_to_aggregate) < self.min_models_for_aggregation: return

        self.logger.info(f"✨ [{self.name} Aggregation] {len(models_to_aggregate)}개 모델(v >= {version_lower_bound})과 기존 글로벌 모델(v{self.global_model.version}) 취합 시작...")
        
        state_dicts_to_avg = [self.global_model.model_state_dict] + [m.model_state_dict for m in models_to_aggregate]
        new_state_dict = fed_avg(state_dicts_to_avg)
        
        new_version = self.global_model.version + 1 # 버전업
        all_contributors = list(set(self.global_model.trained_by + [p for model in models_to_aggregate for p in model.trained_by]))
        self.global_model = PyTorchModel(version=new_version, model_state_dict=new_state_dict, trained_by=all_contributors)
        self.logger.info(f"✨ [{self.name} Aggregation] 새로운 글로벌 모델 생성 완료! (버전 {self.global_model.version})")

        loop = asyncio.get_running_loop()
        accuracy, loss = await loop.run_in_executor(None, evaluate_model, self.global_model.model_state_dict, self.test_loader, self.device)
        self.logger.info(f"  🧪 [Global Test] Owner: {self.name}, Version: {self.global_model.version}, Accuracy: {accuracy:.2f}%, Loss: {loss:.4f}")
        self.perf_logger.info(f"{datetime.now(KST).isoformat()},GLOBAL_TEST,{self.name},{self.global_model.version},N/A,{accuracy:.4f},{loss:.6f}")

        aggregated_model_ids = {id(m) for m in models_to_aggregate}
        self.received_models_buffer = [m for m in self.received_models_buffer if id(m) not in aggregated_model_ids]
        
        if self.received_models_buffer:
            try:
                current_max_version = max(m.version for m in self.received_models_buffer)
                cleanup_lower_bound = current_max_version - self.staleness_threshold
                models_to_discard = [m for m in self.received_models_buffer if m.version < cleanup_lower_bound]
                if models_to_discard:
                    discard_versions = {m.version for m in models_to_discard}
                    self.logger.info(f"  🗑️  [{self.name}] {len(models_to_discard)}개의 오래된 모델 정리 (버전: {discard_versions})")
                    discard_model_ids = {id(m) for m in models_to_discard}
                    self.received_models_buffer = [m for m in self.received_models_buffer if id(m) not in discard_model_ids]
            except ValueError: pass

def create_simulation_environment(
    clock: 'SimulationClock', 
    eval_infra: dict, 
    all_sats_skyfield: Dict[int, EarthSatellite]
) -> Tuple[Dict[int, Satellite], List[GroundStation]]:
    """
    시뮬레이션 환경을 구성하는 모든 객체(위성, 지상국, IoT)를 생성합니다.
    main.py에서 TLE 로딩 후, 이 함수를 호출하여 환경을 구성합니다.
    """
    logger = eval_infra['sim_logger']
    
    # 1. 초기 글로벌 모델 생성
    initial_pytorch_model = create_mobilenet()
    initial_global_model = PyTorchModel(version=0, model_state_dict=initial_pytorch_model.state_dict())
    
    # 2. 지상국 및 IoT 클러스터 생성
    ground_stations = [
        GroundStation("Seoul-GS", 37.5665, 126.9780, 34, initial_model=initial_global_model, eval_infra=eval_infra),
        # GroundStation("Houston-GS", 29.7604, -95.3698, 12, initial_model=initial_global_model, eval_infra=eval_infra)
    ]
    
    iot_clusters = [
        IoTCluster("Amazon_Forest", -3.47, -62.37, 100, sim_logger=logger),
        IoTCluster("Great_Barrier_Reef", -18.29, 147.77, 0, sim_logger=logger),
        IoTCluster("Siberian_Tundra", 68.35, 18.79, 420, sim_logger=logger)
    ]

    # 3. 위성 객체 및 클러스터 구성 (기존 main.py 로직)
    satellites_in_sim: Dict[int, Satellite] = {}
    sat_ids = sorted(list(all_sats_skyfield.keys()))
    
    if len(sat_ids) < NUM_MASTERS * (SATS_PER_PLANE // NUM_MASTERS if NUM_MASTERS > 0 else SATS_PER_PLANE):
       raise ValueError(f"시뮬레이션을 위해 충분한 수의 위성 TLE가 필요합니다.")
       
    master_ids = [sat_ids[i * SATS_PER_PLANE] for i in range(NUM_MASTERS)]
    worker_ids = [sid for sid in sat_ids if sid not in master_ids]
    
    logger.info(f"마스터 위성으로 {master_ids}가 선정되었습니다.")

    masters = []
    for m_id in master_ids:
        master_sat = MasterSatellite(
            m_id, all_sats_skyfield[m_id], clock,
            initial_model=initial_global_model,
            iot_clusters=iot_clusters, eval_infra=eval_infra
        )
        satellites_in_sim[m_id] = master_sat
        masters.append(master_sat)

    for i, w_id in enumerate(worker_ids):
        assigned_master = masters[i % NUM_MASTERS]
        worker_sat = WorkerSatellite(
            w_id, all_sats_skyfield[w_id], clock,
            initial_model=initial_global_model,
            iot_clusters=iot_clusters,
            master=assigned_master, eval_infra=eval_infra
        )
        assigned_master.add_member(worker_sat)
        satellites_in_sim[w_id] = worker_sat
            
    logger.info(f"총 {len(satellites_in_sim)}개 위성 생성 완료. ({len(masters)} Masters, {len(satellites_in_sim) - len(masters)} Workers)")
    
    return satellites_in_sim, ground_stations    