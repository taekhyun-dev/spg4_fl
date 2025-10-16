from skyfield.api import Topos
from typing import List, Dict, Tuple
from ml.model import PyTorchModel, create_mobilenet

# ----- CLASS DEFINITION ----- #
class IoT:
    def __init__ (self, name, sim_logger, latitude, longitude, elevation, initial_model: PyTorchModel, test_loader):
        self.name = name
        self.logger = sim_logger
        self.topos = Topos(latitude_degrees=latitude, longitude_degrees=longitude, elevation_m=elevation)
        self.global_model = initial_model
        self.test_loader = test_loader
        self.logger.info(f"IoT 클러스터 '{self.name}' 생성 완료.")
    
    async def receive_global_model(self, model: PyTorchModel):
        """위성으로부터 글로벌 모델을 수신"""
        if model.version > self.global_model.version:
            self.logger.info(f"  📡  IoT {self.name}: 새로운 글로벌 모델 수신 (v{model.version}).")
            self.global_model = model

class GroundStation:
    def __init__ (self, name, latitude, longitude, elevation, sim_logger, threshold_deg, staleness_threshold,
                  initial_model: PyTorchModel, test_loader, perf_logger):
        self.name = name
        self.logger = sim_logger
        self.topos = Topos(latitude_degrees=latitude, longitude_degrees=longitude, elevation_m=elevation)
        self.threshold_deg = threshold_deg
        self._comm_status: Dict[int, bool] = {}
        self.staleness_threshold = staleness_threshold
        self.global_model = initial_model
        self.test_loader = test_loader
        self.perf_logger = perf_logger
        self.logger.info(f"지상국 '{self.name}' 생성 완료. 글로벌 모델 버전: {self.global_model.version}")
        self.logger.info(f"  - Aggregation 정책: 버전 허용치 {self.staleness_threshold}")


def create_minumum_simulation_environment(
    clock: 'SimulationClock', 
    eval_infra: dict, 
    all_sats_skyfield: Dict[int, EarthSatellite]
) -> Tuple[Dict[int, Satellite], List[GroundStation]]: