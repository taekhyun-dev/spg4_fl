import asyncio
import torch
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import List, Dict

# 프로젝트 모듈 import
from config import *
from utils.logging_setup import setup_loggers
from utils.skyfield_utils import EarthSatellite
from ml.data import get_cifar10_loaders
from ml.model import PyTorchModel, create_mobilenet
from simulation.clock import SimulationClock
from simulation.environment import GroundStation, IoTCluster
from simulation.satellite import Satellite, MasterSatellite, WorkerSatellite

def load_constellation(tle_path: str, sim_logger) -> Dict[int, EarthSatellite]:
    """TLE 파일에서 위성군 정보를 불러오는 함수"""
    if not Path(tle_path).exists(): raise FileNotFoundError(f"'{tle_path}' 파일을 찾을 수 없습니다.")
    satellites = {}
    with open(tle_path, "r") as f:
        lines = [line.strip() for line in f.readlines()]; i = 0
        while i < len(lines):
            name, line1, line2 = lines[i:i+3]; sat_id = int(name.replace("SAT", ""))
            satellites[sat_id] = EarthSatellite(line1, line2, name)
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
        time_step=timedelta(minutes=10),
        real_interval=1.0,
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
    sat_ids = sorted(list(all_sats_skyfield.keys()))
    
    if len(sat_ids) < NUM_MASTERS * (SATS_PER_PLANE / NUM_MASTERS):
        raise ValueError(f"시뮬레이션을 위해 충분한 수의 위성 TLE가 필요합니다.")
        
    master_ids = [sat_ids[i * SATS_PER_PLANE] for i in range(NUM_MASTERS)]
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
        assigned_master = masters[i % NUM_MASTERS]
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
