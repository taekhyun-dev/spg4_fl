import asyncio
import torch
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import List, Dict, Coroutine

# 프로젝트 모듈 import
from config import *
from utils.logging_setup import setup_loggers
from utils.skyfield_utils import EarthSatellite
from ml.data import get_cifar10_loaders
from simulation.clock import SimulationClock
from simulation.environment import create_simulation_environment
from simulation.satellite import Satellite


async def training_worker(name: str, queue: asyncio.Queue, satellites: Dict[int, Satellite], sim_logger):
    """
    작업 큐에서 위성 ID를 가져와 해당 위성의 학습을 실행하는 '소비자' 역할을 합니다.
    """
    sim_logger.info(f"💪 학습 워커 '{name}' 시작.")
    while True:
        # 큐에서 작업(위성 ID)을 기다립니다.
        sat_id = await queue.get()
        
        sim_logger.info(f"💪 학습 워커 '{name}': SAT-{sat_id}의 학습 작업 시작.")
        satellite = satellites.get(sat_id)
        if satellite:
            try:
                await satellite.train_local_model()
            except Exception as e:
                sim_logger.error(f"💪 학습 워커 '{name}'가 SAT {sat_id} 학습 중 에러 발생: {e}", exc_info=True)
        
        # 큐에 작업이 완료되었음을 알립니다. (선택적이지만 좋은 습관)
        queue.task_done()

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

    # --- 학습 제어를 위한 크기 제한 큐 생성 ---
    training_queue = asyncio.Queue(maxsize=TRAINING_QUEUE_MAX_SIZE)
    sim_logger.info(f"최대 {TRAINING_QUEUE_MAX_SIZE}개의 작업을 담을 수 있는 학습 대기열을 생성합니다.")

    # --- 동시 학습 제어를 위한 Semaphore 생성 ---
    # GPU 메모리 고갈을 방지하기 위해 동시에 실행될 수 있는 최대 학습 작업 수를 제한합니다.
    # training_semaphore = asyncio.Semaphore(MAX_CONCURRENT_TRAINING_SESSIONS)
    # sim_logger.info(f"동시 학습 가능한 최대 위성 수를 {MAX_CONCURRENT_TRAINING_SESSIONS}(으)로 제한합니다.")

    sim_logger.info("Loading CIFAR10 dataset...")
    train_loader, val_loader, test_loader = get_cifar10_loaders()
    sim_logger.info("Dataset loaded.")

    eval_infra = {
        "sim_logger": sim_logger,
        "perf_logger": perf_logger,
        "train_loader": train_loader,
        "val_loader": val_loader,
        "test_loader": test_loader,
        "device": device,
        "training_queue": training_queue, # 모든 위성이 공유할 작업 큐
    }

    start_time = datetime.now(timezone.utc)
    simulation_clock = SimulationClock(
        start_dt=start_time, 
        time_step=timedelta(minutes=10),
        real_interval=1.0,
        sim_logger=sim_logger
    )

    # 1. TLE 데이터 로드
    all_sats_skyfield = load_constellation("constellation.tle", sim_logger)

    # 2. 로드된 데이터를 전달하여 시뮬레이션 환경 구성
    sim_logger.info("시뮬레이션 환경을 구성합니다...")
    satellites, ground_stations = create_simulation_environment(
        simulation_clock, eval_infra, all_sats_skyfield
    )
    

    # satellites 리스트 출력
    print("Loaded Satellites:")
    for sat in satellites:
        print(f"  - SAT TYPE: {type(sat)}")

    # satellites_in_sim: Dict[int, Satellite] = {s.sat_id: s for s in satellites}
    sim_logger.info("환경 구성 완료.")

    # --- 학습 워커(소비자) 태스크 생성 ---
    # --- 시뮬레이션 태스크 생성 ---
    # 1. 학습 워커 태스크 (소비자) - GPU 안정성을 위해 단 하나만 생성
    # config.py의 MAX_CONCURRENT_TRAINING_SESSIONS는 이제 사용되지 않습니다.
    training_worker_task = asyncio.create_task(
        training_worker("GPU-Worker", training_queue, satellites, sim_logger)
    )

    # 2. 시뮬레이션 메인 태스크들
    sim_tasks: List[Coroutine] = [
        simulation_clock.run(),
        *[gs.run(simulation_clock, satellites) for gs in ground_stations],
        *[sat.run() for sat in satellites.values()]
    ]
    sim_logger.info("시뮬레이션을 시작합니다.")
    await asyncio.gather(*[asyncio.create_task(task) for task in sim_tasks])

if __name__ == "__main__":
    try:
        # 멀티프로세싱 시작 방식 설정 (macOS, Windows 호환성)
        torch.multiprocessing.set_start_method('spawn', force=True)
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n시뮬레이션을 종료합니다.")
    except FileNotFoundError as e:
        print(e)
    except Exception as e:
        # 예기치 않은 에러 발생 시 로깅
        sim_logger, _ = setup_loggers()
        sim_logger.error(f"\n시뮬레이션 중 치명적인 에러 발생: {e}", exc_info=True)

