# config.py
"""
시뮬레이션의 모든 설정값을 관리하는 파일입니다.
이 파일의 값을 변경하여 다양한 시나리오를 실험할 수 있습니다.
"""

# --- 통신 및 궤도 관련 상수 ---
MAX_ISL_DISTANCE_KM = 2000          # 위성 간 통신(ISL)이 가능한 최대 거리 (km)
IOT_FLYOVER_THRESHOLD_DEG = 30.0    # 위성이 IoT 클러스터 상공을 통과했다고 판단하는 최소 고도각 (degrees)
GROUND_STATION_THRESHOLD_DEG = 10.0 # 지상국과 위성이 통신 가능하다고 판단하는 최소 고도각 (degrees)

# --- 연합학습 관련 상수 ---
MIN_MODELS_FOR_AGGREGATION = 2      # 글로벌 Aggregation을 시작하기 위해 필요한 최소 클러스터 모델 수
AGGREGATION_STALENESS_THRESHOLD = 1 # 글로벌 Aggregation 시 허용하는 최대 모델 버전 차이 (Staleness)
LOCAL_EPOCHS = 3                    # 각 위성이 로컬에서 학습을 수행할 에포크 수

# --- 클러스터링 관련 상수 ---
NUM_MASTERS = 10                    # 전체 위성군에서 마스터 위성의 수
SATS_PER_PLANE = 20                 # 하나의 궤도면에 포함된 위성의 수
