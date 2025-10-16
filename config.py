# 시뮬레이션 전반에 사용되는 설정값들을 정의하는 파일입니다.

# --- Satellite Constellation Settings ---
# TLE 파일에 정의된 전체 위성군 내에서 마스터 위성의 수와 각 궤도면(plane)의 위성 수를 정의합니다.
NUM_MASTERS = 10      # 마스터 위성의 총 개수
SATS_PER_PLANE = 20   # 하나의 궤도면에 포함된 위성의 수

# --- Satellite Local Training Settings ---
LOCAL_EPOCHS = 1              # 각 위성이 로컬 학습을 수행할 에포크 수
MAX_ISL_DISTANCE_KM = 2500    # 위성 간 통신(ISL)이 가능한 최대 거리 (km)

# --- Ground Station & IoT Settings ---
IOT_FLYOVER_THRESHOLD_DEG = 30.0  # 워커 위성이 이 고도각 이상으로 IoT 클러스터 상공을 통과할 때 학습을 시작합니다.
GS_FLYOVER_THRESHOLD_DEG = 10.0   # 마스터 위성이 이 고도각 이상으로 지상국 상공을 통과할 때 통신을 시작합니다.

# --- Federated Learning Aggregation Policy (Ground Station) ---
MIN_MODELS_FOR_AGGREGATION = 2      # 지상국이 글로벌 모델을 생성하기 위해 필요한 최소 클러스터 모델의 수
AGGREGATION_STALENESS_THRESHOLD = 1 # 글로벌 모델 버전과 취합 대상 클러스터 모델 버전 간의 최대 차이 허용치 (Staleness)

# --- Simulation Performance & Stability Settings ---
# GPU 자원 경쟁 및 교착 상태를 방지하기 위해, 한 번에 하나의 위성만 학습을 수행하도록 설정합니다.
MAX_CONCURRENT_TRAINING_SESSIONS = 1 
# 학습 요청을 저장할 큐의 최대 크기. 이 크기를 초과하는 요청은 무시됩니다.
TRAINING_QUEUE_MAX_SIZE = 10

