#  Blade Defect Detection System (풍력 발전기 블레이드 마모 탐지)

## 📖 Overview
본 프로젝트는 풍력 발전기 블레이드(Blade) 표면의 마모 및 결함을 자동으로 탐지하는 딥러닝 모델 개발 프로젝트입니다.
최신 객체 탐지 모델인 **YOLO v11**을 기반으로 하며, 데이터 라벨링의 효율성과 정확도를 높이기 위해 **SAM 2 (Segment Anything Model 2)**를 활용한 반자동 라벨링 파이프라인을 구축했습니다.

- **주요 목표:** 블레이드 결함의 위치 및 형태(Oriented Bounding Box) 정밀 탐지
  <img width="2898" height="1524" alt="image" src="https://github.com/user-attachments/assets/e11bee2b-3677-463e-8ab2-b937fc9ae956" />


## 🚀 Key Features

### 1. Advanced Data Pipeline with SAM 2
- **SAM 2 활용:** Meta의 Segment Anything Model 2를 사용하여 블레이드 결함 부위를 정밀하게 세그먼테이션(Segmentation).
- **OBB(Oriented Bounding Box) 변환:** 세그먼테이션 마스크를 회전된 바운딩 박스로 변환하여, 길고 기울어진 블레이드 결함의 형상 정보를 정확히 반영.

### 2. Object Detection Model
- **YOLO Fine-tuning:** YOLO 아키텍처를 기반으로 커스텀 데이터셋을 학습시켜 탐지 성능 극대화.
- **Robustness:** 다양한 배경과 결함 형태에 대해 강건한 탐지 성능 확보.

## 🛠️ Model Architecture
본 모델은 **Pretrained YOLO**을 Backbone으로 사용하며, 다음과 같은 구조를 가집니다.

| Component | Description |
| --- | --- |
| **Backbone** | 이미지 특징 추출 (Conv, C3K2 Layers) |
| **Neck** | Multi-scale Feature Fusion (Upsample, Concat, C3K2) |
| **Head** | 최종 객체 탐지 및 클래스 분류 (Detect Head) |

## 📊 Performance Analysis
학습된 모델은 검증 데이터셋에서 매우 우수한 성능을 보였습니다.

- **Precision & Recall:** 1.0에 근접한 높은 수치 기록.
- **mAP50:** **0.99+** (매우 높은 탐지 정확도).
- **F1 Score:** Confidence Threshold 0.308 기준 **1.00** 달성.
- **Confusion Matrix:**
    - True Positive: 1,525건
    - False Positive: 2건 (오탐률 극소화)

## 💻 Tech Stack
- **Language:** Python 3.9+
- **Deep Learning Framework:** PyTorch
- **Models:**
    - [Ultralytics YOLO](https://github.com/ultralytics/ultralytics)
    - [Meta SAM 2](https://github.com/facebookresearch/segment-anything-2)
- **Tools:** OpenCV, NumPy, Matplotlib

## 🚀 How to Run

본 프로젝트는 두 단계로 실행됩니다:
1.  **데이터셋 생성 (`draw_obb_video.py`):** 원본 비디오와 JSON 라벨을 처리하여 YOLO OBB 포맷의 데이터셋을 생성합니다.
2.  **모델 학습 (`train.py`):** 생성된 데이터셋을 사용하여 YOLO v11 모델을 학습시킵니다.

### 1. Environment Setup
필요한 라이브러리를 설치합니다.
```bash
# 가상환경 생성 및 활성화 (선택 사항)
conda create -n capstone_env python=3.9
conda activate capstone_env

# 의존성 패키지 설치
pip install ultralytics opencv-python numpy pycocotools pyyaml

yolo detect predict model=runs/detect/train/weights/best.pt source='path/to/video_or_image'
📝 Result Preview
```

### 2. Prepare Data
프로젝트 루트 디렉토리에 원본 비디오 파일(.mp4)과 라벨링 JSON 파일(.json)을 준비합니다.

- **Video Files**: 251119 1-4 #11.mp4, 251119 1-4 #14.mp4, ...

- **JSON Files**: 1_4_11.json, 1_4_14.json, ... (파일 경로나 이름이 다를 경우 draw_obb_video.py 내부의 VIDEO_JSON_PAIRS 리스트를 수정하세요.)

### 3.Generate Dataset
draw_obb_video.py를 실행하여 비디오 프레임별 이미지와 YOLO OBB 라벨 텍스트 파일을 생성합니다.
```
# --process-all 옵션으로 설정된 모든 비디오/JSON 쌍 처리
# --save-videos 옵션 추가 시 시각화된 결과 비디오도 함께 저장됨
python draw_obb_video.py --process-all --save-videos
```

### 실행 결과:
- yolo_obb_dataset/ 폴더가 생성됩니다.
- images/train/: 추출된 프레임 이미지들
- labels/train/: 각 이미지에 대응하는 YOLO OBB 라벨 파일들 (.txt)
### 4.Train Model
데이터셋 준비가 완료되면 train.py를 실행하여 모델 학습을 시작합니다. 이 스크립트는 자동으로 데이터셋을 train/val/test로 나누고, 설정 파일(obb_dataset.yaml)을 생성한 뒤 학습을 진행합니다.
```
# 기본 설정으로 학습 실행 (Epochs: 50, Image Size: 640)
python train.py

# (선택) 하이퍼파라미터 변경하여 실행 예시
python train.py --epochs 100 --imgsz 1024 --batch 8 --model yolo11s-obb.pt
```
### 학습 결과:
- runs/obb/train/ 디렉토리에 학습 로그, 가중치(best.pt, last.pt), 결과 그래프 등이 저장됩니다.
- yolo_obb_dataset/obb_dataset.yaml: 생성된 데이터셋 설정 파일.

  
