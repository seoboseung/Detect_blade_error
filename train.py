#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
train.py

- draw_obb_video.py로 만든 yolo_obb_dataset을 이용해서
  YOLO OBB(Oriented Bounding Box) 모델을 학습하는 스크립트.

기능 요약
1) yolo_obb_dataset/images/train 안에 있는 모든 프레임 이미지 목록 수집
2) 같은 이름의 labels/train/*.txt가 있는 것만 사용
3) (기본) 8:1:1 비율로 train/val/test split (이미지는 한 폴더에 두고, txt 리스트로만 구분)
4) yolo_obb_dataset/obb_dataset.yaml 자동 생성
5) Ultralytics YOLO OBB 모델(yolo11n-obb.pt)을 사용해 학습 수행

사용 방법 (Capstone/labeling_auto에서 실행한다고 가정):
    (capstone_env) python train.py
또는:
    (capstone_env) python train.py --dataset-root yolo_obb_dataset --epochs 100 --imgsz 640
"""

import os
import argparse
import random
from pathlib import Path
from typing import List, Tuple

import yaml  # pip install pyyaml 필요
from ultralytics import YOLO  # pip install ultralytics 필요


# =========================
# 1. 데이터 split 함수
# =========================
def collect_image_label_pairs(dataset_root: Path) -> List[Tuple[Path, Path]]:
    """
    yolo_obb_dataset/images/train 아래의 모든 이미지와
    yolo_obb_dataset/labels/train 아래의 라벨을 쌍으로 모은다.

    반환: [(image_path, label_path), ...]
    """
    img_dir = dataset_root / "images" / "train"
    lbl_dir = dataset_root / "labels" / "train"

    if not img_dir.exists():
        raise FileNotFoundError(f"[ERROR] 이미지 디렉토리 없음: {img_dir}")
    if not lbl_dir.exists():
        raise FileNotFoundError(f"[ERROR] 라벨 디렉토리 없음: {lbl_dir}")

    # jpg, png 모두 지원 (실제로는 draw_obb_video.py에서 jpg로 저장했음)
    img_exts = [".jpg", ".jpeg", ".png"]
    all_imgs = []
    for ext in img_exts:
        all_imgs.extend(img_dir.rglob(f"*{ext}"))

    pairs = []
    missing_labels = 0

    for img_path in all_imgs:
        stem = img_path.stem  # 파일명 (확장자 제외)
        lbl_path = lbl_dir / f"{stem}.txt"
        if lbl_path.exists():
            pairs.append((img_path, lbl_path))
        else:
            missing_labels += 1

    print(f"[INFO] 총 이미지 수: {len(all_imgs)}")
    print(f"[INFO] 라벨이 있는 이미지 수: {len(pairs)}")
    if missing_labels > 0:
        print(f"[WARN] 라벨 없는 이미지 수: {missing_labels} (학습에서는 제외됨)")

    if len(pairs) == 0:
        raise RuntimeError("[ERROR] 라벨이 있는 이미지가 하나도 없습니다. draw_obb_video.py가 제대로 돌았는지 확인하세요.")

    return pairs

def split_dataset(
    pairs: List[Tuple[Path, Path]],
    train_ratio: float = 0.7,
    val_ratio: float = 0.2,
    seed: int = 42,
) -> Tuple[List[Path], List[Path], List[Path]]:
    """
    (이미지, 라벨) 쌍 리스트를 받아 train/val/test로 나눈다.
    여기선 실제 파일 이동은 하지 않고, '이미지 경로 리스트'만 나눈다.
    라벨은 YOLO 내부에서 이미지 경로를 기반으로 자동으로 찾는다.

    반환: (train_imgs, val_imgs, test_imgs)
    """
    random.seed(seed)
    random.shuffle(pairs)

    n = len(pairs)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    # 나머지는 test
    n_test = n - n_train - n_val

    train_pairs = pairs[:n_train]
    val_pairs = pairs[n_train:n_train + n_val]
    test_pairs = pairs[n_train + n_val:]

    train_imgs = [p[0] for p in train_pairs]
    val_imgs = [p[0] for p in val_pairs]
    test_imgs = [p[0] for p in test_pairs]

    print(f"[INFO] Split 결과: train={len(train_imgs)}, val={len(val_imgs)}, test={len(test_imgs)}")
    if len(test_imgs) == 0:
        print("[WARN] test가 0장입니다. 작은 데이터셋이라면 괜찮지만, 필요하다면 비율을 조정하세요.")

    return train_imgs, val_imgs, test_imgs

def write_split_txt(
    dataset_root: Path,
    train_imgs: List[Path],
    val_imgs: List[Path],
    test_imgs: List[Path],
) -> Tuple[Path, Path, Path]:
    """
    train.txt / val.txt / test.txt를 dataset_root 바로 아래에 생성한다.
    내용은 '이미지의 절대 경로'로 작성한다.
    (Ultralytics는 txt 모드에서는 path: 를 기준으로 붙이지 않고,
     각 줄을 그대로 경로로 사용하기 때문.)
    """
    def to_abs(p: Path) -> str:
        # 플랫폼 상관 없이 절대 경로 문자열로
        return str(p.resolve())

    train_txt = dataset_root / "train.txt"
    val_txt = dataset_root / "val.txt"
    test_txt = dataset_root / "test.txt"

    with train_txt.open("w", encoding="utf-8") as f:
        for img in train_imgs:
            f.write(to_abs(img) + "\n")

    with val_txt.open("w", encoding="utf-8") as f:
        for img in val_imgs:
            f.write(to_abs(img) + "\n")

    with test_txt.open("w", encoding="utf-8") as f:
        for img in test_imgs:
            f.write(to_abs(img) + "\n")

    print(f"[INFO] train.txt  경로: {train_txt}")
    print(f"[INFO] val.txt    경로: {val_txt}")
    print(f"[INFO] test.txt   경로: {test_txt}")

    return train_txt, val_txt, test_txt

# =========================
# 2. YAML 생성 함수
# =========================
def create_data_yaml(
    dataset_root: Path,
    train_txt: Path,
    val_txt: Path,
    test_txt: Path,
    yaml_path: Path,
    class_names: dict,
):
    """
    YOLO OBB 학습용 data.yaml(여기서는 obb_dataset.yaml)을 생성한다.

    - train, val, test: 각각 train.txt, val.txt, test.txt 의 절대 경로
    - names: {class_id: class_name}
    """
    data_cfg = {
        # txt 파일의 "절대 경로"를 직접 넣어준다.
        # txt 안에는 이미지의 절대 경로가 들어있다.
        "train": str(train_txt.resolve()),
        "val": str(val_txt.resolve()),
        "test": str(test_txt.resolve()),
        "names": class_names,
    }

    with yaml_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(data_cfg, f, sort_keys=False, allow_unicode=True)

    print(f"[INFO] 데이터셋 YAML 생성 완료: {yaml_path}")
    print(f"       train : {data_cfg['train']}")
    print(f"       val   : {data_cfg['val']}")
    print(f"       test  : {data_cfg['test']}")
    print(f"       classes: {data_cfg['names']}")

# =========================
# 3. 학습 함수
# =========================
def train_yolo_obb(
    data_yaml: Path,
    model_name: str = "yolo11n-obb.pt",
    epochs: int = 50,
    imgsz: int = 640,
    batch: int = 16,
    device: str = "0",
):
    """
    Ultralytics YOLO OBB 모델을 학습한다.

    - model_name: "yolo11n-obb.pt" (Ultralytics 최신 기준)
      만약 YOLOv8 버전을 쓰고 있다면 "yolov8n-obb.pt" 등으로 바꿔도 됨.
    """
    print(f"[INFO] 모델 로딩: {model_name}")
    model = YOLO(model_name)

    print(f"[INFO] 학습 시작")
    results = model.train(
        data=str(data_yaml),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=device,  # "0" -> GPU 0, "cpu" -> CPU
    )

    print("[INFO] 학습 완료")
    # best.pt, last.pt 등의 weight가 자동으로 runs/obb/exp*/ 에 저장됨
    return results


# =========================
# 4. main
# =========================
def parse_args():
    parser = argparse.ArgumentParser(description="Train YOLO OBB model on yolo_obb_dataset")
    parser.add_argument(
        "--dataset-root",
        type=str,
        default="yolo_obb_dataset",
        help="draw_obb_video.py에서 생성한 데이터셋 루트 경로",
    )
    parser.add_argument(
        "--yaml-name",
        type=str,
        default="obb_dataset.yaml",
        help="생성할 데이터셋 yaml 파일 이름 (dataset-root 아래에 생성)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="yolo11n-obb.pt",
        help="Ultralytics OBB 모델 가중치 (예: yolo11n-obb.pt, yolov8n-obb.pt 등)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="에폭 수",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="학습에 사용할 입력 이미지 크기",
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=16,
        help="배치 크기",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="0",
        help='학습 디바이스 ("0" -> GPU 0, "cpu" -> CPU)',
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.8,
        help="train 비율 (나머지는 val/test)",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.1,
        help="val 비율 (test는 자동으로 1 - train - val)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    dataset_root = Path(args.dataset_root)
    if not dataset_root.exists():
        raise FileNotFoundError(f"[ERROR] dataset_root가 존재하지 않습니다: {dataset_root}")

    print("[INFO] 데이터 쌍 수집 중...")
    pairs = collect_image_label_pairs(dataset_root)

    print("[INFO] train/val/test split 수행...")
    train_imgs, val_imgs, test_imgs = split_dataset(
        pairs,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        seed=42,
    )

    print("[INFO] split txt 파일 생성...")
    train_txt, val_txt, test_txt = write_split_txt(
        dataset_root, train_imgs, val_imgs, test_imgs
    )

    # 클래스 이름 정의
    # draw_obb_video.py에서 항상 cls_id=0으로 저장했으므로, 여기서는 0번 클래스만 정의.
    # 필요하면 여러 클래스 이름으로 확장 가능.
    class_names = {
        0: "blade_defect"  # 원하는 이름으로 바꿔도 됨 (예: "wear", "crack" 등)
    }

    data_yaml_path = dataset_root / args.yaml_name
    print("[INFO] data.yaml 생성...")
    create_data_yaml(
        dataset_root,
        train_txt,
        val_txt,
        test_txt,
        data_yaml_path,
        class_names,
    )

    print("[INFO] YOLO OBB 학습 시작...")
    train_yolo_obb(
        data_yaml=data_yaml_path,
        model_name=args.model,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
    )


if __name__ == "__main__":
    main()
