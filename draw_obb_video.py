import os
import re
import json
import cv2
import numpy as np
from pycocotools import mask as maskUtils


# ---------------------------------------------------------
#  JSON 로딩 / 마스크 디코딩 / OBB 계산 유틸
# ---------------------------------------------------------
def load_frame_annotations(json_path):
    """
    1_4_11.json, 1_4_14.json, 1_4_19.json, 1_4_20.json 구조 예시:
    {
      "video_filename": "...",
      "session_id": "...",
      "frames": [
        {
          "frame_index": 0,
          "results": [
            {
              "object_id": 0,
              "mask": {"size": [1080, 1920], "counts": "..."},
              ...
            },
            ...
          ]
        },
        ...
      ]
    }

    -> { frame_index: results 리스트 } 형태로 변환해서 반환
    """
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    frame_dict = {}
    for frame in data["frames"]:
        idx = frame["frame_index"]
        frame_dict[idx] = frame["results"]
    return frame_dict


def rle_to_mask(mask_obj):
    """
    pycocotools를 이용해 COCO RLE(counts 문자열) -> 바이너리 마스크로 디코딩.
    mask_obj: {"size": [H, W], "counts": "..."}

    반환: (H, W) uint8 배열, 0/1 값
    """
    h, w = mask_obj["size"]
    rle = {
        "size": [h, w],
        "counts": mask_obj["counts"].encode("ascii"),
    }
    mask = maskUtils.decode(rle)  # (H, W) 혹은 (H, W, 1)
    if mask.ndim == 3:
        mask = mask[:, :, 0]
    mask = (mask > 0).astype(np.uint8)
    return mask


def mask_to_obb(mask, min_area=10):
    """
    바이너리 마스크(0/1)에서 Oriented Bounding Box를 계산.
    - mask: (H, W) uint8
    - 반환: 4x2 int numpy 배열 (x,y 좌표 네 점) 또는 None
    """
    contours_info = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours_info) == 3:
        _, contours, _ = contours_info
    else:
        contours, _ = contours_info

    if not contours:
        return None

    # 가장 큰 컨투어만 사용 (노이즈 제거용)
    cnt = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(cnt)
    if area < min_area:
        return None

    # 최소 면적 회전 사각형
    rect = cv2.minAreaRect(cnt)   # ((cx, cy), (w, h), angle)
    box = cv2.boxPoints(rect)     # 4x2 float
    box = box.astype(np.int32)    # int 좌표로 변환
    return box


def draw_obb_on_frame(frame, box, color, thickness=2):
    """
    frame 위에 회전된 박스를 그림.
    """
    cv2.polylines(frame, [box], isClosed=True, color=color, thickness=thickness)


# ---------------------------------------------------------
#  YOLO OBB 라벨 저장 + 이미지 이름 유틸
# ---------------------------------------------------------
def sanitize_stem(path):
    """
    비디오 파일 이름에서 확장자 제거 + YOLO-friendly한 safe한 스템 생성.
    예: "251119 1-4 #11.mp4" -> "251119_1_4__11"
    """
    stem = os.path.splitext(os.path.basename(path))[0]
    stem = re.sub(r"[^0-9a-zA-Z]+", "_", stem)
    return stem


def save_yolo_obb_labels_for_image(
    image_stem,
    boxes,
    class_ids,
    labels_dir,
    img_width,
    img_height,
):
    """
    한 이미지에 대한 YOLO OBB 라벨(.txt)을 저장.
    - image_stem: 이미지 파일명에서 확장자 뺀 부분 (예: '251119_1_4__11_f000123')
    - boxes: [N, 4, 2] (픽셀 단위 x,y)
    - class_ids: 길이 N 리스트 (정수 클래스 id)
    - labels_dir: 라벨 저장 디렉터리
    - img_width, img_height: 이미지 해상도 (w,h)
    """
    if not boxes:
        return

    os.makedirs(labels_dir, exist_ok=True)
    label_path = os.path.join(labels_dir, f"{image_stem}.txt")

    lines = []
    for box, cls_id in zip(boxes, class_ids):
        # box: 4x2, 각 점 (x, y)
        norm_coords = []
        for (x, y) in box:
            nx = float(x) / float(img_width)
            ny = float(y) / float(img_height)
            # 혹시 벗어나면 클램핑
            nx = max(0.0, min(1.0, nx))
            ny = max(0.0, min(1.0, ny))
            norm_coords.extend([nx, ny])

        # YOLO OBB 포맷: cls x1 y1 x2 y2 x3 y3 x4 y4
        line = f"{int(cls_id)} " + " ".join(f"{v:.6f}" for v in norm_coords)
        lines.append(line)

    with open(label_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


# ---------------------------------------------------------
#  비디오 1개 처리: (옵션) mp4 + (옵션) YOLO Dataset
# ---------------------------------------------------------
def process_video(
    video_path,
    json_path,
    output_path=None,           # None이면 비디오 저장 안 함
    draw_mask_overlay=False,
    save_dataset=False,         # True면 images/labels 저장
    dataset_images_dir=None,
    dataset_labels_dir=None,
    default_class_id=0,
):
    """
    - video_path: 원본 mp4 경로 (예: '251119 1-4 #11.mp4')
    - json_path: 세그멘테이션 JSON 경로 (예: '1_4_11.json')
    - output_path: 결과 mp4 저장 경로 (None이면 동영상 저장 X)
    - draw_mask_overlay: True이면, 박스 + 반투명 마스크도 같이 오버레이
    - save_dataset: True이면, 프레임 이미지를 dataset_images_dir에,
                    YOLO OBB 라벨을 dataset_labels_dir에 저장
    - dataset_images_dir: e.g. 'yolo_obb_dataset/images/train'
    - dataset_labels_dir: e.g. 'yolo_obb_dataset/labels/train'
    - default_class_id: JSON에 클래스 정보가 없을 때 사용할 기본 클래스 id
    """
    frame_annos = load_frame_annotations(json_path)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    # 비디오 정보
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30.0  # FPS 정보가 없으면 기본값
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    writer = None
    if output_path is not None:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    if save_dataset:
        if dataset_images_dir is None or dataset_labels_dir is None:
            raise ValueError("save_dataset=True 인데 dataset_images_dir 또는 dataset_labels_dir가 None 입니다.")
        os.makedirs(dataset_images_dir, exist_ok=True)
        os.makedirs(dataset_labels_dir, exist_ok=True)

    video_stem_safe = sanitize_stem(video_path)

    frame_idx = 0
    print(f"[INFO] Start processing {video_path}")
    print(f"       size={width}x{height}, fps={fps:.2f}")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 데이터셋 저장용 원본 프레임 (OBB/마스크 그리기 전에 복사)
        raw_frame = frame.copy()

        yolo_boxes = []    # 이 프레임에서 YOLO 라벨로 쓸 박스들
        yolo_classes = []  # 각 박스의 클래스 id

        # 해당 프레임에 annotation이 있으면 처리
        if frame_idx in frame_annos:
            results = frame_annos[frame_idx]

            for obj in results:
                obj_id = obj["object_id"]
                mask_obj = obj["mask"]

                mask = rle_to_mask(mask_obj)

                # 혹시 마스크 해상도와 비디오 해상도가 다르면 맞춰줌
                if mask.shape[0] != height or mask.shape[1] != width:
                    mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)

                box = mask_to_obb(mask)
                if box is None:
                    continue

                # YOLO 라벨용으로 저장
                yolo_boxes.append(box)
                # 필요하면 obj["class_id"]를 읽어와서 쓰면 됨
                yolo_classes.append(default_class_id)

                # 동영상용 색상 (object_id마다 고정)
                rng = np.random.RandomState(obj_id)
                color = tuple(int(c) for c in rng.randint(0, 255, size=3))

                # OBB 그리기 (비디오용)
                draw_obb_on_frame(frame, box, color=color, thickness=2)

                # 중심에 object_id 텍스트 (선택)
                cx, cy = box.mean(axis=0).astype(int)
                cv2.putText(
                    frame,
                    f"id={obj_id}",
                    (cx, cy),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    color,
                    1,
                    cv2.LINE_AA,
                )

                # 옵션: 마스크 자체를 반투명하게 덮어주기 (비디오 시각화용)
                if draw_mask_overlay:
                    mask_bool = mask.astype(bool)
                    overlay = frame.astype(np.float32)
                    overlay[mask_bool] = 0.5 * overlay[mask_bool] + 0.5 * np.array(color, dtype=np.float32)
                    frame = overlay.astype(np.uint8)

        # 데이터셋 저장 (프레임에 최소 한 개의 박스가 있을 때만 저장)
        if save_dataset and len(yolo_boxes) > 0:
            image_stem = f"{video_stem_safe}_f{frame_idx:06d}"
            img_path = os.path.join(dataset_images_dir, f"{image_stem}.jpg")
            cv2.imwrite(img_path, raw_frame)
            save_yolo_obb_labels_for_image(
                image_stem=image_stem,
                boxes=yolo_boxes,
                class_ids=yolo_classes,
                labels_dir=dataset_labels_dir,
                img_width=width,
                img_height=height,
            )

        # 비디오 저장
        if writer is not None:
            writer.write(frame)

        frame_idx += 1
        if frame_idx % 100 == 0:
            print(f"[INFO] processed frame {frame_idx}")

    cap.release()
    if writer is not None:
        writer.release()
        print(f"[INFO] Done. Saved video to {output_path}")

    print(f"[INFO] Finished {video_path}, total frames = {frame_idx}")


# ---------------------------------------------------------
#  메인: 4개 mp4 + 4개 json 한 번에 돌려서 YOLO OBB Dataset 생성
# ---------------------------------------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description=(
            "세그멘테이션 JSON을 이용해 mp4에 OBB를 그려 넣고, "
            "동시에 YOLO OBB 학습용 이미지/라벨 데이터셋을 만든다."
        )
    )

    # (옵션) 단일 비디오만 처리하고 싶을 때 사용
    parser.add_argument("--video", type=str, help="단일 입력 비디오 (.mp4)")
    parser.add_argument("--json", type=str, help="단일 세그멘테이션 JSON 파일")
    parser.add_argument("--out", type=str, default="output_obb.mp4", help="단일 비디오 모드에서 결과 mp4 경로")

    # 여러 비디오(현재 4개) 한 번에 처리하는 모드
    parser.add_argument(
        "--process-all",
        action="store_true",
        help="현재 코드에 하드코딩된 4개 (mp4, json) 쌍을 모두 처리",
    )

    # 시각화 옵션
    parser.add_argument(
        "--overlay-mask",
        action="store_true",
        help="비디오에 반투명 세그멘테이션 마스크도 같이 오버레이",
    )
    parser.add_argument(
        "--save-videos",
        action="store_true",
        help="각 비디오마다 OBB가 그려진 mp4를 저장",
    )

    # YOLO Dataset 옵션
    parser.add_argument(
        "--dataset-root",
        type=str,
        default=None,
        help="YOLO OBB 데이터셋 루트 디렉터리 (예: 'yolo_obb_dataset'). "
             "process-all 모드에서는 지정 안 하면 자동으로 'yolo_obb_dataset' 사용.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        help="데이터셋 하위 split 이름 (기본: train) -> images/<split>, labels/<split>",
    )
    parser.add_argument(
        "--class-id",
        type=int,
        default=0,
        help="JSON에 클래스 정보가 없을 때 사용할 기본 YOLO class id",
    )

    args = parser.parse_args()

    # process-all 모드: 4개 비디오 + 4개 json 모두 돌리기
    if args.process_all:
        dataset_root = args.dataset_root or "yolo_obb_dataset"
        images_dir = os.path.join(dataset_root, "images", args.split)
        labels_dir = os.path.join(dataset_root, "labels", args.split)

        os.makedirs(images_dir, exist_ok=True)
        os.makedirs(labels_dir, exist_ok=True)

        # 🔴 여기서 4개 (mp4, json) 쌍을 정의
        VIDEO_JSON_PAIRS = [
            ("251119 1-4 #11.mp4", "1_4_11.json", "1_4_11_obb.mp4"),
            ("251119 1-4 #14.mp4", "1_4_14.json", "1_4_14_obb.mp4"),
            ("251119 1-4 #19.mp4", "1_4_19.json", "1_4_19_obb.mp4"),
            ("251119 1-4 #20.mp4", "1_4_20.json", "1_4_20_obb.mp4"),
        ]

        print(f"[INFO] YOLO OBB dataset root = {dataset_root}")
        print(f"[INFO] images -> {images_dir}")
        print(f"[INFO] labels -> {labels_dir}")

        for video_path, json_path, out_name in VIDEO_JSON_PAIRS:
            out_path = out_name if args.save_videos else None
            process_video(
                video_path=video_path,
                json_path=json_path,
                output_path=out_path,
                draw_mask_overlay=args.overlay_mask,
                save_dataset=True,
                dataset_images_dir=images_dir,
                dataset_labels_dir=labels_dir,
                default_class_id=args.class_id,
            )

        print("[INFO] All videos processed. YOLO OBB dataset ready.")

    else:
        # 단일 비디오 모드
        if args.video is None or args.json is None:
            parser.error("단일 비디오 모드에서는 --video 와 --json 을 모두 지정하거나, --process-all 을 사용해야 합니다.")

        # 단일 비디오에서도 원하면 데이터셋에 추가할 수 있게 옵션 허용
        save_dataset = args.dataset_root is not None
        if save_dataset:
            dataset_root = args.dataset_root
            images_dir = os.path.join(dataset_root, "images", args.split)
            labels_dir = os.path.join(dataset_root, "labels", args.split)
        else:
            images_dir = None
            labels_dir = None

        process_video(
            video_path=args.video,
            json_path=args.json,
            output_path=args.out if args.save_videos or args.out is not None else None,
            draw_mask_overlay=args.overlay_mask,
            save_dataset=save_dataset,
            dataset_images_dir=images_dir,
            dataset_labels_dir=labels_dir,
            default_class_id=args.class_id,
        )
