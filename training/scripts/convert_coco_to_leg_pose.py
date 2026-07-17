import argparse
import json
import random
import shutil
from pathlib import Path
from typing import Dict, List, Tuple

# COCO keypoint indexes
LEFT_KNEE = 13
RIGHT_KNEE = 14
LEFT_ANKLE = 15
RIGHT_ANKLE = 16
LEG_KEYPOINT_INDEXES = [LEFT_KNEE, LEFT_ANKLE, RIGHT_KNEE, RIGHT_ANKLE]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convierte anotaciones COCO person keypoints a dataset YOLO Pose de 4 keypoints (rodilla/tobillo)."
    )
    parser.add_argument("--images", required=True, help="Carpeta con imagenes COCO (ej. train2017 o val2017)")
    parser.add_argument("--annotations", required=True, help="Archivo JSON de anotaciones COCO person_keypoints")
    parser.add_argument("--output", required=True, help="Directorio de salida para dataset YOLO pose")
    parser.add_argument("--split", default="train", choices=["train", "val"], help="Split a generar")
    parser.add_argument("--sample", type=int, default=0, help="Numero maximo de imagenes a muestrear (0 = todas)")
    parser.add_argument("--seed", type=int, default=42, help="Semilla para muestreo")
    parser.add_argument(
        "--min-visible-keypoints",
        type=int,
        default=2,
        help="Minimo de keypoints visibles (v>0) entre los 4 para incluir una instancia",
    )
    return parser.parse_args()


def yolo_bbox(bbox: List[float], width: int, height: int) -> Tuple[float, float, float, float]:
    x, y, w, h = bbox
    cx = (x + (w / 2.0)) / width
    cy = (y + (h / 2.0)) / height
    return cx, cy, w / width, h / height


def yolo_keypoints(coco_keypoints: List[float], width: int, height: int) -> Tuple[List[Tuple[float, float, int]], int]:
    kpts = []
    visible_count = 0

    for idx in LEG_KEYPOINT_INDEXES:
        base = idx * 3
        x = coco_keypoints[base]
        y = coco_keypoints[base + 1]
        v = int(coco_keypoints[base + 2])
        if v > 0:
            visible_count += 1
            kpts.append((x / width, y / height, min(v, 2)))
        else:
            kpts.append((0.0, 0.0, 0))

    return kpts, visible_count


def ensure_dirs(root: Path, split: str) -> Tuple[Path, Path]:
    images_out = root / "images" / split
    labels_out = root / "labels" / split
    images_out.mkdir(parents=True, exist_ok=True)
    labels_out.mkdir(parents=True, exist_ok=True)
    return images_out, labels_out


def main() -> None:
    args = parse_args()

    images_dir = Path(args.images)
    ann_path = Path(args.annotations)
    output_dir = Path(args.output)

    if not images_dir.exists():
        raise FileNotFoundError(f"No existe carpeta de imagenes: {images_dir}")
    if not ann_path.exists():
        raise FileNotFoundError(f"No existe archivo de anotaciones: {ann_path}")

    images_out, labels_out = ensure_dirs(output_dir, args.split)

    with ann_path.open("r", encoding="utf-8") as f:
        coco = json.load(f)

    image_meta: Dict[int, Dict] = {img["id"]: img for img in coco.get("images", [])}
    anns_by_image: Dict[int, List[Dict]] = {}

    for ann in coco.get("annotations", []):
        if ann.get("iscrowd", 0) == 1:
            continue
        if ann.get("category_id") != 1:
            continue
        anns_by_image.setdefault(ann["image_id"], []).append(ann)

    image_ids = list(anns_by_image.keys())
    if args.sample > 0 and args.sample < len(image_ids):
        random.seed(args.seed)
        image_ids = random.sample(image_ids, args.sample)

    written_images = 0
    written_labels = 0

    for image_id in image_ids:
        meta = image_meta.get(image_id)
        if not meta:
            continue

        file_name = meta["file_name"]
        width = int(meta["width"])
        height = int(meta["height"])
        src_image_path = images_dir / file_name

        if not src_image_path.exists():
            continue

        label_lines: List[str] = []

        for ann in anns_by_image.get(image_id, []):
            bbox = ann.get("bbox", None)
            kps = ann.get("keypoints", None)
            if not bbox or not kps or len(kps) < (17 * 3):
                continue

            kpt_values, visible_count = yolo_keypoints(kps, width, height)
            if visible_count < args.min_visible_keypoints:
                continue

            cx, cy, bw, bh = yolo_bbox(bbox, width, height)

            parts = ["0", f"{cx:.6f}", f"{cy:.6f}", f"{bw:.6f}", f"{bh:.6f}"]
            for x, y, v in kpt_values:
                parts.extend([f"{x:.6f}", f"{y:.6f}", str(v)])

            label_lines.append(" ".join(parts))

        if not label_lines:
            continue

        dst_image_path = images_out / file_name
        dst_label_path = labels_out / f"{Path(file_name).stem}.txt"

        shutil.copy2(src_image_path, dst_image_path)
        with dst_label_path.open("w", encoding="utf-8") as f:
            f.write("\n".join(label_lines) + "\n")

        written_images += 1
        written_labels += 1

    dataset_yaml = output_dir / "dataset.yaml"
    dataset_yaml.write_text(
        "\n".join(
            [
                f"path: {output_dir.as_posix()}",
                "train: images/train",
                "val: images/val",
                "kpt_shape: [4, 3]",
                "flip_idx: [2, 3, 0, 1]",
                "names:",
                "  0: leg",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    print(f"Imagenes escritas: {written_images}")
    print(f"Etiquetas escritas: {written_labels}")
    print(f"Dataset YAML: {dataset_yaml}")


if __name__ == "__main__":
    main()
