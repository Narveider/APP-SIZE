"""
Descarga los datasets de Legs Detection desde Roboflow Universe:
  1. estagio2223/legs-detection-igg2c  (bbox detector, 59 imgs)  — capa de recorte
  2. sebatests/legs-8aok7              (keypoint detection, 312 imgs) — poses de pierna

Uso:
    python training/scripts/download_legs_dataset.py --api-key TU_API_KEY

Obtener API key gratuita en: https://app.roboflow.com/settings/api
"""

import argparse
import textwrap
from pathlib import Path


DATASETS = [
    {
        "workspace": "estagio2223",
        "project": "legs-detection-igg2c",
        "version": 2,
        "format": "yolov8",
        "outdir": "datasets/legs-bbox",
        "description": "Legs bbox detector (capa de recorte pre-MoveNet)",
    },
    {
        "workspace": "sebatests",
        "project": "legs-8aok7",
        "version": 9,
        "format": "yolov8",
        "outdir": "datasets/legs-keypoints",
        "description": "Legs keypoint detection YOLOv11n-pose",
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=textwrap.dedent(__doc__),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--api-key",
        required=True,
        help="Roboflow API key (https://app.roboflow.com/settings/api)",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=["legs-bbox", "legs-keypoints", "all"],
        default=["all"],
        help="Que datasets descargar",
    )
    return parser.parse_args()


def download_dataset(cfg: dict, api_key: str) -> None:
    from roboflow import Roboflow  # pylint: disable=import-outside-toplevel

    print(f"\n>>> Descargando: {cfg['description']}")
    print(f"    {cfg['workspace']}/{cfg['project']} v{cfg['version']} → {cfg['outdir']}")

    rf = Roboflow(api_key=api_key)
    project = rf.workspace(cfg["workspace"]).project(cfg["project"])
    version = project.version(cfg["version"])
    dataset = version.download(cfg["format"], location=cfg["outdir"])
    print(f"    Guardado en: {dataset.location}")


def main() -> None:
    args = parse_args()

    want_all = "all" in args.datasets
    to_download = [
        cfg
        for cfg in DATASETS
        if want_all or cfg["outdir"].split("/")[-1] in args.datasets
    ]

    for cfg in to_download:
        download_dataset(cfg, args.api_key)

    print(
        "\nDatasets listos. Siguiente paso:\n"
        "  python training/scripts/train_legs_detector.py --help\n"
        "  python training/scripts/train_legs_detector.py --data datasets/legs-bbox --epochs 80\n"
    )


if __name__ == "__main__":
    main()
