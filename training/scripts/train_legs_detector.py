"""
Entrena un detector de piernas con YOLO (para la capa de recorte previo a MoveNet).

Uso rapido con el dataset descargado:
    python training/scripts/train_legs_detector.py \
        --data datasets/legs-bbox \
        --epochs 80

El modelo entrenado queda en training/runs/legs-detector-v1/weights/best.pt
"""

import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--data", required=True, help="Carpeta del dataset descargado por Roboflow (contiene data.yaml)")
    parser.add_argument("--model", default="yolo11n.pt", help="Modelo base para transfer learning")
    parser.add_argument("--epochs", type=int, default=80, help="Epocas de entrenamiento")
    parser.add_argument("--imgsz", type=int, default=640, help="Resolucion")
    parser.add_argument("--batch", type=int, default=16, help="Batch size")
    parser.add_argument("--project", default="training/runs", help="Directorio de salidas")
    parser.add_argument("--name", default="legs-detector-v1", help="Nombre del experimento")
    parser.add_argument("--device", default="auto", help="cpu, 0, cuda:0, ...")
    parser.add_argument("--patience", type=int, default=20, help="Early stopping patience")
    return parser.parse_args()


def find_yaml(data_dir: Path) -> Path:
    """Busca el data.yaml dentro de la carpeta descargada por Roboflow."""
    candidates = list(data_dir.glob("data.yaml")) + list(data_dir.glob("*.yaml"))
    if not candidates:
        raise FileNotFoundError(
            f"No se encontro data.yaml en {data_dir}. "
            "Asegurate de haber descargado el dataset primero con download_legs_dataset.py"
        )
    return candidates[0]


def main() -> None:
    args = parse_args()
    data_dir = Path(args.data)

    if not data_dir.exists():
        raise FileNotFoundError(
            f"No existe la carpeta de dataset: {data_dir}\n"
            "Ejecuta primero: python training/scripts/download_legs_dataset.py --api-key TU_KEY"
        )

    yaml_path = find_yaml(data_dir)
    print(f"Usando YAML: {yaml_path}")

    from ultralytics import YOLO  # pylint: disable=import-outside-toplevel

    model = YOLO(args.model)
    result = model.train(
        data=str(yaml_path),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        project=args.project,
        name=args.name,
        device=args.device,
        patience=args.patience,
    )

    best_pt = Path(result.save_dir) / "weights" / "best.pt"
    print(f"\nEntrenamiento finalizado.")
    print(f"Mejor checkpoint : {best_pt}")
    print(f"\nSiguiente paso - exportar a ONNX:")
    print(f"  python training/scripts/export_to_onnx.py --weights {best_pt}")


if __name__ == "__main__":
    main()
