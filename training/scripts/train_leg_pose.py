import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Entrena un modelo YOLO Pose para rodilla/tobillo.")
    parser.add_argument("--data", required=True, help="Ruta a dataset.yaml")
    parser.add_argument("--model", default="yolov8n-pose.pt", help="Modelo base de transferencia")
    parser.add_argument("--epochs", type=int, default=80, help="Numero de epocas")
    parser.add_argument("--imgsz", type=int, default=640, help="Resolucion de entrenamiento")
    parser.add_argument("--batch", type=int, default=16, help="Batch size")
    parser.add_argument("--project", default="training/runs", help="Directorio de salidas")
    parser.add_argument("--name", default="leg-pose", help="Nombre del experimento")
    parser.add_argument("--device", default="auto", help="cpu, 0, 0,1,2... o auto")
    parser.add_argument("--patience", type=int, default=20, help="Early stopping patience")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    data_path = Path(args.data)
    if not data_path.exists():
        raise FileNotFoundError(f"No existe dataset yaml: {data_path}")

    # Import diferido para que el script pueda mostrar --help aunque ultralytics no este instalado.
    from ultralytics import YOLO  # pylint: disable=import-outside-toplevel

    model = YOLO(args.model)
    result = model.train(
        data=str(data_path),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        project=args.project,
        name=args.name,
        device=args.device,
        patience=args.patience,
    )

    best_weights = Path(result.save_dir) / "weights" / "best.pt"
    print(f"Entrenamiento finalizado. Mejor checkpoint: {best_weights}")


if __name__ == "__main__":
    main()
