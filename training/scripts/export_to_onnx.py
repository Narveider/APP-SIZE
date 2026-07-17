"""
Exporta el modelo YOLO entrenado (.pt) a ONNX para integrarlo en la app
o en un backend de inferencia.

Uso:
    python training/scripts/export_to_onnx.py \
        --weights training/runs/legs-detector-v1/weights/best.pt
"""

import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--weights", required=True, help="Ruta al archivo best.pt o last.pt")
    parser.add_argument("--imgsz", type=int, default=640, help="Resolucion de exportacion")
    parser.add_argument(
        "--format",
        default="onnx",
        choices=["onnx", "tflite", "torchscript"],
        help="Formato de exportacion",
    )
    parser.add_argument(
        "--simplify",
        action="store_true",
        default=True,
        help="Simplificar grafo ONNX (recomendado)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    weights = Path(args.weights)

    if not weights.exists():
        raise FileNotFoundError(f"No existe el checkpoint: {weights}")

    from ultralytics import YOLO  # pylint: disable=import-outside-toplevel

    model = YOLO(str(weights))
    out = model.export(
        format=args.format,
        imgsz=args.imgsz,
        simplify=args.simplify,
    )

    out_path = Path(out) if isinstance(out, str) else out
    print(f"\nModelo exportado: {out_path}")

    if args.format == "onnx":
        print(
            "\nPara convertir a TensorFlow.js y usar en la app web:\n"
            "  pip install tensorflowjs\n"
            f"  tensorflowjs_converter --input_format=tf_saved_model {out_path} src/ai/model/legs_detector_tfjs/\n"
        )
    elif args.format == "tflite":
        print(
            "\nEl modelo .tflite puede usarse con react-native-fast-tflite en la app nativa.\n"
            f"  Copia {out_path} a assets/ y referencialo desde poseDetector.ts\n"
        )


if __name__ == "__main__":
    main()
