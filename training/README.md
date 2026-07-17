# Entrenamiento IA para rodilla y tobillo

Este flujo entrena un detector de 4 keypoints (rodilla izq, tobillo izq, rodilla der, tobillo der) para reducir fallos en casos clinicos reales.

## 1) Requisitos

```bash
python -m pip install -r training/requirements.txt
```

## 2) Preparar dataset desde COCO

Debes tener descargado COCO en local, por ejemplo:

- `datasets/coco/train2017`
- `datasets/coco/val2017`
- `datasets/coco/annotations/person_keypoints_train2017.json`
- `datasets/coco/annotations/person_keypoints_val2017.json`

Generar split `train`:

```bash
python training/scripts/convert_coco_to_leg_pose.py \
  --images datasets/coco/train2017 \
  --annotations datasets/coco/annotations/person_keypoints_train2017.json \
  --output datasets/leg-pose \
  --split train \
  --sample 20000
```

Generar split `val`:

```bash
python training/scripts/convert_coco_to_leg_pose.py \
  --images datasets/coco/val2017 \
  --annotations datasets/coco/annotations/person_keypoints_val2017.json \
  --output datasets/leg-pose \
  --split val \
  --sample 5000
```

Notas:
- Ajusta `--sample` segun capacidad de maquina.
- Si ya tienes tu propio dataset clinico etiquetado, es mejor usarlo para reducir errores en postrados.

## 3) Entrenar modelo

```bash
python training/scripts/train_leg_pose.py \
  --data datasets/leg-pose/dataset.yaml \
  --model yolov8n-pose.pt \
  --epochs 80 \
  --imgsz 640 \
  --batch 16 \
  --project training/runs \
  --name leg-pose-v1
```

El mejor checkpoint queda en:

- `training/runs/leg-pose-v1/weights/best.pt`

## 4) Recomendacion para mejorar precision real

1. Mezcla COCO + fotos reales del caso de uso (paciente acostado, sabanas, baja luz).
2. Revisa etiquetas manuales de rodilla/tobillo en tus muestras reales.
3. Reentrena cada lote de 200 a 500 fotos nuevas.
4. Evalua por separado izquierda/derecha y por condiciones de iluminacion.

## 5) Integracion con la app

La app actual usa MoveNet de TensorFlow.js para inferencia. Este entrenamiento crea un modelo YOLO Pose (`.pt`).

Para usar el modelo entrenado en movil/web hay dos rutas:

1. Convertir y servir el modelo en un backend de inferencia.
2. Convertir a un formato compatible con inferencia local en Expo y crear adaptador de salida a `{knee, ankle}`.

Si quieres, en el siguiente paso te implemento la ruta 1 (API local) o la ruta 2 (inferencia on-device), segun prefieras latencia vs complejidad.
