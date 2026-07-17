import * as FileSystem from 'expo-file-system';
import * as tf from '@tensorflow/tfjs';
import * as poseDetection from '@tensorflow-models/pose-detection';
import jpeg from 'jpeg-js';
import { Point } from '../utils/measurement';

const LEFT_HIP_INDEX = 11;
const RIGHT_HIP_INDEX = 12;
const LEFT_KNEE_INDEX = 13;
const RIGHT_KNEE_INDEX = 14;
const LEFT_ANKLE_INDEX = 15;
const RIGHT_ANKLE_INDEX = 16;
const INFERENCE_MAX_SIDES = [1024, 768, 512] as const;

// Umbral minimo para considerar un keypoint como valido en la primera pasada
const FIRST_PASS_SCORE_THRESHOLD = 0.15;
// Padding porcentual alrededor del recorte de pierna
const CROP_PADDING_RATIO = 0.25;

type KeypointLike = {
  x?: number;
  y?: number;
  score?: number;
  name?: string;
};

export type PoseLegEstimate = {
  knee: Point;
  ankle: Point;
  confidence: number;
  side: 'left' | 'right';
};

let detectorPromise: Promise<poseDetection.PoseDetector> | null = null;

const getKeypointByNameOrIndex = (
  keypoints: KeypointLike[],
  name: string,
  index: number,
): KeypointLike | null => {
  const byName = keypoints.find((point) => point.name === name);
  if (byName) {
    return byName;
  }

  return keypoints[index] ?? null;
};

const scorePair = (knee: KeypointLike | null, ankle: KeypointLike | null): number => {
  const kneeScore = knee?.score ?? 0;
  const ankleScore = ankle?.score ?? 0;
  return (kneeScore + ankleScore) / 2;
};

const hasCoordinates = (point: KeypointLike | null): point is Required<Pick<KeypointLike, 'x' | 'y'>> => {
  return Boolean(point && Number.isFinite(point.x) && Number.isFinite(point.y));
};

/**
 * Calcula el bounding box de la region de piernas (caderas, rodillas, tobillos)
 * a partir de los keypoints de la primera pasada, con padding porcentual.
 * Retorna null si no hay suficiente informacion.
 */
const computeLegCropRegion = (
  keypoints: KeypointLike[],
  imgW: number,
  imgH: number,
): { x: number; y: number; w: number; h: number } | null => {
  const legIndices = [LEFT_HIP_INDEX, RIGHT_HIP_INDEX, LEFT_KNEE_INDEX, RIGHT_KNEE_INDEX, LEFT_ANKLE_INDEX, RIGHT_ANKLE_INDEX];
  const validPoints = legIndices
    .map((i) => keypoints[i])
    .filter((kp): kp is Required<Pick<KeypointLike, 'x' | 'y' | 'score'>> =>
      Boolean(kp && Number.isFinite(kp.x) && Number.isFinite(kp.y) && (kp.score ?? 0) >= FIRST_PASS_SCORE_THRESHOLD),
    );

  if (validPoints.length < 2) {
    return null;
  }

  const xs = validPoints.map((p) => p.x);
  const ys = validPoints.map((p) => p.y);
  const minX = Math.min(...xs);
  const maxX = Math.max(...xs);
  const minY = Math.min(...ys);
  const maxY = Math.max(...ys);

  const regionW = maxX - minX;
  const regionH = maxY - minY;
  const padX = regionW * CROP_PADDING_RATIO;
  const padY = regionH * CROP_PADDING_RATIO;

  const x = Math.max(0, Math.round(minX - padX));
  const y = Math.max(0, Math.round(minY - padY));
  const w = Math.min(imgW - x, Math.round(regionW + padX * 2));
  const h = Math.min(imgH - y, Math.round(regionH + padY * 2));

  if (w < 32 || h < 32) {
    return null;
  }

  return { x, y, w, h };
};

/**
 * Segunda pasada: recorta el tensor a la region de piernas y re-corre MoveNet
 * a mayor resolucion para mejorar precision de rodilla/tobillo.
 * Las coordenadas se mapean de vuelta al espacio original.
 */
const refineEstimateWithCrop = async (
  detector: poseDetection.PoseDetector,
  imageTensor: tf.Tensor3D,
  firstPassKeypoints: KeypointLike[],
  imageWidth: number,
  imageHeight: number,
): Promise<PoseLegEstimate | null> => {
  const region = computeLegCropRegion(firstPassKeypoints, imageWidth, imageHeight);
  if (!region) {
    return null;
  }

  const cropTensor = tf.slice(imageTensor, [region.y, region.x, 0], [region.h, region.w, 3]) as tf.Tensor3D;
  const maxSide = 640;
  const scale = Math.min(1, maxSide / Math.max(region.w, region.h));
  const resizedTensor =
    scale < 1
      ? tf.image.resizeBilinear(cropTensor, [Math.round(region.h * scale), Math.round(region.w * scale)])
      : null;
  const inferTensor = scale < 1 && resizedTensor ? tf.cast(resizedTensor, 'int32') : cropTensor;

  try {
    const poses = await detector.estimatePoses(inferTensor, { maxPoses: 1, flipHorizontal: false });
    const kps = (poses[0]?.keypoints ?? []) as KeypointLike[];
    if (!kps.length) return null;

    // Ajustar coordenadas de vuelta al espacio de imagen completa
    const remapped: KeypointLike[] = kps.map((kp) => ({
      ...kp,
      x: kp.x != null ? kp.x / scale + region.x : kp.x,
      y: kp.y != null ? kp.y / scale + region.y : kp.y,
    }));

    return buildLegEstimateFromKeypoints(remapped, 1);
  } finally {
    if (inferTensor !== cropTensor) inferTensor.dispose();
    resizedTensor?.dispose();
    cropTensor.dispose();
  }
};

const buildLegEstimateFromKeypoints = (keypoints: KeypointLike[], scale: number): PoseLegEstimate | null => {
  const leftKnee = getKeypointByNameOrIndex(keypoints, 'left_knee', LEFT_KNEE_INDEX);
  const leftAnkle = getKeypointByNameOrIndex(keypoints, 'left_ankle', LEFT_ANKLE_INDEX);
  const rightKnee = getKeypointByNameOrIndex(keypoints, 'right_knee', RIGHT_KNEE_INDEX);
  const rightAnkle = getKeypointByNameOrIndex(keypoints, 'right_ankle', RIGHT_ANKLE_INDEX);

  const candidates: PoseLegEstimate[] = [];

  if (hasCoordinates(leftKnee) && hasCoordinates(leftAnkle)) {
    candidates.push({
      knee: {
        x: leftKnee.x / scale,
        y: leftKnee.y / scale,
      },
      ankle: {
        x: leftAnkle.x / scale,
        y: leftAnkle.y / scale,
      },
      confidence: scorePair(leftKnee, leftAnkle),
      side: 'left',
    });
  }

  if (hasCoordinates(rightKnee) && hasCoordinates(rightAnkle)) {
    candidates.push({
      knee: {
        x: rightKnee.x / scale,
        y: rightKnee.y / scale,
      },
      ankle: {
        x: rightAnkle.x / scale,
        y: rightAnkle.y / scale,
      },
      confidence: scorePair(rightKnee, rightAnkle),
      side: 'right',
    });
  }

  if (!candidates.length) {
    return null;
  }

  return candidates.reduce((best, current) => (current.confidence > best.confidence ? current : best));
};

const estimateLegWithMultiScale = async (
  detector: poseDetection.PoseDetector,
  imageTensor: tf.Tensor3D,
  imageWidth: number,
  imageHeight: number,
): Promise<PoseLegEstimate | null> => {
  let bestEstimate: PoseLegEstimate | null = null;
  let bestFirstPassKeypoints: KeypointLike[] | null = null;

  for (const maxSide of INFERENCE_MAX_SIDES) {
    const scale = Math.min(1, maxSide / Math.max(imageWidth, imageHeight));
    const resizedTensor =
      scale < 1
        ? tf.image.resizeBilinear(imageTensor, [Math.round(imageHeight * scale), Math.round(imageWidth * scale)])
        : null;
    const inferTensor =
      scale < 1 && resizedTensor ? tf.cast(resizedTensor, 'int32') : imageTensor;

    try {
      const poses = await detector.estimatePoses(inferTensor, {
        maxPoses: 1,
        flipHorizontal: false,
      });

      const rawKeypoints = (poses[0]?.keypoints ?? []) as KeypointLike[];
      if (!rawKeypoints.length) {
        continue;
      }

      // Convertir coordenadas de la escala reducida al espacio original
      const remappedKeypoints: KeypointLike[] = rawKeypoints.map((kp) => ({
        ...kp,
        x: kp.x != null ? kp.x / scale : kp.x,
        y: kp.y != null ? kp.y / scale : kp.y,
      }));

      const candidate = buildLegEstimateFromKeypoints(remappedKeypoints, 1);
      if (!candidate) {
        continue;
      }

      if (!bestEstimate || candidate.confidence > bestEstimate.confidence) {
        bestEstimate = candidate;
        bestFirstPassKeypoints = remappedKeypoints;
      }
    } finally {
      if (inferTensor !== imageTensor) {
        inferTensor.dispose();
      }
      resizedTensor?.dispose();
    }
  }

  // Segunda pasada: crop-then-refine para mayor precision en rodilla/tobillo
  if (bestFirstPassKeypoints) {
    const refined = await refineEstimateWithCrop(detector, imageTensor, bestFirstPassKeypoints, imageWidth, imageHeight);
    if (refined && refined.confidence > (bestEstimate?.confidence ?? 0)) {
      return refined;
    }
  }

  return bestEstimate;
};

const createDetector = async (): Promise<poseDetection.PoseDetector> => {
  await tf.ready();

  try {
    await tf.setBackend('webgl');
  } catch {
    await tf.setBackend('cpu');
  }

  await tf.ready();

  return poseDetection.createDetector(poseDetection.SupportedModels.MoveNet, {
    modelType: poseDetection.movenet.modelType.SINGLEPOSE_THUNDER,
    enableSmoothing: true,
  });
};

const getDetector = async (): Promise<poseDetection.PoseDetector> => {
  if (!detectorPromise) {
    detectorPromise = createDetector();
  }

  return detectorPromise;
};

export const detectLegPointsFromImage = async (uri: string): Promise<PoseLegEstimate | null> => {
  const detector = await getDetector();
  const base64Image = await FileSystem.readAsStringAsync(uri, {
    encoding: FileSystem.EncodingType.Base64,
  });

  const imageBuffer = tf.util.encodeString(base64Image, 'base64').buffer;
  const decoded = jpeg.decode(new Uint8Array(imageBuffer), {
    useTArray: true,
  });
  const rgbaTensor = tf.tensor3d(decoded.data, [decoded.height, decoded.width, 4], 'int32');
  const imageTensor = tf.slice(rgbaTensor, [0, 0, 0], [-1, -1, 3]);

  try {
    return estimateLegWithMultiScale(detector, imageTensor, decoded.width, decoded.height);
  } finally {
    rgbaTensor.dispose();
    imageTensor.dispose();
  }
};
