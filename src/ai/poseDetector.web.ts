import * as tf from '@tensorflow/tfjs';
import '@tensorflow/tfjs-backend-webgl';
import * as poseDetection from '@tensorflow-models/pose-detection';
import { Point } from '../utils/measurement';

const LEFT_KNEE_INDEX = 13;
const RIGHT_KNEE_INDEX = 14;
const LEFT_ANKLE_INDEX = 15;
const RIGHT_ANKLE_INDEX = 16;

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

const createDetector = async (): Promise<poseDetection.PoseDetector> => {
  await tf.ready();
  await tf.setBackend('webgl');
  await tf.ready();

  return poseDetection.createDetector(poseDetection.SupportedModels.MoveNet, {
    modelType: poseDetection.movenet.modelType.SINGLEPOSE_LIGHTNING,
    enableSmoothing: true,
  });
};

const getDetector = async (): Promise<poseDetection.PoseDetector> => {
  if (!detectorPromise) {
    detectorPromise = createDetector();
  }

  return detectorPromise;
};

const loadImage = (uri: string): Promise<HTMLImageElement> => {
  return new Promise((resolve, reject) => {
    const image = new Image();
    image.crossOrigin = 'anonymous';
    image.onload = () => resolve(image);
    image.onerror = () => reject(new Error('No se pudo cargar la imagen para IA.'));
    image.src = uri;
  });
};

export const detectLegPointsFromImage = async (uri: string): Promise<PoseLegEstimate | null> => {
  const detector = await getDetector();
  const image = await loadImage(uri);
  const imageTensor = tf.browser.fromPixels(image);
  const maxSide = 512;
  const scale = Math.min(1, maxSide / Math.max(image.width, image.height));
  const resizedTensor =
    scale < 1
      ? tf.image.resizeBilinear(imageTensor, [Math.round(image.height * scale), Math.round(image.width * scale)])
      : imageTensor;
  const inferTensor = scale < 1 ? tf.cast(resizedTensor, 'int32') : imageTensor;

  try {
    const poses = await detector.estimatePoses(inferTensor as unknown as poseDetection.PoseDetectorInput, {
      maxPoses: 1,
      flipHorizontal: false,
    });

    const keypoints = (poses[0]?.keypoints ?? []) as KeypointLike[];
    if (!keypoints.length) {
      return null;
    }

    const leftKnee = getKeypointByNameOrIndex(keypoints, 'left_knee', LEFT_KNEE_INDEX);
    const leftAnkle = getKeypointByNameOrIndex(keypoints, 'left_ankle', LEFT_ANKLE_INDEX);
    const rightKnee = getKeypointByNameOrIndex(keypoints, 'right_knee', RIGHT_KNEE_INDEX);
    const rightAnkle = getKeypointByNameOrIndex(keypoints, 'right_ankle', RIGHT_ANKLE_INDEX);

    const leftScore = scorePair(leftKnee, leftAnkle);
    const rightScore = scorePair(rightKnee, rightAnkle);
    const selectedSide = leftScore >= rightScore ? 'left' : 'right';

    const selectedKnee = selectedSide === 'left' ? leftKnee : rightKnee;
    const selectedAnkle = selectedSide === 'left' ? leftAnkle : rightAnkle;

    if (!hasCoordinates(selectedKnee) || !hasCoordinates(selectedAnkle)) {
      return null;
    }

    return {
      knee: {
        x: selectedKnee.x / scale,
        y: selectedKnee.y / scale,
      },
      ankle: {
        x: selectedAnkle.x / scale,
        y: selectedAnkle.y / scale,
      },
      confidence: selectedSide === 'left' ? leftScore : rightScore,
      side: selectedSide,
    };
  } finally {
    if (scale < 1) {
      resizedTensor.dispose();
      inferTensor.dispose();
    }
    imageTensor.dispose();
  }
};
