import * as FileSystem from 'expo-file-system';
import * as tf from '@tensorflow/tfjs';
import * as poseDetection from '@tensorflow-models/pose-detection';
import jpeg from 'jpeg-js';
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

  try {
    await tf.setBackend('webgl');
  } catch {
    await tf.setBackend('cpu');
  }

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
    const poses = await detector.estimatePoses(imageTensor, {
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
        x: selectedKnee.x,
        y: selectedKnee.y,
      },
      ankle: {
        x: selectedAnkle.x,
        y: selectedAnkle.y,
      },
      confidence: selectedSide === 'left' ? leftScore : rightScore,
      side: selectedSide,
    };
  } finally {
    rgbaTensor.dispose();
    imageTensor.dispose();
  }
};
