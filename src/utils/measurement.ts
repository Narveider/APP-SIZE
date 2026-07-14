export type Point = {
  x: number;
  y: number;
};

type CalculateMeasuredLengthInput = {
  referenceCm: number;
  referencePixelDistance: number;
  targetPixelDistance: number;
};

type PerspectiveCorrectionInput = {
  referenceMidY: number;
  targetMidY: number;
  imageHeight: number;
  strength?: number;
};

export type SavedMeasurement = {
  id: string;
  createdAt: string;
  referenceCm: number;
  measuredCm: number;
  correctedCm: number | null;
  perspectiveEnabled: boolean;
  referencePixelDistance: number;
  targetPixelDistance: number;
  aiConfidence: number | null;
  aiSide: 'left' | 'right' | null;
};

export const distance = (a: Point, b: Point): number => {
  const dx = b.x - a.x;
  const dy = b.y - a.y;
  return Math.sqrt(dx * dx + dy * dy);
};

export const calculateMeasuredLengthCm = ({
  referenceCm,
  referencePixelDistance,
  targetPixelDistance,
}: CalculateMeasuredLengthInput): number | null => {
  if (referenceCm <= 0 || referencePixelDistance <= 0 || targetPixelDistance <= 0) {
    return null;
  }

  return (targetPixelDistance / referencePixelDistance) * referenceCm;
};

export const parseReferenceCm = (value: string): number | null => {
  const normalized = value.replace(',', '.').trim();
  const parsed = Number(normalized);
  if (!Number.isFinite(parsed) || parsed <= 0) {
    return null;
  }

  return parsed;
};

export const formatCm = (value: number): string => {
  return value.toFixed(2);
};

export const midpointY = (a: Point, b: Point): number => {
  return (a.y + b.y) / 2;
};

export const applyBasicPerspectiveCorrection = ({
  referenceMidY,
  targetMidY,
  imageHeight,
  strength = 0.35,
}: PerspectiveCorrectionInput): number => {
  if (imageHeight <= 0) {
    return 1;
  }

  const normalizedDelta = (referenceMidY - targetMidY) / imageHeight;
  const rawFactor = 1 + normalizedDelta * strength;

  return Math.min(1.3, Math.max(0.7, rawFactor));
};
