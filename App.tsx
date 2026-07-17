import { CameraView, useCameraPermissions } from 'expo-camera';
import { StatusBar } from 'expo-status-bar';
import AsyncStorage from '@react-native-async-storage/async-storage';
import { useEffect, useMemo, useRef, useState } from 'react';
import {
  ActivityIndicator,
  Alert,
  Image,
  Linking,
  Pressable,
  SafeAreaView,
  Share,
  ScrollView,
  StyleSheet,
  Text,
  TextInput,
  View,
} from 'react-native';
import {
  Point,
  SavedMeasurement,
  applyBasicPerspectiveCorrection,
  calculateMeasuredLengthCm,
  distance,
  formatCm,
  midpointY,
  parseReferenceCm,
} from './src/utils/measurement';
import { detectLegPointsFromImage } from './src/ai/poseDetector';

const DEFAULT_REFERENCE_CM = '8.56';
const HISTORY_STORAGE_KEY = 'appsize_measurements_v1';
const AI_MIN_CONFIDENCE = 0.2;
const REFERENCE_PRESETS = {
  a4: {
    label: 'Hoja A4 (ancho 21.0 cm)',
    cm: 21.0,
  },
  carta: {
    label: 'Hoja Carta (ancho 21.59 cm)',
    cm: 21.59,
  },
} as const;

type ReferencePreset = keyof typeof REFERENCE_PRESETS;
type MainTab = 'medicion' | 'historial';
type MeasurementMode = 'ia' | 'manual';

type CapturedPhoto = {
  uri: string;
  width: number;
  height: number;
};

export default function App() {
  const [permission, requestPermission] = useCameraPermissions();
  const cameraRef = useRef<CameraView | null>(null);
  const [isTakingPhoto, setIsTakingPhoto] = useState(false);
  const [referenceCmInput, setReferenceCmInput] = useState(DEFAULT_REFERENCE_CM);
  const [selectedReferencePreset, setSelectedReferencePreset] = useState<ReferencePreset | null>(null);
  const [photo, setPhoto] = useState<CapturedPhoto | null>(null);
  const [imageBoxWidth, setImageBoxWidth] = useState(0);
  const [referencePoints, setReferencePoints] = useState<Point[]>([]);
  const [manualTargetPoints, setManualTargetPoints] = useState<Point[]>([]);
  const [aiTargetPoints, setAITargetPoints] = useState<{ knee: Point; ankle: Point } | null>(null);
  const [perspectiveEnabled, setPerspectiveEnabled] = useState(true);
  const [history, setHistory] = useState<SavedMeasurement[]>([]);
  const [isHistoryLoading, setIsHistoryLoading] = useState(true);
  const [isDetectingLeg, setIsDetectingLeg] = useState(false);
  const [lastAIDetectionInfo, setLastAIDetectionInfo] = useState<string | null>(null);
  const [aiDetectionMeta, setAIDetectionMeta] = useState<{ confidence: number; side: 'left' | 'right' } | null>(
    null,
  );
  const [aiAutoAttempted, setAIAutoAttempted] = useState(false);
  const [activeTab, setActiveTab] = useState<MainTab>('medicion');
  const [measurementMode, setMeasurementMode] = useState<MeasurementMode>('ia');

  const hasReference = referencePoints.length === 2;
  const selectedTargetPoints = measurementMode === 'manual'
    ? manualTargetPoints
    : aiTargetPoints
      ? [aiTargetPoints.knee, aiTargetPoints.ankle]
      : [];
  const canMeasure = hasReference && selectedTargetPoints.length === 2;
  const referenceCm = selectedReferencePreset
    ? REFERENCE_PRESETS[selectedReferencePreset].cm
    : parseReferenceCm(referenceCmInput);
  const cameraGranted = permission?.granted ?? false;

  const measurement = useMemo(() => {
    if (!canMeasure || !referenceCm) {
      return null;
    }

    const referencePixelDistance = distance(referencePoints[0], referencePoints[1]);
    const targetPixelDistance = distance(selectedTargetPoints[0], selectedTargetPoints[1]);
    const measuredCm = calculateMeasuredLengthCm({
      referenceCm,
      referencePixelDistance,
      targetPixelDistance,
    });

    if (!measuredCm) {
      return null;
    }

    const referenceMidY = midpointY(referencePoints[0], referencePoints[1]);
    const targetMidY = midpointY(selectedTargetPoints[0], selectedTargetPoints[1]);
    const perspectiveFactor = photo
      ? applyBasicPerspectiveCorrection({
          referenceMidY,
          targetMidY,
          imageHeight: photo.height,
        })
      : 1;

    return {
      measuredCm,
      correctedCm: perspectiveEnabled ? measuredCm * perspectiveFactor : measuredCm,
      perspectiveFactor,
      referencePixelDistance,
      targetPixelDistance,
    };
  }, [canMeasure, perspectiveEnabled, photo, referenceCm, referencePoints, selectedTargetPoints]);

  const aiLowConfidence = Boolean(
    aiDetectionMeta && aiDetectionMeta.confidence < AI_MIN_CONFIDENCE,
  );

  useEffect(() => {
    const loadHistory = async () => {
      try {
        const raw = await AsyncStorage.getItem(HISTORY_STORAGE_KEY);
        if (!raw) {
          return;
        }

        const parsed = JSON.parse(raw) as SavedMeasurement[];
        if (Array.isArray(parsed)) {
          setHistory(parsed);
        }
      } catch {
        Alert.alert('Historial no disponible', 'No se pudo cargar el historial guardado.');
      } finally {
        setIsHistoryLoading(false);
      }
    };

    loadHistory();
  }, []);

  useEffect(() => {
    if (
      measurementMode !== 'ia' ||
      !photo ||
      referencePoints.length !== 2 ||
      isDetectingLeg ||
      aiTargetPoints ||
      aiAutoAttempted
    ) {
      return;
    }

    setAIAutoAttempted(true);
    handleDetectLegWithAI();
  }, [aiAutoAttempted, aiTargetPoints, isDetectingLeg, measurementMode, photo, referencePoints.length]);

  const persistHistory = async (nextHistory: SavedMeasurement[]) => {
    setHistory(nextHistory);
    await AsyncStorage.setItem(HISTORY_STORAGE_KEY, JSON.stringify(nextHistory));
  };

  const handleSaveMeasurement = async () => {
    if (!referenceCm) {
      Alert.alert('Falta referencia', 'Debes ingresar o seleccionar una referencia valida en cm.');
      return;
    }

    if (!measurement) {
      if (referencePoints.length < 2) {
        Alert.alert('Faltan puntos', 'Marca primero 2 puntos de referencia.');
        return;
      }

      if (measurementMode === 'ia') {
        Alert.alert(
          'Falta deteccion IA',
          'Aun no hay deteccion de pierna. Presiona "Detectar una pierna con IA" o cambia a modo manual.',
        );
        return;
      }

      Alert.alert('Faltan puntos manuales', 'Marca 2 puntos manuales de altura para poder guardar.');
      return;
    }

    try {
      const item: SavedMeasurement = {
        id: `${Date.now()}`,
        createdAt: new Date().toISOString(),
        referenceCm,
        measuredCm: measurement.measuredCm,
        correctedCm: perspectiveEnabled ? measurement.correctedCm : null,
        perspectiveEnabled,
        referencePixelDistance: measurement.referencePixelDistance,
        targetPixelDistance: measurement.targetPixelDistance,
        aiConfidence: aiDetectionMeta?.confidence ?? null,
        aiSide: aiDetectionMeta?.side ?? null,
      };
      const nextHistory = [item, ...history].slice(0, 30);
      await persistHistory(nextHistory);
      Alert.alert('Guardado', 'La medicion se guardo en el historial.');
    } catch {
      Alert.alert('Error', 'No se pudo guardar la medicion.');
    }
  };

  const handleExportReport = async () => {
    if (!history.length) {
      Alert.alert('Sin datos', 'No hay mediciones en el historial para exportar.');
      return;
    }

    const header =
      'fecha,resultado_cm,base_cm,referencia_cm,perspectiva,ia_confianza,ia_lado,ref_px,obj_px';
    const rows = history.map((item) => {
      const resultCm = item.correctedCm ?? item.measuredCm;
      const confidence = item.aiConfidence !== null ? item.aiConfidence.toFixed(3) : '';
      return [
        item.createdAt,
        formatCm(resultCm),
        formatCm(item.measuredCm),
        formatCm(item.referenceCm),
        item.perspectiveEnabled ? 'on' : 'off',
        confidence,
        item.aiSide ?? '',
        item.referencePixelDistance.toFixed(2),
        item.targetPixelDistance.toFixed(2),
      ].join(',');
    });

    const csv = [header, ...rows].join('\n');

    try {
      await Share.share({
        title: 'Reporte APP-Size',
        message: csv,
      });
    } catch {
      Alert.alert('Error', 'No se pudo abrir el menú para exportar el reporte.');
    }
  };

  const handleClearHistory = async () => {
    try {
      await AsyncStorage.removeItem(HISTORY_STORAGE_KEY);
      setHistory([]);
    } catch {
      Alert.alert('Error', 'No se pudo limpiar el historial.');
    }
  };

  const handleTakePhoto = async () => {
    if (!cameraRef.current) {
      return;
    }

    try {
      setIsTakingPhoto(true);
      const result = await cameraRef.current.takePictureAsync({ quality: 1 });
      if (!result?.uri || !result?.width || !result?.height) {
        Alert.alert('No se pudo capturar la foto', 'Intenta nuevamente.');
        return;
      }

      setPhoto({
        uri: result.uri,
        width: result.width,
        height: result.height,
      });
      setReferencePoints([]);
      setManualTargetPoints([]);
      setAITargetPoints(null);
      setAIDetectionMeta(null);
      setLastAIDetectionInfo(null);
      setAIAutoAttempted(false);
    } catch {
      Alert.alert('Error de cámara', 'Hubo un problema al tomar la foto.');
    } finally {
      setIsTakingPhoto(false);
    }
  };

  const handleAddPoint = (x: number, y: number) => {
    if (!photo || !imageBoxWidth) {
      return;
    }

    const scale = photo.width / imageBoxWidth;
    const pixelPoint = {
      x: x * scale,
      y: y * scale,
    };

    if (referencePoints.length < 2) {
      setAIDetectionMeta(null);
      setAITargetPoints(null);
      setManualTargetPoints([]);
      setLastAIDetectionInfo(null);
      setAIAutoAttempted(false);
      setReferencePoints((prev) => [...prev, pixelPoint]);
      return;
    }

    if (measurementMode === 'manual' && manualTargetPoints.length < 2) {
      setManualTargetPoints((prev) => [...prev, pixelPoint]);
    }
  };

  const handleUndoPoint = () => {
    if (measurementMode === 'manual' && manualTargetPoints.length > 0) {
      setManualTargetPoints((prev) => prev.slice(0, Math.max(0, prev.length - 1)));
      return;
    }

    if (referencePoints.length > 0) {
      setReferencePoints((prev) => prev.slice(0, Math.max(0, prev.length - 1)));
      setAIDetectionMeta(null);
      setAITargetPoints(null);
      setLastAIDetectionInfo(null);
      setAIAutoAttempted(false);
    }
  };

  const handleDetectLegWithAI = async () => {
    if (!photo) {
      Alert.alert('Toma una foto', 'Primero debes capturar una imagen.');
      return;
    }

    if (referencePoints.length < 2) {
      Alert.alert(
        'Falta referencia',
        'Primero marca los puntos 1 y 2 de la referencia para poder convertir la estimacion a cm.',
      );
      return;
    }

    try {
      setIsDetectingLeg(true);
      setLastAIDetectionInfo(null);

      const estimate = await detectLegPointsFromImage(photo.uri);
      if (!estimate) {
        Alert.alert(
          'IA sin deteccion',
          'No se detecto rodilla/tobillo con suficiente calidad. Cambia a modo manual para marcar 2 puntos de altura.',
        );
        return;
      }

      setAITargetPoints({
        knee: estimate.knee,
        ankle: estimate.ankle,
      });
      setAIDetectionMeta({ confidence: estimate.confidence, side: estimate.side });
      setLastAIDetectionInfo(
        `Pierna: ${estimate.side} | Confianza aprox.: ${(estimate.confidence * 100).toFixed(1)}%`,
      );
    } catch {
      Alert.alert('Error IA', 'No se pudo ejecutar la deteccion de pose en este dispositivo.');
    } finally {
      setIsDetectingLeg(false);
    }
  };

  const imageHeight = photo && imageBoxWidth ? (photo.height / photo.width) * imageBoxWidth : 0;

  if (!permission) {
    return (
      <SafeAreaView style={styles.centeredScreen}>
        <ActivityIndicator size="large" color="#1f6d58" />
      </SafeAreaView>
    );
  }

  return (
    <SafeAreaView style={styles.screen}>
      <StatusBar style="dark" />
      <ScrollView
        contentContainerStyle={styles.content}
        style={styles.mainScroll}
        scrollEnabled={!(activeTab === 'medicion' && Boolean(photo))}
      >
        <Text style={styles.title}>Medición por cámara</Text>
        <Text style={styles.helperText}>
          1) Coloca un objeto de referencia visible en la foto. 2) Marca dos puntos de la referencia.
          3) Marca dos puntos del objeto o usa IA para detectar rodilla y tobillo automaticamente.
        </Text>
        <View style={styles.tabsContainer}>
          <Pressable
            style={[styles.tabPill, activeTab === 'medicion' && styles.tabPillActive]}
            onPress={() => setActiveTab('medicion')}
          >
            <Text style={[styles.tabPillText, activeTab === 'medicion' && styles.tabPillTextActive]}>
              Realizar medicion
            </Text>
          </Pressable>
          <Pressable
            style={[styles.tabPill, activeTab === 'historial' && styles.tabPillActive]}
            onPress={() => setActiveTab('historial')}
          >
            <Text style={[styles.tabPillText, activeTab === 'historial' && styles.tabPillTextActive]}>
              Historial
            </Text>
          </Pressable>
        </View>

        {activeTab === 'medicion' ? (
          <>
            <View style={styles.card}>
              <Text style={styles.label}>Modo de medicion</Text>
              <View style={styles.rowButtons}>
                <Pressable
                  style={[styles.secondaryButton, measurementMode === 'ia' && styles.selectedReferenceButton]}
                  onPress={() => {
                    setMeasurementMode('ia');
                    setManualTargetPoints([]);
                  }}
                >
                  <Text
                    style={[
                      styles.secondaryButtonText,
                      measurementMode === 'ia' && styles.selectedReferenceButtonText,
                    ]}
                  >
                    IA (2 puntos)
                  </Text>
                </Pressable>
                <Pressable
                  style={[styles.secondaryButton, measurementMode === 'manual' && styles.selectedReferenceButton]}
                  onPress={() => {
                    setMeasurementMode('manual');
                    setAIDetectionMeta(null);
                    setAITargetPoints(null);
                    setLastAIDetectionInfo(null);
                  }}
                >
                  <Text
                    style={[
                      styles.secondaryButtonText,
                      measurementMode === 'manual' && styles.selectedReferenceButtonText,
                    ]}
                  >
                    Manual (2+2)
                  </Text>
                </Pressable>
              </View>
              <Text style={styles.caption}>
                {measurementMode === 'ia'
                  ? 'Marca 2 puntos de referencia y la IA estimara rodilla/tobillo.'
                  : 'Marca 2 puntos de referencia y luego 2 puntos manuales de la altura.'}
              </Text>
            </View>

            <View style={styles.card}>
              <Text style={styles.label}>Referencia conocida (cm)</Text>
              <View style={styles.rowButtons}>
                <Pressable
                  style={[
                    styles.secondaryButton,
                    selectedReferencePreset === 'a4' && styles.selectedReferenceButton,
                  ]}
                  onPress={() => {
                    setSelectedReferencePreset((prev) => (prev === 'a4' ? null : 'a4'));
                  }}
                >
                  <Text
                    style={[
                      styles.secondaryButtonText,
                      selectedReferencePreset === 'a4' && styles.selectedReferenceButtonText,
                    ]}
                  >
                    Hoja A4
                  </Text>
                </Pressable>
                <Pressable
                  style={[
                    styles.secondaryButton,
                    selectedReferencePreset === 'carta' && styles.selectedReferenceButton,
                  ]}
                  onPress={() => {
                    setSelectedReferencePreset((prev) => (prev === 'carta' ? null : 'carta'));
                  }}
                >
                  <Text
                    style={[
                      styles.secondaryButtonText,
                      selectedReferencePreset === 'carta' && styles.selectedReferenceButtonText,
                    ]}
                  >
                    Hoja Carta
                  </Text>
                </Pressable>
              </View>

              <Text style={styles.caption}>
                {selectedReferencePreset
                  ? `Preset seleccionado: ${REFERENCE_PRESETS[selectedReferencePreset].label}`
                  : 'Si no seleccionas A4 o Carta, puedes ingresar una referencia manual.'}
              </Text>

              <TextInput
                value={referenceCmInput}
                onChangeText={setReferenceCmInput}
                keyboardType="decimal-pad"
                style={styles.input}
                editable={!selectedReferencePreset}
                placeholder="Ej: 8.56"
              />
              <Text style={styles.caption}>
                {selectedReferencePreset
                  ? 'Referencia manual deshabilitada mientras haya un preset activo.'
                  : 'Referencia manual activa en centimetros.'}
              </Text>
              <Pressable
                style={styles.toggleButton}
                onPress={() => {
                  setPerspectiveEnabled((prev) => !prev);
                }}
              >
                <Text style={styles.toggleText}>
                  Correccion de perspectiva: {perspectiveEnabled ? 'Activada' : 'Desactivada'}
                </Text>
              </Pressable>
              <Text style={styles.caption}>
                Ajuste basico para escenas inclinadas. Si referencia y objeto estan en el mismo plano, puedes
                dejarla desactivada.
              </Text>
            </View>

            {!photo ? (
              <View style={styles.card}>
                {cameraGranted ? (
                  <>
                    <View style={styles.cameraContainer}>
                      <CameraView ref={cameraRef} style={styles.camera} facing="back" />
                    </View>
                    <Pressable style={styles.primaryButton} onPress={handleTakePhoto} disabled={isTakingPhoto}>
                      <Text style={styles.primaryButtonText}>{isTakingPhoto ? 'Capturando...' : 'Tomar foto'}</Text>
                    </Pressable>
                  </>
                ) : (
                  <>
                    <Text style={styles.label}>Permiso de camara pendiente</Text>
                    <Text style={styles.caption}>
                      Para usar captura directa, habilita la camara. El resto de la interfaz ya esta disponible.
                    </Text>
                    <Pressable style={styles.primaryButton} onPress={requestPermission}>
                      <Text style={styles.primaryButtonText}>Dar permiso de camara</Text>
                    </Pressable>
                  </>
                )}
              </View>
            ) : (
              <View style={styles.card}>
                <Text style={styles.label}>Toca 2 puntos sobre la imagen</Text>
                <Text style={styles.caption}>
                  {referencePoints.length < 2
                    ? 'Paso 1: marca 2 puntos de referencia.'
                    : measurementMode === 'manual'
                      ? 'Paso 2: marca 2 puntos de altura manual.'
                      : 'Paso 2: ejecuta IA para detectar la pierna.'}
                </Text>
                <Text style={styles.captionStrong}>
                  Referencia: {Math.min(referencePoints.length, 2)}/2
                  {measurementMode === 'manual' ? ` | Altura: ${manualTargetPoints.length}/2` : ''}
                </Text>

                <View
                  style={styles.imageBox}
                  onLayout={(event) => {
                    setImageBoxWidth(event.nativeEvent.layout.width);
                  }}
                >
                  <Image source={{ uri: photo.uri }} style={[styles.image, { height: imageHeight || 300 }]} />

                  {imageHeight > 0 && (
                    <Pressable
                      style={[styles.tapLayer, { height: imageHeight }]}
                      onPress={(event) => {
                        handleAddPoint(event.nativeEvent.locationX, event.nativeEvent.locationY);
                      }}
                    >
                      {[...referencePoints, ...selectedTargetPoints].map((point, index) => {
                        const displayScale = imageBoxWidth / photo.width;
                        const left = point.x * displayScale - 8;
                        const top = point.y * displayScale - 8;
                        return (
                          <View key={`${point.x}-${point.y}-${index}`} style={[styles.point, { left, top }]}>
                            <Text style={styles.pointText}>{index + 1}</Text>
                          </View>
                        );
                      })}
                    </Pressable>
                  )}
                </View>

                <View style={styles.rowButtons}>
                  <Pressable
                    style={styles.secondaryButton}
                    onPress={() => {
                      setReferencePoints([]);
                      setManualTargetPoints([]);
                      setAITargetPoints(null);
                      setAIDetectionMeta(null);
                      setLastAIDetectionInfo(null);
                      setAIAutoAttempted(false);
                    }}
                  >
                    <Text style={styles.secondaryButtonText}>Reiniciar puntos</Text>
                  </Pressable>
                  <Pressable
                    style={[
                      styles.secondaryButton,
                      referencePoints.length === 0 && manualTargetPoints.length === 0 && styles.disabledButton,
                    ]}
                    onPress={handleUndoPoint}
                    disabled={referencePoints.length === 0 && manualTargetPoints.length === 0}
                  >
                    <Text style={styles.secondaryButtonText}>Deshacer punto</Text>
                  </Pressable>
                  <Pressable
                    style={styles.secondaryButton}
                    onPress={() => {
                      setPhoto(null);
                      setReferencePoints([]);
                      setManualTargetPoints([]);
                      setAITargetPoints(null);
                      setAIDetectionMeta(null);
                      setLastAIDetectionInfo(null);
                      setAIAutoAttempted(false);
                    }}
                  >
                    <Text style={styles.secondaryButtonText}>Nueva foto</Text>
                  </Pressable>
                </View>

                {measurementMode === 'ia' && (
                  <Pressable
                    style={[
                      styles.primaryButton,
                      (isDetectingLeg || referencePoints.length < 2) && styles.disabledButton,
                    ]}
                    disabled={isDetectingLeg || referencePoints.length < 2}
                    onPress={handleDetectLegWithAI}
                  >
                    <Text style={styles.primaryButtonText}>
                      {isDetectingLeg
                        ? 'Detectando...'
                        : aiDetectionMeta
                          ? 'Recalcular IA'
                          : 'Detectar una pierna con IA'}
                    </Text>
                  </Pressable>
                )}

                {referencePoints.length < 2 && (
                  <Text style={styles.caption}>Marca primero los puntos 1 y 2 de la referencia.</Text>
                )}
                {measurementMode === 'ia' && referencePoints.length === 2 && !aiDetectionMeta && !isDetectingLeg && (
                  <Text style={styles.caption}>
                    Aun no hay deteccion IA. Presiona el boton de IA para estimar rodilla/tobillo.
                  </Text>
                )}
                {measurementMode === 'manual' && referencePoints.length === 2 && manualTargetPoints.length < 2 && (
                  <Text style={styles.caption}>Marca ahora 2 puntos manuales para la altura estimada.</Text>
                )}

                {lastAIDetectionInfo && <Text style={styles.caption}>{lastAIDetectionInfo}</Text>}

                {aiLowConfidence && (
                  <Text style={styles.errorText}>
                    Deteccion IA con confianza baja ({(aiDetectionMeta!.confidence * 100).toFixed(1)}%).
                    Puedes usarla, pero intenta mejorar iluminacion para mayor precision.
                  </Text>
                )}

                {!referenceCm && (
                  <Text style={styles.errorText}>Ingresa una longitud de referencia válida en cm.</Text>
                )}

                <View style={styles.resultBox}>
                  <Text style={styles.resultTitle}>Resultado</Text>
                  <Text style={styles.resultText}>
                    {measurement
                      ? `${formatCm(measurement.correctedCm)} cm`
                      : 'Marca 2 puntos de referencia y ejecuta la IA para obtener la medicion.'}
                  </Text>
                  {measurement && (
                    <>
                      <Text style={styles.caption}>
                        Ref(px): {measurement.referencePixelDistance.toFixed(1)} | Obj(px):{' '}
                        {measurement.targetPixelDistance.toFixed(1)}
                      </Text>
                      <Text style={styles.caption}>
                        Base: {formatCm(measurement.measuredCm)} cm | Factor perspectiva:{' '}
                        {measurement.perspectiveFactor.toFixed(3)}
                      </Text>
                    </>
                  )}
                </View>

                <Pressable
                  style={styles.primaryButton}
                  onPress={handleSaveMeasurement}
                >
                  <Text style={styles.primaryButtonText}>Guardar medicion</Text>
                </Pressable>
              </View>
            )}
          </>
        ) : (
          <View style={styles.card}>
            <View style={styles.historyHeader}>
              <Text style={styles.label}>Historial</Text>
              <View style={styles.historyActions}>
                <Pressable style={styles.ghostButton} onPress={handleExportReport}>
                  <Text style={styles.ghostButtonText}>Exportar</Text>
                </Pressable>
                <Pressable style={styles.ghostButton} onPress={handleClearHistory}>
                  <Text style={styles.ghostButtonText}>Limpiar</Text>
                </Pressable>
              </View>
            </View>

            {isHistoryLoading ? (
              <Text style={styles.caption}>Cargando historial...</Text>
            ) : history.length === 0 ? (
              <Text style={styles.caption}>Aun no hay mediciones guardadas.</Text>
            ) : (
              history.map((item) => (
                <View key={item.id} style={styles.historyItem}>
                  <Text style={styles.historyMain}>
                    {formatCm(item.correctedCm ?? item.measuredCm)} cm
                  </Text>
                  <Text style={styles.caption}>
                    Ref: {formatCm(item.referenceCm)} cm | Base: {formatCm(item.measuredCm)} cm
                  </Text>
                  <Text style={styles.caption}>
                    {new Date(item.createdAt).toLocaleString()} | Perspectiva:{' '}
                    {item.perspectiveEnabled ? 'On' : 'Off'}
                  </Text>
                  <Text style={styles.caption}>
                    IA: {item.aiConfidence !== null ? `${(item.aiConfidence * 100).toFixed(1)}%` : 'manual'} | Lado:{' '}
                    {item.aiSide ?? '-'}
                  </Text>
                </View>
              ))
            )}
          </View>
        )}

      </ScrollView>

      <View style={styles.footerBox}>
        <Pressable
          onPress={() => {
            Linking.openURL('https://www.maxinfo.cl');
          }}
        >
          <Text style={styles.footerText}>Realizado por MAXINFO</Text>
        </Pressable>
      </View>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  screen: {
    flex: 1,
    backgroundColor: '#edf4fb',
  },
  centeredScreen: {
    flex: 1,
    backgroundColor: '#edf4fb',
    justifyContent: 'center',
    alignItems: 'center',
    padding: 20,
  },
  content: {
    padding: 16,
    gap: 12,
    paddingBottom: 24,
  },
  mainScroll: {
    flex: 1,
  },
  title: {
    fontSize: 28,
    fontWeight: '800',
    color: '#124a8a',
  },
  helperText: {
    fontSize: 14,
    lineHeight: 20,
    color: '#335b7a',
  },
  card: {
    backgroundColor: '#ffffff',
    borderRadius: 14,
    padding: 14,
    borderWidth: 1,
    borderColor: '#c7dced',
    gap: 10,
  },
  label: {
    fontSize: 16,
    fontWeight: '700',
    color: '#185b9a',
  },
  input: {
    borderWidth: 1,
    borderColor: '#a8c5df',
    borderRadius: 10,
    paddingVertical: 10,
    paddingHorizontal: 12,
    fontSize: 16,
    backgroundColor: '#f6fbff',
  },
  caption: {
    fontSize: 12,
    color: '#4a6d89',
  },
  captionStrong: {
    fontSize: 12,
    color: '#1f5f96',
    fontWeight: '700',
  },
  toggleButton: {
    backgroundColor: '#eaf4fb',
    borderRadius: 8,
    paddingVertical: 8,
    paddingHorizontal: 10,
    alignSelf: 'flex-start',
  },
  toggleText: {
    color: '#1f5f96',
    fontWeight: '700',
    fontSize: 12,
  },
  cameraContainer: {
    width: '100%',
    aspectRatio: 3 / 4,
    overflow: 'hidden',
    borderRadius: 12,
  },
  camera: {
    flex: 1,
  },
  primaryButton: {
    backgroundColor: '#1f6fb2',
    paddingVertical: 12,
    borderRadius: 10,
    alignItems: 'center',
  },
  primaryButtonText: {
    color: '#ffffff',
    fontWeight: '700',
    fontSize: 15,
  },
  disabledButton: {
    opacity: 0.5,
  },
  secondaryButton: {
    flex: 1,
    backgroundColor: '#e8f2fb',
    paddingVertical: 10,
    borderRadius: 10,
    alignItems: 'center',
  },
  secondaryButtonText: {
    color: '#245f94',
    fontWeight: '700',
    fontSize: 13,
  },
  selectedReferenceButton: {
    backgroundColor: '#1f6fb2',
  },
  selectedReferenceButtonText: {
    color: '#ffffff',
  },
  imageBox: {
    width: '100%',
    position: 'relative',
  },
  image: {
    width: '100%',
    borderRadius: 12,
  },
  tapLayer: {
    position: 'absolute',
    left: 0,
    top: 0,
    right: 0,
  },
  point: {
    position: 'absolute',
    width: 22,
    height: 22,
    borderRadius: 11,
    backgroundColor: '#e64f4f',
    justifyContent: 'center',
    alignItems: 'center',
    borderWidth: 2,
    borderColor: '#ffffff',
  },
  pointText: {
    color: '#fff',
    fontSize: 11,
    fontWeight: '800',
  },
  rowButtons: {
    flexDirection: 'row',
    gap: 8,
  },
  tabsContainer: {
    flexDirection: 'row',
    gap: 8,
  },
  tabPill: {
    flex: 1,
    backgroundColor: '#e6eef7',
    borderRadius: 999,
    paddingVertical: 10,
    paddingHorizontal: 12,
    alignItems: 'center',
    borderWidth: 1,
    borderColor: '#c9dced',
  },
  tabPillActive: {
    backgroundColor: '#1f6fb2',
    borderColor: '#1f6fb2',
  },
  tabPillText: {
    color: '#1e5f95',
    fontWeight: '700',
    fontSize: 13,
  },
  tabPillTextActive: {
    color: '#ffffff',
  },
  errorText: {
    color: '#b43636',
    fontSize: 13,
    fontWeight: '600',
  },
  resultBox: {
    backgroundColor: '#eff6fc',
    borderRadius: 10,
    padding: 12,
    borderWidth: 1,
    borderColor: '#c7dced',
    gap: 4,
  },
  resultTitle: {
    fontSize: 14,
    fontWeight: '700',
    color: '#1f5f96',
  },
  resultText: {
    fontSize: 24,
    fontWeight: '800',
    color: '#124a8a',
  },
  historyHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
  },
  historyActions: {
    flexDirection: 'row',
    gap: 8,
  },
  ghostButton: {
    backgroundColor: '#edf6fc',
    borderRadius: 8,
    paddingVertical: 6,
    paddingHorizontal: 10,
  },
  ghostButtonText: {
    color: '#245f94',
    fontSize: 12,
    fontWeight: '700',
  },
  historyItem: {
    borderWidth: 1,
    borderColor: '#c7dced',
    borderRadius: 10,
    padding: 10,
    gap: 2,
    backgroundColor: '#f8fbff',
  },
  historyMain: {
    fontSize: 20,
    fontWeight: '800',
    color: '#1c5a92',
  },
  footerBox: {
    alignItems: 'center',
    justifyContent: 'center',
    paddingVertical: 10,
    borderTopWidth: 1,
    borderTopColor: '#c7dced',
    backgroundColor: '#edf4fb',
  },
  footerText: {
    fontSize: 12,
    color: 'rgba(31, 111, 178, 0.55)',
    fontWeight: '700',
  },
});
