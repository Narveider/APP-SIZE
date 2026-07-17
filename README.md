# APP-Size

Aplicacion movil (Expo + React Native + TypeScript) para estimar longitudes usando la camara del telefono.

Nuevas funciones incluidas:

- Correccion basica de perspectiva (opcional) para reducir error cuando la escena tiene inclinacion.
- Historial local de mediciones guardado en el telefono.
- Deteccion automatica de rodilla y tobillo con IA gratuita on-device (MoveNet).

## Como funciona

La medicion se basa en proporcion de pixeles:

1. Defines una referencia de longitud conocida en cm (por ejemplo, una tarjeta de 8.56 cm).
1. Tomas una foto donde se vea esa referencia y el objeto a medir.
1. Marcas 4 puntos sobre la imagen.

- Puntos 1 y 2: extremos de la referencia.
- Puntos 3 y 4: extremos del objeto (manual) o rodilla/tobillo detectados por IA.

1. La app calcula:

`longitud_objeto_cm = (pixeles_objeto / pixeles_referencia) * referencia_cm`

Si activas la correccion basica de perspectiva, se aplica un factor adicional en funcion de la posicion vertical de referencia y objeto en la imagen.

## IA gratuita recomendada para tu caso

Para estimar longitud en personas postradas sin pagos ni suscripciones, la opcion recomendada es:

- MoveNet (TensorFlow):
  - Gratis y open source (Apache 2.0).
  - Corre en dispositivo (sin backend de pago).
  - Detecta keypoints de cuerpo, incluyendo rodilla y tobillo.

Alternativa tambien gratuita:

- MediaPipe BlazePose:
  - Gratis y open source.
  - Muy buena para pose humana.

En este proyecto se implemento MoveNet para deteccion automatica de rodilla/tobillo desde foto.

## Entrenamiento con dataset

Se agrego un pipeline de entrenamiento en `training/` para reducir fallos con datos reales:

- Conversion de COCO a formato pose de 4 keypoints (rodilla/tobillo).
- Entrenamiento por transferencia con YOLO Pose.
- Guia paso a paso en `training/README.md`.

Recomendacion: combinar COCO con fotos reales del escenario clinico para mejorar robustez en pacientes postrados.

### Flujo para pacientes postrados

1. Captura una foto lateral donde se vea bien la pierna.
2. Marca primero la referencia conocida (puntos 1 y 2).
3. Pulsa "Detectar rodilla/tobillo con IA".
4. Revisa la confianza reportada y el resultado en cm.
5. Si la deteccion falla, usa modo manual para puntos 3 y 4.

## Ejecutar

Requisitos:

- Node.js LTS
- npm
- Expo Go instalado en el telefono

Comandos:

```bash
npm install
npm run start
```

Luego:

- Android: escanea el QR con Expo Go
- iOS: escanea el QR con la camara (o usa Expo Go)

Tambien puedes usar:

```bash
npm run android
npm run ios
npm run web
```

## Limitaciones importantes

- Es una estimacion, no un instrumento metrologico certificado.
- La precision depende de:
  - Que referencia y objeto esten en el mismo plano.
  - Distorsion de lente y perspectiva.
  - Calidad de la foto y precision al tocar puntos.
- Si referencia y objeto no estan alineados en profundidad, habra error.
- Para mejores resultados: buena luz, pulso estable, toma perpendicular y referencia cerca del objeto.
- La IA puede fallar con oclusion (sabanas), poca luz o postura fuera de plano lateral.

## Historial de mediciones

- Puedes guardar una medicion desde la pantalla principal.
- Se almacenan hasta 30 registros en el dispositivo.
- Incluye fecha, medicion base, medicion corregida, estado de perspectiva, confianza IA y lado detectado.
- Puedes limpiar el historial cuando lo necesites.
- Puedes exportar el reporte en formato CSV usando el boton "Exportar".

## Controles de calidad IA (implementados)

- Regla minima de confianza IA: 0.45.
- Si la deteccion de rodilla/tobillo queda por debajo del umbral, se bloquea el guardado para evitar mediciones inestables.
- Siempre hay fallback manual para marcar puntos 3 y 4.

## Estructura principal

- `App.tsx`: flujo UI + camara + captura de puntos.
- `src/utils/measurement.ts`: utilidades de distancia y conversion a cm.
