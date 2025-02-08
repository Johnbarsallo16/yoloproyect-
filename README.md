# Detector de Movimientos y Posturas con YOLO y MediaPipe

Este proyecto implementa un sistema de detección de movimientos y posturas en tiempo real utilizando YOLOv8 y MediaPipe.

## Características

- Detección de personas en tiempo real con YOLOv8
- Seguimiento de posturas con MediaPipe
- Detección de transiciones de movimiento:
  - De pie
  - Sentado
  - Agachado
  - Inclinado
  - En movimiento
- Panel de información en tiempo real
- Visualización de ángulos de rodilla y cadera
- Historial de transiciones de movimiento

## Requisitos

- Python 3.8+
- OpenCV
- MediaPipe
- Ultralytics (YOLOv8)
- NumPy

## Instalación

1. Clonar el repositorio:
```bash
git clone https://github.com/Johnbarsallo16/yoloproyect-
```

2. Instalar dependencias:
```bash
pip install ultralytics opencv-python mediapipe numpy
```

## Uso

Ejecutar el script principal:
```bash
python detector_personas.py
```

- Presionar 'q' para salir del programa
- Asegurarse de estar completamente visible en la cámara para mejor detección

## Funcionalidades

- Detección de personas con alta precisión
- Seguimiento de posturas en tiempo real
- Cálculo de ángulos corporales
- Detección de transiciones de movimiento
- Panel informativo con datos en tiempo real 