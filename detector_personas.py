import cv2
import numpy as np
from ultralytics import YOLO
from collections import defaultdict, deque
import mediapipe as mp
import time

# Inicializar YOLO
modelo = YOLO("yolov8n.pt")

# Inicializar MediaPipe Pose
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
pose = mp_pose.Pose(
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7,
    model_complexity=1  # 0=Lite, 1=Full, 2=Heavy
)

# Diccionario completo de traducción de clases al español
TRADUCCION_CLASES = {
    'person': 'persona',
    'bicycle': 'bicicleta',
    'car': 'carro',
    'motorcycle': 'motocicleta',
    'airplane': 'avión',
    'bus': 'autobús',
    'train': 'tren',
    'truck': 'camión',
    'boat': 'bote',
    'traffic light': 'semáforo',
    'fire hydrant': 'hidrante',
    'stop sign': 'señal de alto',
    'parking meter': 'parquímetro',
    'bench': 'banca',
    'bird': 'pájaro',
    'cat': 'gato',
    'dog': 'perro',
    'horse': 'caballo',
    'sheep': 'oveja',
    'cow': 'vaca',
    'elephant': 'elefante',
    'bear': 'oso',
    'zebra': 'cebra',
    'giraffe': 'jirafa',
    'backpack': 'mochila',
    'umbrella': 'paraguas',
    'handbag': 'bolso',
    'tie': 'corbata',
    'suitcase': 'maleta',
    'frisbee': 'frisbee',
    'skis': 'esquís',
    'snowboard': 'tabla de nieve',
    'sports ball': 'pelota deportiva',
    'kite': 'cometa',
    'baseball bat': 'bate de béisbol',
    'baseball glove': 'guante de béisbol',
    'skateboard': 'patineta',
    'surfboard': 'tabla de surf',
    'tennis racket': 'raqueta de tenis',
    'bottle': 'botella',
    'wine glass': 'copa de vino',
    'cup': 'taza',
    'fork': 'tenedor',
    'knife': 'cuchillo',
    'spoon': 'cuchara',
    'bowl': 'tazón',
    'banana': 'plátano',
    'apple': 'manzana',
    'sandwich': 'sándwich',
    'orange': 'naranja',
    'broccoli': 'brócoli',
    'carrot': 'zanahoria',
    'hot dog': 'perro caliente',
    'pizza': 'pizza',
    'donut': 'dona',
    'cake': 'pastel',
    'chair': 'silla',
    'couch': 'sofá',
    'potted plant': 'planta en maceta',
    'bed': 'cama',
    'dining table': 'mesa de comedor',
    'toilet': 'inodoro',
    'tv': 'televisión',
    'laptop': 'laptop',
    'mouse': 'ratón',
    'remote': 'control remoto',
    'keyboard': 'teclado',
    'cell phone': 'celular',
    'microwave': 'microondas',
    'oven': 'horno',
    'toaster': 'tostadora',
    'sink': 'lavabo',
    'refrigerator': 'refrigerador',
    'book': 'libro',
    'clock': 'reloj',
    'vase': 'jarrón',
    'scissors': 'tijeras',
    'teddy bear': 'oso de peluche',
    'hair drier': 'secador',
    'toothbrush': 'cepillo de dientes'
}

# Inicializar la cámara
cap = cv2.VideoCapture(0)

# Configuración de detección
min_confianza_deteccion = 0.45  # Reducir umbral para detectar más objetos

# Colores para diferentes clases (colores más brillantes)
COLORES = np.random.randint(100, 255, size=(80, 3), dtype=np.uint8)

# Función para calcular ángulos entre puntos
def calcular_angulo(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angulo = np.abs(radians*180.0/np.pi)
    
    if angulo > 180.0:
        angulo = 360-angulo
    
    return angulo

# Configuración para detección de movimientos
class DetectorMovimientos:
    def __init__(self):
        self.historial_angulos = {
            'cadera': deque(maxlen=10),
            'rodilla': deque(maxlen=10),
            'tobillo': deque(maxlen=10)
        }
        self.estado_actual = "De pie"
        self.tiempo_ultimo_cambio = time.time()
        self.umbral_tiempo = 1.0  # Tiempo mínimo entre transiciones
        self.historial_transiciones = deque(maxlen=3)  # Guardar últimas 3 transiciones
        
    def actualizar_angulos(self, angulos):
        for parte, angulo in angulos.items():
            self.historial_angulos[parte].append(angulo)
    
    def detectar_postura(self, angulo_rodilla, angulo_cadera):
        # Definir umbrales para diferentes posturas
        if angulo_rodilla > 160:  # Piernas rectas
            if angulo_cadera > 160:  # Torso recto
                return "De pie"
            else:
                return "Inclinado"
        elif angulo_rodilla < 120:  # Rodillas flexionadas
            if angulo_cadera < 120:  # Torso inclinado
                return "Agachado"
            else:
                return "Sentado"
        return "En movimiento"
    
    def detectar_transicion(self, angulo_rodilla, angulo_cadera):
        tiempo_actual = time.time()
        if tiempo_actual - self.tiempo_ultimo_cambio < self.umbral_tiempo:
            return None
        
        nueva_postura = self.detectar_postura(angulo_rodilla, angulo_cadera)
        if nueva_postura != self.estado_actual:
            transicion = f"{self.estado_actual} → {nueva_postura}"
            self.estado_actual = nueva_postura
            self.tiempo_ultimo_cambio = tiempo_actual
            self.historial_transiciones.append(transicion)
            return transicion
        return None

# Inicializar detector de movimientos
detector_movimientos = DetectorMovimientos()

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Crear un panel de información
    panel_altura = 200
    panel = np.zeros((panel_altura, frame.shape[1], 3), dtype=np.uint8)
    
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    resultados_pose = pose.process(frame_rgb)
    resultados = modelo(frame, conf=min_confianza_deteccion)
    
    # Variables para almacenar ángulos
    angulos = {}
    
    if resultados_pose.pose_landmarks:
        # Dibujar pose con líneas más gruesas y puntos más grandes
        mp_drawing.draw_landmarks(
            frame,
            resultados_pose.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=4, circle_radius=6),
            connection_drawing_spec=mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=3)
        )
        
        landmarks = resultados_pose.pose_landmarks.landmark
        
        if all(landmarks[punto].visibility > 0.5 for punto in [
            mp_pose.PoseLandmark.RIGHT_HIP,
            mp_pose.PoseLandmark.RIGHT_KNEE,
            mp_pose.PoseLandmark.RIGHT_ANKLE,
            mp_pose.PoseLandmark.RIGHT_SHOULDER
        ]):
            # Calcular puntos y ángulos como antes...
            cadera = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP].x * frame.shape[1],
                     landmarks[mp_pose.PoseLandmark.RIGHT_HIP].y * frame.shape[0]]
            rodilla = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE].x * frame.shape[1],
                      landmarks[mp_pose.PoseLandmark.RIGHT_KNEE].y * frame.shape[0]]
            tobillo = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE].x * frame.shape[1],
                      landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE].y * frame.shape[0]]
            hombro = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].x * frame.shape[1],
                     landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].y * frame.shape[0]]
            
            angulo_rodilla = calcular_angulo(cadera, rodilla, tobillo)
            angulo_cadera = calcular_angulo(hombro, cadera, rodilla)
            
            angulos['rodilla'] = angulo_rodilla
            angulos['cadera'] = angulo_cadera
            
            detector_movimientos.actualizar_angulos(angulos)
            transicion = detector_movimientos.detectar_transicion(angulo_rodilla, angulo_cadera)
            
            # Dibujar líneas para los ángulos
            cv2.line(frame, (int(cadera[0]), int(cadera[1])), (int(rodilla[0]), int(rodilla[1])), (0, 255, 255), 2)
            cv2.line(frame, (int(rodilla[0]), int(rodilla[1])), (int(tobillo[0]), int(tobillo[1])), (0, 255, 255), 2)
            
            # Actualizar panel de información
            # Título
            cv2.putText(panel, "INFORMACIÓN DE POSTURA", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
            # Ángulos
            cv2.putText(panel, f'Ángulo Rodilla: {int(angulo_rodilla)}°', (10, 70),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.putText(panel, f'Ángulo Cadera: {int(angulo_cadera)}°', (10, 100),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            
            # Estado actual
            cv2.putText(panel, f'Estado Actual: {detector_movimientos.estado_actual}', (10, 130),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            
            # Última transición
            if detector_movimientos.historial_transiciones:
                cv2.putText(panel, f'Última Transición: {detector_movimientos.historial_transiciones[-1]}', 
                           (10, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 2)
    
    # Procesar detecciones de YOLO (mantener solo para personas)
    for r in resultados:
        boxes = r.boxes
        for box in boxes:
            if box.conf[0].item() > min_confianza_deteccion:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                clase = int(box.cls[0].item())
                nombre_clase = modelo.names[clase]
                
                if nombre_clase.lower() == 'person':
                    nombre_clase_esp = TRADUCCION_CLASES.get(nombre_clase.lower(), nombre_clase)
                    color = (0, 255, 0)  # Verde fijo para personas
                    grosor = 3  # Grosor fijo para mejor visibilidad
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, grosor)
                    
                    confianza_txt = f'{nombre_clase_esp} ({int(box.conf[0].item() * 100)}%)'
                    # Fondo negro semi-transparente para el texto
                    (tw, th), _ = cv2.getTextSize(confianza_txt, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
                    cv2.rectangle(frame, (x1, y1-30), (x1 + tw, y1), color, -1)
                    cv2.putText(frame, confianza_txt, (x1, y1-5),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
    
    # Combinar frame con panel de información
    frame_completo = np.vstack([frame, panel])
    
    # Mostrar el frame
    cv2.imshow('Detector de Movimientos y Posturas', frame_completo)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar recursos
cap.release()
cv2.destroyAllWindows() 