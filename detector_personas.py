import cv2
import numpy as np
from ultralytics import YOLO
from collections import defaultdict, deque
import mediapipe as mp
import time
import torch

# Inicializar YOLO con la nueva versión y configuración optimizada
modelo = YOLO("yolo11n.pt")
modelo.to('cuda' if torch.cuda.is_available() else 'cpu')  # Usar GPU si está disponible

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

# Ajustar configuración de la cámara para mejor rendimiento con YOLO11
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)  # Resolución recomendada para YOLO11n
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FPS, 30)

# Configuración optimizada para YOLO11
min_confianza_deteccion = 0.55  # Umbral de confianza ajustado para YOLO11
max_detecciones = 10  # Límite de detecciones simultáneas

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
            'tobillo': deque(maxlen=10),
            'codo': deque(maxlen=10),
            'hombro': deque(maxlen=10),
            'muñeca': deque(maxlen=10)
        }
        # Nuevas variables para movimientos dinámicos
        self.historial_posiciones = deque(maxlen=30)  # Para tracking de movimiento
        self.velocidad_movimiento = 0
        self.direccion_actual = None
        self.cambios_direccion = deque(maxlen=5)
        self.estado_movimiento = 'estático'
        self.tiempo_ultimo_cambio_direccion = time.time()
        
        # Umbrales para clasificación de movimiento
        self.umbral_velocidad_caminar = 0.3  # metros por segundo
        self.umbral_velocidad_correr = 0.8   # metros por segundo
        self.umbral_cambio_direccion = 45    # grados
        
        # Mantener el resto de las variables existentes
        self.estado_actual = "De pie"
        self.tiempo_ultimo_cambio = time.time()
        self.umbral_tiempo = 0.5
        self.historial_transiciones = deque(maxlen=5)
        self.visibilidad_partes = {}
        self.estado_extremidades = {
            'brazos': 'neutral',
            'piernas': 'neutral',
            'torso': 'neutral'
        }
        self.historial_posturas = deque(maxlen=30)
        self.transicion_en_progreso = False
        self.inicio_transicion = 0
        self.prediccion_siguiente_postura = None
        
        # Actualizar patrones de transición
        self.patrones_transicion = {
            ('De pie', 'Sentado'): {
                'secuencia': ['De pie', 'Inclinado', 'Sentado'],
                'duracion_tipica': 1.5  # segundos
            },
            ('Sentado', 'De pie'): {
                'secuencia': ['Sentado', 'Inclinado', 'De pie'],
                'duracion_tipica': 1.5
            },
            ('De pie', 'Agachado'): {
                'secuencia': ['De pie', 'Inclinado', 'Agachado'],
                'duracion_tipica': 1.0
            },
            ('Agachado', 'De pie'): {
                'secuencia': ['Agachado', 'Inclinado', 'De pie'],
                'duracion_tipica': 1.0
            },
            ('De pie', 'Acostado'): {
                'secuencia': ['De pie', 'Agachado', 'Acostado'],
                'duracion_tipica': 2.0
            },
            ('Acostado', 'De pie'): {
                'secuencia': ['Acostado', 'Sentado', 'De pie'],
                'duracion_tipica': 2.5
            }
        }
        
        # Nuevas variables para análisis de lenguaje corporal
        self.lenguaje_corporal = {
            'postura_general': 'neutral',
            'brazos': 'neutral',
            'confianza': 'neutral',
            'apertura': 'neutral',
            'tension': 'baja'
        }
        
        # Patrones de lenguaje corporal
        self.patrones_lenguaje_corporal = {
            'brazos_cruzados': {
                'angulos': {'codo_izq': (70, 110), 'codo_der': (70, 110)},
                'distancia_manos_torso': 'cerca',
                'significado': 'postura cerrada/defensiva'
            },
            'brazos_abiertos': {
                'angulos': {'codo_izq': (160, 180), 'codo_der': (160, 180)},
                'distancia_hombros': 'amplia',
                'significado': 'postura abierta/receptiva'
            },
            'encorvado': {
                'angulos': {'columna': (0, 60)},
                'hombros': 'adelante',
                'significado': 'inseguridad/cansancio'
            },
            'erguido': {
                'angulos': {'columna': (80, 100)},
                'hombros': 'atras',
                'significado': 'confianza/atención'
            }
        }
        
        # Ajustar umbrales para mejor detección de posturas
        self.umbrales_postura = {
            'rodilla_flexion': 130,    # Ángulo menor indica rodillas flexionadas
            'cadera_flexion': 130,     # Ángulo menor indica cadera flexionada
            'torso_vertical': 80,      # Ángulo para considerar torso vertical
            'torso_inclinado': 60,     # Ángulo para considerar torso inclinado
            'torso_acostado': 30       # Ángulo para considerar persona acostada
        }
        
        # Tiempo mínimo para confirmar una postura
        self.tiempo_confirmacion_postura = 0.3  # segundos
        self.ultima_postura_tiempo = time.time()
        self.postura_temporal = None
    
    def analizar_movimiento_dinamico(self, landmarks, frame_shape):
        if not landmarks or not self.visibilidad_partes['torso']:
            return
        
        # Obtener posición actual del centro del cuerpo
        centro_cadera = np.array([
            (landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x + 
             landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x) * frame_shape[1] / 2,
            (landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y + 
             landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y) * frame_shape[0] / 2
        ])
        
        tiempo_actual = time.time()
        self.historial_posiciones.append((centro_cadera, tiempo_actual))
        
        if len(self.historial_posiciones) >= 2:
            # Calcular velocidad de movimiento
            pos_anterior, tiempo_anterior = self.historial_posiciones[-2]
            delta_tiempo = tiempo_actual - tiempo_anterior
            if delta_tiempo > 0:
                distancia = np.linalg.norm(centro_cadera - pos_anterior)
                self.velocidad_movimiento = distancia / delta_tiempo
                
                # Determinar dirección del movimiento
                if distancia > 0:
                    direccion = np.arctan2(centro_cadera[1] - pos_anterior[1],
                                         centro_cadera[0] - pos_anterior[0])
                    
                    # Detectar cambio de dirección
                    if self.direccion_actual is not None:
                        cambio_angulo = abs(direccion - self.direccion_actual)
                        if cambio_angulo > np.radians(self.umbral_cambio_direccion):
                            if tiempo_actual - self.tiempo_ultimo_cambio_direccion > 1.0:
                                self.cambios_direccion.append(tiempo_actual)
                                self.tiempo_ultimo_cambio_direccion = tiempo_actual
                    
                    self.direccion_actual = direccion
                
                # Clasificar tipo de movimiento
                if self.velocidad_movimiento > self.umbral_velocidad_correr:
                    self.estado_movimiento = 'corriendo'
                elif self.velocidad_movimiento > self.umbral_velocidad_caminar:
                    self.estado_movimiento = 'caminando'
                else:
                    self.estado_movimiento = 'estático'
                
                # Detectar patrón de caminar de un lado a otro
                if len(self.cambios_direccion) >= 3:
                    tiempo_entre_cambios = np.mean([
                        self.cambios_direccion[i] - self.cambios_direccion[i-1]
                        for i in range(1, len(self.cambios_direccion))
                    ])
                    if tiempo_entre_cambios < 3.0:  # Cambios frecuentes de dirección
                        self.estado_movimiento = 'caminando_lado_a_lado'

    def detectar_postura_base(self, angulos):
        if not angulos:
            return "Desconocido"
            
        angulo_rodilla = angulos.get('rodilla', 180)
        angulo_cadera = angulos.get('cadera', 180)
        angulo_torso = angulos.get('torso', 90)
        
        tiempo_actual = time.time()
        
        # Determinar postura actual
        nueva_postura = None
        
        if self.estado_movimiento in ['caminando', 'corriendo']:
            nueva_postura = self.estado_movimiento.capitalize()
        elif angulo_torso < self.umbrales_postura['torso_acostado']:
            nueva_postura = "Acostado"
        elif angulo_rodilla < self.umbrales_postura['rodilla_flexion']:
            if angulo_cadera < self.umbrales_postura['cadera_flexion']:
                nueva_postura = "Agachado"
            else:
                nueva_postura = "Sentado"
        elif angulo_torso < self.umbrales_postura['torso_inclinado']:
            nueva_postura = "Inclinado"
        elif angulo_torso > self.umbrales_postura['torso_vertical']:
            nueva_postura = "De pie"
        else:
            nueva_postura = "Transición"
        
        # Sistema de confirmación de postura
        if nueva_postura != self.postura_temporal:
            self.postura_temporal = nueva_postura
            self.ultima_postura_tiempo = tiempo_actual
        elif tiempo_actual - self.ultima_postura_tiempo > self.tiempo_confirmacion_postura:
            return nueva_postura
            
        return self.estado_actual
    
    def analizar_transicion(self, nueva_postura):
        tiempo_actual = time.time()
        self.historial_posturas.append((nueva_postura, tiempo_actual))
        
        # Si no hay transición en progreso, verificar si comienza una
        if not self.transicion_en_progreso:
            if nueva_postura != self.estado_actual:
                self.transicion_en_progreso = True
                self.inicio_transicion = tiempo_actual
                # Predecir siguiente postura basado en patrones conocidos
                for (inicio, fin), patron in self.patrones_transicion.items():
                    if inicio == self.estado_actual:
                        self.prediccion_siguiente_postura = fin
                        break
        
        # Si hay una transición en progreso, analizarla
        if self.transicion_en_progreso:
            duracion_transicion = tiempo_actual - self.inicio_transicion
            
            # Verificar si la transición se completó
            if nueva_postura != "Transición" and duracion_transicion > self.umbral_tiempo:
                transicion = f"{self.estado_actual} → {nueva_postura}"
                self.historial_transiciones.append({
                    'desde': self.estado_actual,
                    'hasta': nueva_postura,
                    'duracion': duracion_transicion,
                    'tiempo': tiempo_actual
                })
                self.estado_actual = nueva_postura
                self.transicion_en_progreso = False
                return transicion
            
            # Si la transición está tardando demasiado, cancelarla
            elif duracion_transicion > 3.0:  # Máximo 3 segundos por transición
                self.transicion_en_progreso = False
                self.prediccion_siguiente_postura = None
        
        return None
    
    def predecir_siguiente_movimiento(self):
        if len(self.historial_transiciones) < 2:
            return None
        
        # Analizar patrones recientes
        ultimas_transiciones = list(self.historial_transiciones)[-2:]
        patron_actual = (ultimas_transiciones[-1]['desde'], ultimas_transiciones[-1]['hasta'])
        
        # Buscar patrones comunes
        for (inicio, fin), patron in self.patrones_transicion.items():
            if patron_actual == (inicio, fin):
                return f"Posible retorno a {inicio}"
        
        return None

    def calcular_visibilidad_partes(self, landmarks):
        partes_cuerpo = {
            'cabeza': [mp_pose.PoseLandmark.NOSE, mp_pose.PoseLandmark.LEFT_EYE, mp_pose.PoseLandmark.RIGHT_EYE],
            'torso': [mp_pose.PoseLandmark.LEFT_SHOULDER, mp_pose.PoseLandmark.RIGHT_SHOULDER, 
                     mp_pose.PoseLandmark.LEFT_HIP, mp_pose.PoseLandmark.RIGHT_HIP],
            'brazo_izq': [mp_pose.PoseLandmark.LEFT_SHOULDER, mp_pose.PoseLandmark.LEFT_ELBOW, 
                         mp_pose.PoseLandmark.LEFT_WRIST],
            'brazo_der': [mp_pose.PoseLandmark.RIGHT_SHOULDER, mp_pose.PoseLandmark.RIGHT_ELBOW, 
                         mp_pose.PoseLandmark.RIGHT_WRIST],
            'pierna_izq': [mp_pose.PoseLandmark.LEFT_HIP, mp_pose.PoseLandmark.LEFT_KNEE, 
                          mp_pose.PoseLandmark.LEFT_ANKLE],
            'pierna_der': [mp_pose.PoseLandmark.RIGHT_HIP, mp_pose.PoseLandmark.RIGHT_KNEE, 
                          mp_pose.PoseLandmark.RIGHT_ANKLE]
        }
        
        for parte, puntos in partes_cuerpo.items():
            visibilidad = sum(landmarks[punto.value].visibility for punto in puntos) / len(puntos)
            self.visibilidad_partes[parte] = visibilidad > 0.5
    
    def calcular_distancia_normalizada(self, punto1, punto2, frame_shape):
        x1, y1 = punto1
        x2, y2 = punto2
        distancia = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        return distancia / np.sqrt(frame_shape[0]**2 + frame_shape[1]**2)
    
    def analizar_lenguaje_corporal(self, landmarks, frame_shape):
        if not landmarks:
            return
        
        # Definir centro_torso al inicio del método
        centro_torso = np.array([
            (landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x + 
             landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x) * frame_shape[1] / 2,
            (landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y + 
             landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y) * frame_shape[0] / 2
        ])
        
        # Análisis de postura de brazos
        if self.visibilidad_partes['brazo_izq'] and self.visibilidad_partes['brazo_der']:
            # Detectar brazos cruzados
            mano_izq = np.array([
                landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x * frame_shape[1],
                landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y * frame_shape[0]
            ])
            mano_der = np.array([
                landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x * frame_shape[1],
                landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y * frame_shape[0]
            ])
            
            distancia_manos = self.calcular_distancia_normalizada(mano_izq, mano_der, frame_shape)
            distancia_manos_torso = self.calcular_distancia_normalizada(
                (mano_izq + mano_der) / 2, centro_torso, frame_shape)
            
            # Clasificar postura de brazos
            if distancia_manos < 0.15 and distancia_manos_torso < 0.1:
                self.lenguaje_corporal['brazos'] = 'cruzados'
                self.lenguaje_corporal['apertura'] = 'cerrada'
            elif distancia_manos > 0.3:
                self.lenguaje_corporal['brazos'] = 'abiertos'
                self.lenguaje_corporal['apertura'] = 'abierta'
        
        # Análisis de postura corporal
        if self.visibilidad_partes['torso']:
            # Calcular ángulo de la columna
            hombros_centro = np.array([
                (landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x + 
                 landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x) * frame_shape[1] / 2,
                (landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y + 
                 landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y) * frame_shape[0] / 2
            ])
            
            angulo_columna = calcular_angulo(
                [landmarks[mp_pose.PoseLandmark.NOSE.value].x * frame_shape[1],
                 landmarks[mp_pose.PoseLandmark.NOSE.value].y * frame_shape[0]],
                hombros_centro,
                centro_torso
            )
            
            # Clasificar postura corporal
            if angulo_columna < 60:
                self.lenguaje_corporal['postura_general'] = 'encorvada'
                self.lenguaje_corporal['confianza'] = 'baja'
            elif angulo_columna > 80:
                self.lenguaje_corporal['postura_general'] = 'erguida'
                self.lenguaje_corporal['confianza'] = 'alta'
            
            # Analizar tensión corporal
            velocidad_movimiento = 0
            if len(self.historial_posturas) > 1:
                ultimo_tiempo = self.historial_posturas[-1][1]
                penultimo_tiempo = self.historial_posturas[-2][1]
                if ultimo_tiempo - penultimo_tiempo > 0:
                    velocidad_movimiento = 1 / (ultimo_tiempo - penultimo_tiempo)
            
            if velocidad_movimiento > 2:  # Más de 2 cambios por segundo
                self.lenguaje_corporal['tension'] = 'alta'
            elif velocidad_movimiento > 1:
                self.lenguaje_corporal['tension'] = 'media'
            else:
                self.lenguaje_corporal['tension'] = 'baja'

    def analizar_postura_detallada(self, landmarks, frame_shape):
        self.calcular_visibilidad_partes(landmarks)
        self.analizar_movimiento_dinamico(landmarks, frame_shape)
        angulos = {}
        
        # Análisis de brazos
        if self.visibilidad_partes['brazo_izq'] and self.visibilidad_partes['brazo_der']:
            angulo_codo_izq = calcular_angulo(
                [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x * frame_shape[1],
                 landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y * frame_shape[0]],
                [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x * frame_shape[1],
                 landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y * frame_shape[0]],
                [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x * frame_shape[1],
                 landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y * frame_shape[0]]
            )
            
            angulo_codo_der = calcular_angulo(
                [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x * frame_shape[1],
                 landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y * frame_shape[0]],
                [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x * frame_shape[1],
                 landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y * frame_shape[0]],
                [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x * frame_shape[1],
                 landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y * frame_shape[0]]
            )
            
            # Analizar posición de brazos
            if angulo_codo_izq < 90 and angulo_codo_der < 90:
                self.estado_extremidades['brazos'] = 'flexionados'
            elif angulo_codo_izq > 160 and angulo_codo_der > 160:
                self.estado_extremidades['brazos'] = 'extendidos'
            else:
                self.estado_extremidades['brazos'] = 'neutral'
            angulos['codo_izq'] = angulo_codo_izq
            angulos['codo_der'] = angulo_codo_der
        
        # Análisis de piernas
        if self.visibilidad_partes['pierna_izq'] and self.visibilidad_partes['pierna_der']:
            angulo_rodilla_izq = calcular_angulo(
                [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x * frame_shape[1],
                 landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y * frame_shape[0]],
                [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x * frame_shape[1],
                 landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y * frame_shape[0]],
                [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x * frame_shape[1],
                 landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y * frame_shape[0]]
            )
            
            angulo_rodilla_der = calcular_angulo(
                [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x * frame_shape[1],
                 landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y * frame_shape[0]],
                [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x * frame_shape[1],
                 landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y * frame_shape[0]],
                [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x * frame_shape[1],
                 landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y * frame_shape[0]]
            )
            
            # Analizar posición de piernas
            if angulo_rodilla_izq < 120 and angulo_rodilla_der < 120:
                self.estado_extremidades['piernas'] = 'flexionadas'
            elif angulo_rodilla_izq > 160 and angulo_rodilla_der > 160:
                self.estado_extremidades['piernas'] = 'extendidas'
            else:
                self.estado_extremidades['piernas'] = 'neutral'
            angulos['rodilla'] = (angulo_rodilla_izq + angulo_rodilla_der) / 2
        
        # Análisis de torso
        if self.visibilidad_partes['torso']:
            angulo_torso = calcular_angulo(
                [landmarks[mp_pose.PoseLandmark.NOSE.value].x * frame_shape[1],
                 landmarks[mp_pose.PoseLandmark.NOSE.value].y * frame_shape[0]],
                [(landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x + 
                  landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x) * frame_shape[1] / 2,
                 (landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y + 
                  landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y) * frame_shape[0] / 2],
                [(landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x + 
                  landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x) * frame_shape[1] / 2,
                 (landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y + 
                  landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y) * frame_shape[0] / 2]
            )
            
            # Analizar inclinación del torso
            if angulo_torso < 60:
                self.estado_extremidades['torso'] = 'inclinado_adelante'
            elif angulo_torso > 120:
                self.estado_extremidades['torso'] = 'inclinado_atras'
            else:
                self.estado_extremidades['torso'] = 'erguido'
            angulos['torso'] = angulo_torso
        
        # Detectar postura base y analizar transición
        postura_base = self.detectar_postura_base(angulos)
        transicion = self.analizar_transicion(postura_base)
        
        # Actualizar estados
        if transicion:
            prediccion = self.predecir_siguiente_movimiento()
            if prediccion:
                self.prediccion_siguiente_postura = prediccion
        
        # Añadir análisis de lenguaje corporal
        self.analizar_lenguaje_corporal(landmarks, frame_shape)

    def obtener_estado_detallado(self):
        estado = {
            'postura_general': self.estado_actual,
            'movimiento': self.estado_movimiento,
            'velocidad': f"{self.velocidad_movimiento:.2f} m/s" if hasattr(self, 'velocidad_movimiento') else "0 m/s",
            'partes_visibles': [parte for parte, visible in self.visibilidad_partes.items() if visible],
            'estado_extremidades': self.estado_extremidades,
            'en_transicion': self.transicion_en_progreso,
            'lenguaje_corporal': self.lenguaje_corporal
        }
        
        if self.transicion_en_progreso:
            estado['duracion_transicion'] = time.time() - self.inicio_transicion
            estado['prediccion_siguiente'] = self.prediccion_siguiente_postura
        
        if self.historial_transiciones:
            ultima_transicion = self.historial_transiciones[-1]
            estado['ultima_transicion'] = f"{ultima_transicion['desde']} → {ultima_transicion['hasta']}"
            estado['duracion_ultima'] = f"{ultima_transicion['duracion']:.1f}s"
        
        return estado

# Inicializar detector de movimientos
detector_movimientos = DetectorMovimientos()

# Clase para rastrear personas y asignar IDs únicas
class RastreadorPersonas:
    def __init__(self):
        self.siguiente_id = 1
        self.personas_previas = []
        self.tiempo_max_perdida = 0.5  # Reducido para eliminar más rápido personas no visibles
        self.historial_trayectoria = {}
        self.max_puntos_trayectoria = 30
        self.contador_actual = 0  # Nuevo contador para personas actualmente visibles
        
    def obtener_centro_bbox(self, bbox):
        x1, y1, x2, y2 = bbox
        return ((x1 + x2) // 2, (y1 + y2) // 2)
    
    def calcular_distancia(self, centro1, centro2):
        return np.sqrt((centro1[0] - centro2[0])**2 + (centro1[1] - centro2[1])**2)
    
    def actualizar(self, detecciones_actuales):
        tiempo_actual = time.time()
        personas_actualizadas = []
        detecciones_asignadas = set()
        
        # Actualizar contador actual
        self.contador_actual = len(detecciones_actuales)
        
        # Limpiar personas no visibles
        self.personas_previas = [p for p in self.personas_previas 
                               if (tiempo_actual - p['tiempo_ultima_deteccion']) < self.tiempo_max_perdida]
        
        # Actualizar trayectorias
        for id_persona in list(self.historial_trayectoria.keys()):
            if not any(p['id'] == id_persona for p in self.personas_previas):
                if tiempo_actual - self.historial_trayectoria[id_persona][-1]['tiempo'] > self.tiempo_max_perdida:
                    del self.historial_trayectoria[id_persona]
        
        # Para cada persona previa, buscar la detección más cercana
        for persona_previa in self.personas_previas:
            centro_previo = self.obtener_centro_bbox(persona_previa['bbox'])
            mejor_distancia = float('inf')
            mejor_deteccion = None
            mejor_idx = None
            
            for idx, deteccion in enumerate(detecciones_actuales):
                if idx in detecciones_asignadas:
                    continue
                    
                centro_actual = self.obtener_centro_bbox(deteccion['bbox'])
                distancia = self.calcular_distancia(centro_previo, centro_actual)
                
                if distancia < mejor_distancia and distancia < 100:  # Umbral de distancia máxima
                    mejor_distancia = distancia
                    mejor_deteccion = deteccion
                    mejor_idx = idx
            
            if mejor_deteccion is not None:
                detecciones_asignadas.add(mejor_idx)
                persona_actualizada = {
                    'id': persona_previa['id'],
                    'bbox': mejor_deteccion['bbox'],
                    'conf': mejor_deteccion['conf'],
                    'tiempo_ultima_deteccion': tiempo_actual,
                    'centro': self.obtener_centro_bbox(mejor_deteccion['bbox'])
                }
                personas_actualizadas.append(persona_actualizada)
                
                # Actualizar trayectoria
                if persona_previa['id'] not in self.historial_trayectoria:
                    self.historial_trayectoria[persona_previa['id']] = []
                
                self.historial_trayectoria[persona_previa['id']].append({
                    'centro': persona_actualizada['centro'],
                    'tiempo': tiempo_actual
                })
                
                # Mantener solo los últimos N puntos
                if len(self.historial_trayectoria[persona_previa['id']]) > self.max_puntos_trayectoria:
                    self.historial_trayectoria[persona_previa['id']] = (
                        self.historial_trayectoria[persona_previa['id']][-self.max_puntos_trayectoria:]
                    )
        
        # Asignar nuevas IDs a detecciones no asignadas
        for idx, deteccion in enumerate(detecciones_actuales):
            if idx not in detecciones_asignadas:
                nueva_persona = {
                    'id': self.siguiente_id,
                    'bbox': deteccion['bbox'],
                    'conf': deteccion['conf'],
                    'tiempo_ultima_deteccion': tiempo_actual,
                    'centro': self.obtener_centro_bbox(deteccion['bbox'])
                }
                personas_actualizadas.append(nueva_persona)
                
                # Inicializar trayectoria para nueva persona
                self.historial_trayectoria[self.siguiente_id] = [{
                    'centro': nueva_persona['centro'],
                    'tiempo': tiempo_actual
                }]
                
                self.siguiente_id += 1
        
        self.personas_previas = personas_actualizadas
        return personas_actualizadas

# Inicializar rastreador de personas
rastreador_personas = RastreadorPersonas()

# Clase para gestionar la detección y seguimiento de objetos
class GestorObjetos:
    def __init__(self):
        self.historial_objetos = {}  # Diccionario para seguimiento de objetos
        self.tiempo_max_perdida = 1.0  # Tiempo máximo para mantener objeto sin detección
        self.ultimo_tiempo = time.time()
    
    def actualizar(self, objetos_detectados):
        tiempo_actual = time.time()
        objetos_actualizados = {}
        
        # Actualizar objetos existentes y añadir nuevos
        for obj in objetos_detectados:
            nombre = obj['nombre']
            if nombre not in self.historial_objetos:
                self.historial_objetos[nombre] = {
                    'primera_deteccion': tiempo_actual,
                    'ultima_deteccion': tiempo_actual,
                    'conteo': 1,
                    'tiempo_visible': 0,
                    'confianza_promedio': obj['conf']
                }
            else:
                hist = self.historial_objetos[nombre]
                hist['ultima_deteccion'] = tiempo_actual
                hist['conteo'] += 1
                hist['tiempo_visible'] += tiempo_actual - self.ultimo_tiempo
                hist['confianza_promedio'] = (hist['confianza_promedio'] + obj['conf']) / 2
        
        # Limpiar objetos no detectados por mucho tiempo
        self.historial_objetos = {
            nombre: datos for nombre, datos in self.historial_objetos.items()
            if tiempo_actual - datos['ultima_deteccion'] < self.tiempo_max_perdida
        }
        
        self.ultimo_tiempo = tiempo_actual
        return self.historial_objetos

# Inicializar gestor de objetos
gestor_objetos = GestorObjetos()

# Función para procesar todas las detecciones de YOLO11
def procesar_detecciones_yolo11(resultados, frame):
    detecciones = {
        'personas': [],
        'objetos': []
    }
    
    for r in resultados:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            clase = int(box.cls[0].item())
            confianza = box.conf[0].item()
            
            # Validar que la detección esté dentro del frame y tenga un tamaño mínimo
            if (x1 >= 0 and y1 >= 0 and x2 <= frame.shape[1] and y2 <= frame.shape[0] and 
                x2 - x1 > 20 and y2 - y1 > 20 and confianza > min_confianza_deteccion):
                
                deteccion = {
                    'bbox': (x1, y1, x2, y2),
                    'conf': confianza,
                    'clase': clase
                }
                
                # Clase 0 es persona en COCO
                if clase == 0:
                    detecciones['personas'].append(deteccion)
                else:
                    # Obtener el nombre de la clase en español
                    nombre_clase = list(TRADUCCION_CLASES.values())[clase] if clase < len(TRADUCCION_CLASES) else f"clase_{clase}"
                    deteccion['nombre'] = nombre_clase
                    detecciones['objetos'].append(deteccion)
    
    return detecciones

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Crear un panel de información más alto para incluir información de objetos
    panel_altura = 300  # Aumentado para mostrar más información
    panel = np.zeros((panel_altura, frame.shape[1], 3), dtype=np.uint8)
    
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    resultados_pose = pose.process(frame_rgb)
    resultados = modelo(frame, conf=min_confianza_deteccion)
    
    # Procesar todas las detecciones
    detecciones = procesar_detecciones_yolo11(resultados, frame)
    personas_rastreadas = rastreador_personas.actualizar(detecciones['personas'])
    historial_objetos = gestor_objetos.actualizar(detecciones['objetos'])
    
    # Actualizar el panel con información
    y_offset = 20
    cv2.putText(panel, f"Personas en escena: {rastreador_personas.contador_actual}", 
               (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    # Mostrar información de movimiento para cada persona
    for persona in personas_rastreadas:
        if hasattr(detector_movimientos, 'estado_movimiento'):
            y_offset += 25
            estado = detector_movimientos.obtener_estado_detallado()
            cv2.putText(panel, f"Persona {persona['id']} - {estado['postura_general']}", 
                      (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            y_offset += 20
            cv2.putText(panel, f"Movimiento: {estado['movimiento'].upper()}", 
                      (30, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            y_offset += 20
            cv2.putText(panel, f"Velocidad: {estado['velocidad']}", 
                      (30, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    
    # Mostrar objetos detectados en el panel con información detallada
    if historial_objetos:
        y_offset += 25
        cv2.putText(panel, "Objetos en escena:", 
                   (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        for nombre, datos in historial_objetos.items():
            y_offset += 20
            tiempo_visible = round(datos['tiempo_visible'], 1)
            confianza = int(datos['confianza_promedio'] * 100)
            
            texto = f"- {nombre}: {datos['conteo']} "
            texto += f"(Visible: {tiempo_visible}s, Conf: {confianza}%)"
            
            cv2.putText(panel, texto, 
                      (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    
    # Dibujar todas las detecciones en el frame
    # Primero personas con sus trayectorias e IDs
    for persona in personas_rastreadas:
        x1, y1, x2, y2 = persona['bbox']
        confianza = persona['conf']
        id_persona = persona['id']
        
        # Verde para personas
        color = (0, 255, 0)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        
        # Dibujar ID y confianza con mejor visibilidad
        texto = f'ID: {id_persona} ({int(confianza * 100)}%)'
        (tw, th), _ = cv2.getTextSize(texto, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        cv2.rectangle(frame, (x1, y1-25), (x1 + tw, y1), color, -1)
        cv2.putText(frame, texto, (x1, y1-5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        
        # Mostrar postura y estado de movimiento
        if hasattr(detector_movimientos, 'estado_movimiento'):
            estado = detector_movimientos.obtener_estado_detallado()
            texto_postura = f"Postura: {estado['postura_general']}"
            texto_estado = f"Estado: {estado['movimiento'].upper()}"
            
            cv2.putText(frame, texto_postura, (x1, y2 + 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            cv2.putText(frame, texto_estado, (x1, y2 + 40),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    
    # Luego objetos con información mejorada
    for objeto in detecciones['objetos']:
        x1, y1, x2, y2 = objeto['bbox']
        confianza = objeto['conf']
        nombre = objeto['nombre']
        
        # Color específico para cada tipo de objeto
        color_idx = hash(nombre) % len(COLORES)
        color = tuple(map(int, COLORES[color_idx]))
        
        # Dibujar bounding box con estilo mejorado
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        
        # Añadir texto con más información
        datos_objeto = historial_objetos.get(nombre, {})
        tiempo_visible = round(datos_objeto.get('tiempo_visible', 0), 1)
        texto = f'{nombre} ({int(confianza * 100)}%) - {tiempo_visible}s'
        
        (tw, th), _ = cv2.getTextSize(texto, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        cv2.rectangle(frame, (x1, y1-20), (x1 + tw, y1), color, -1)
        cv2.putText(frame, texto, (x1, y1-5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)

    # Combinar frame con panel de información
    frame_completo = np.vstack([frame, panel])
    
    # Mostrar el frame
    cv2.imshow('Detector de Movimientos y Posturas', frame_completo)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar recursos
cap.release()
cv2.destroyAllWindows() 