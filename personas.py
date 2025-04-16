import cv2
import torch
from ultralytics import YOLO
import random
import mysql.connector
from datetime import datetime

# Configurar el dispositivo
device = "cuda" if torch.cuda.is_available() else "cpu"
model = YOLO("yolov8s.pt")
model.to(device).half() if device == "cuda" else model.to(device)

# Traducción de clase "person"
clases_esp = {
    "person": "Persona"
}

# Color específico para persona
color_persona = (0, 255, 0)  # Verde

# Configurar la cámara
cap = cv2.VideoCapture(2)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
cap.set(cv2.CAP_PROP_FPS, 60)

window_name = "Detección de Personas"
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

# Variable de control para evitar guardar en cada frame
ultimo_registro = datetime.now()

# Función para guardar en MySQL
def guardar_en_mysql(person_count, imagen_path=""):
    try:
        conn = mysql.connector.connect(
            host="localhost",
            user="leonardo",
            password="76811927",
            database="deteccion_personas"
        )
        cursor = conn.cursor()

        fecha_actual = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        nombre_camara = "Camara_2"

        sql = "INSERT INTO detecciones (fecha_hora, cantidad_personas, nombre_camara, imagen_path) VALUES (%s, %s, %s, %s)"
        valores = (fecha_actual, person_count, nombre_camara, imagen_path)

        cursor.execute(sql, valores)
        conn.commit()
        cursor.close()
        conn.close()
        print("✔ Registro guardado en MySQL")

    except mysql.connector.Error as err:
        print(f"❌ Error al conectar con MySQL: {err}")

# Bucle principal
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame, imgsz=640, conf=0.1, iou=0.3, device=device, visualize=False)
    person_count = 0

    for result in results:
        for box in result.boxes:
            class_id = int(box.cls[0])
            label = model.names[class_id]

            if label != "person":
                continue

            label_esp = clases_esp[label]
            person_count += 1

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            text = f"{label_esp} {conf:.2f}"

            cv2.rectangle(frame, (x1, y1), (x2, y2), color_persona, 3)
            (text_width, text_height), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            cv2.rectangle(frame, (x1, y1 - text_height - 10), (x1 + text_width + 6, y1), color_persona, -1)
            cv2.putText(frame, text, (x1 + 3, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

    # Mostrar contador de personas
    contador_texto = f"Personas: {person_count}"
    cv2.putText(frame, contador_texto, (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color_persona, 2)

    # Guardar en MySQL e imagen si hay personas
    ahora = datetime.now()
    if person_count > 0 and (ahora - ultimo_registro).total_seconds() > 1:
        imagen_path = f"capturas/personas_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
        cv2.imwrite(imagen_path, frame)

        guardar_en_mysql(person_count, imagen_path)
        ultimo_registro = ahora

    # Mostrar ventana
    cv2.imshow(window_name, frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
