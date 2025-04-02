import cv2
import os

# === CONFIGURAZIONE ===
video_path = "C:/Users/Marco/Desktop/dataset/video/video01.mp4"   # Cambia con il tuo path video
output_dir = "C:/Users/Marco/Desktop/frame/video01"           # Cartella di output
frame_rate = 5                          # Frame per secondo (puoi cambiare)

# === CREA CARTELLA OUTPUT ===
os.makedirs(output_dir, exist_ok=True)

# === CARICA VIDEO ===
cap = cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS)
frame_interval = int(fps / frame_rate) if frame_rate else 1

frame_count = 0
saved_count = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    if frame_count % frame_interval == 0:
        filename = os.path.join(output_dir, f"frame_{saved_count:05d}.jpg")
        cv2.imwrite(filename, frame)
        saved_count += 1

    frame_count += 1

cap.release()
print(f"✅ Estrazione completata. {saved_count} frame salvati in '{output_dir}'")