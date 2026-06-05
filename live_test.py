# RINGNODE_AI/live_test.py

"""
RingNode AI - Showcase Engine (Çoklu Oyuncu Modu)
Danışman Hocanın isteği üzerine: 2 kişiyi aynı anda tanır, 
farklı renklerde iskelet çizer ve X-Ekseni ile kişileri ayırır.
"""

import cv2
import numpy as np
import torch
import torch.nn as nn
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from collections import deque
import warnings
import joblib
import time

# config.py'den ayarlarımızı çekiyoruz
from config import (WINDOW_SIZE as PENCERE_BOYUTU, INPUT_SIZE, NUM_CLASSES, SINIFLAR, 
                    MODEL_PATH, SCALER_PATH, MEDIAPIPE_MODEL_PATH, IDLE_VARIANCE_THRESHOLD)
from utils.angle_math import extract_upper_body_angles_with_velocity

warnings.filterwarnings("ignore", category=UserWarning)

CONFIDENCE_THRESHOLD = 0.65
FPS_SMOOTHING = 10
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ==========================================
# 🧠 BoxingMLP Mimarisi
# ==========================================
class BoxingMLP(nn.Module):
    def __init__(self, input_size, num_classes):
        super(BoxingMLP, self).__init__()
        self.layer1 = nn.Linear(input_size, 128)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        self.layer2 = nn.Linear(128, 64)
        self.out = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.relu(self.layer1(x))
        x = self.dropout(x)
        x = self.relu(self.layer2(x))
        x = self.out(x)
        return x

# ==========================================
# 🛠️ Yardımcı Fonksiyonlar
# ==========================================
def is_idle(kuyruk: deque) -> bool:
    if len(kuyruk) < PENCERE_BOYUTU: return False
    angles_array = np.array(kuyruk)[:, :10]
    stds = np.std(angles_array, axis=0)
    return np.max(stds) < IDLE_VARIANCE_THRESHOLD

def predict(model, scaler, kuyruk, device) -> tuple[str, float]:
    if is_idle(kuyruk):
        return "Idle", 1.0
    
    pencere = np.array(kuyruk).flatten()
    scaled = scaler.transform([pencere])
    input_tensor = torch.FloatTensor(scaled).to(device)
    
    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = torch.softmax(outputs, dim=1)[0]
        confidence, predicted_idx = torch.max(probabilities, 0)
        
    return SINIFLAR[predicted_idx.item()], confidence.item()

def draw_upper_body_skeleton(frame, landmarks, w, h, renk):
    """Belirtilen renkte üst gövde iskeletini çizer."""
    UST_GOVDE_BAGLANTILARI = [
        (11, 12), (11, 13), (13, 15), (12, 14), (14, 16), (11, 23), (12, 24), (23, 24)
    ]
    
    pts = {i: (int(landmarks[i].x * w), int(landmarks[i].y * h)) for i in range(33)}
    
    # Kemikleri Çiz
    for p1, p2 in UST_GOVDE_BAGLANTILARI:
        if p1 in pts and p2 in pts:
            cv2.line(frame, pts[p1], pts[p2], renk, 3)
            
    # Eklemleri Çiz (İçi dolu beyaz, dışı renkli)
    for p in [11, 12, 13, 14, 15, 16, 23, 24]:
        if p in pts:
            cv2.circle(frame, pts[p], 6, (255, 255, 255), -1)
            cv2.circle(frame, pts[p], 8, renk, 2)

def draw_player_hud(frame, player_name, data, x_offset, y_offset):
    """Her oyuncu için ayrı bir bilgi ekranı (HUD) çizer."""
    sinif_adi = data["sinif"]
    guven = data["guven"]
    renk = data["renk"]

    if guven < CONFIDENCE_THRESHOLD and sinif_adi not in ["Idle", "Kalibre ediliyor...", "Insan Yok"]:
        sinif_adi = "Hareket: ?"
        yazi_renk = (0, 0, 255)
    elif sinif_adi == "Idle":
        yazi_renk = (150, 150, 150)
    else:
        yazi_renk = renk

    # Arka plan kutusu
    cv2.rectangle(frame, (x_offset, y_offset), (x_offset + 320, y_offset + 90), (0, 0, 0), -1)
    # Oyuncu İsmi ve Renk İndikatörü
    cv2.putText(frame, player_name, (x_offset + 10, y_offset + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, renk, 2)
    # Tahmin
    cv2.putText(frame, sinif_adi, (x_offset + 10, y_offset + 60), cv2.FONT_HERSHEY_SIMPLEX, 0.9, yazi_renk, 2)
    # Güven Skoru
    if sinif_adi not in ["Insan Yok", "Kalibre ediliyor...", "Idle"]:
        cv2.putText(frame, f"Guven: %{guven*100:.0f}", (x_offset + 10, y_offset + 85), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)

# ==========================================
# 🎮 ANA DÖNGÜ (Dual-Player Motor)
# ==========================================
def main():
    print("🥊 RingNode AI v2.0 - SHOWCASE ENGINE BAŞLATILIYOR...")
    
    scaler = joblib.load(SCALER_PATH)
    model = BoxingMLP(INPUT_SIZE, NUM_CLASSES).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()

    # DİKKAT: num_poses=2 yapıldı!
    base_options = python.BaseOptions(model_asset_path=MEDIAPIPE_MODEL_PATH)
    options = vision.PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.IMAGE,
        num_poses=2, 
        min_pose_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

    # İki oyuncunun beyinlerini (hafızalarını) ayırıyoruz
    state = {
        "Sol": {"kuyruk": deque(maxlen=PENCERE_BOYUTU), "prev_angles": None, "sinif": "Insan Yok", "guven": 0.0, "renk": (50, 255, 50)},  # Neon Yeşil
        "Sag": {"kuyruk": deque(maxlen=PENCERE_BOYUTU), "prev_angles": None, "sinif": "Insan Yok", "guven": 0.0, "renk": (255, 100, 50)} # Neon Mavi/Turuncu
    }

    fps_queue = deque(maxlen=FPS_SMOOTHING)
    cap = cv2.VideoCapture(0)
    
    # Kamerayı geniş ekran aç (Showcase için daha iyi)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    onceki_zaman = time.time()

    with vision.PoseLandmarker.create_from_options(options) as landmarker:
        while cap.isOpened():
            success, frame = cap.read()
            if not success: break

            frame = cv2.flip(frame, 1)
            h, w, _ = frame.shape

            simdiki_zaman = time.time()
            dt = simdiki_zaman - onceki_zaman + 1e-9
            fps_queue.append(1.0 / dt)
            onceki_zaman = simdiki_zaman
            fps = np.mean(fps_queue)

            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
            result = landmarker.detect(mp_image)

            # Bu frame için güncelleme bayraklarını sıfırla
            for k in state: state[k]["guncellendi"] = False

            if result.pose_landmarks:
                for i in range(len(result.pose_landmarks)):
                    l2d = result.pose_landmarks[i]
                    l3d = result.pose_world_landmarks[i]
                    
                    # Omuzların orta noktasının X eksenindeki konumunu bul (0.0 Sol, 1.0 Sağ)
                    center_x = (l2d[11].x + l2d[12].x) / 2.0
                    oyuncu = "Sol" if center_x < 0.5 else "Sag"

                    # Eğer iki kişi de yanlışlıkla aynı yarıya geçerse çakışmayı önle
                    if state[oyuncu]["guncellendi"]: continue 

                    # 1. Biyomekanik hesaplama ve hafızaya ekleme
                    features = extract_upper_body_angles_with_velocity(l3d, state[oyuncu]["prev_angles"], dt)
                    state[oyuncu]["prev_angles"] = features[:10]
                    state[oyuncu]["kuyruk"].append(features)
                    state[oyuncu]["guncellendi"] = True

                    # 2. İskeleti kendi renginde çiz
                    draw_upper_body_skeleton(frame, l2d, w, h, state[oyuncu]["renk"])

            # 3. Oyuncuların durumlarını değerlendir ve Tahmin yap
            for oyuncu, data in state.items():
                if not data["guncellendi"]:
                    # Oyuncu ekranda yoksa hafızasını sıfırla
                    data["kuyruk"].clear()
                    data["prev_angles"] = None
                    data["sinif"] = "Insan Yok"
                    data["guven"] = 0.0
                else:
                    if len(data["kuyruk"]) == PENCERE_BOYUTU:
                        data["sinif"], data["guven"] = predict(model, scaler, data["kuyruk"], device)
                    else:
                        data["sinif"] = "Kalibre ediliyor..."
                        data["guven"] = 0.0

            # 4. Arayüzü (HUD) Çiz
            # Sol Oyuncu sol üstte, Sağ Oyuncu sağ üstte
            draw_player_hud(frame, "P1 (SOL BOKSOR)", state["Sol"], x_offset=10, y_offset=10)
            draw_player_hud(frame, "P2 (SAG BOKSOR)", state["Sag"], x_offset=w - 330, y_offset=10)
            
            # FPS Sayacı (Alt Orta)
            cv2.putText(frame, f"FPS: {fps:.0f}", (w//2 - 50, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

            cv2.imshow('RingNode AI - VERSUS MODE', frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()