# RINGNODE_AI/showcase_engine.py

"""
RingNode AI - Showcase Engine (Arcade Gamification) v2.3
Hafta 4 Protokolüne uygun olarak: X-Ekseni Sıralaması, Can Barları (HP),
Floating Text Animasyonları ve State-Based Hit Detection (Hasar) içerir.
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
from collections import Counter

# config.py'den ayarlar
from config import (
    WINDOW_SIZE as PENCERE_BOYUTU,
    INPUT_SIZE,
    NUM_CLASSES,
    SINIFLAR,
    MODEL_PATH,
    SCALER_PATH,
    MEDIAPIPE_MODEL_PATH,
    IDLE_VARIANCE_THRESHOLD,
)
from utils.angle_math import extract_upper_body_angles_with_velocity

warnings.filterwarnings("ignore", category=UserWarning)

# --- SHOWCASE ARCADE AYARLARI ---
CONFIDENCE_THRESHOLD = 0.65
FPS_SMOOTHING = 10
HIT_COOLDOWN_SN = 1.0  # Spam koruması (Saniyede 1 kez hasar vurulabilir)
HASAR_MIKTARI = {"Direkt": 10, "Krose": 15, "Aparkat": 20}
SALDIRI_SINIFLARI = ["Direkt", "Krose", "Aparkat"]
SAVUNMA_SINIFLARI = ["Savunma"]
SOL_KALCA = 23  # MediaPipe ID

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ==========================================
# 🧠 BÖLÜM 1: BoxingMLP Mimarisi
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
# 🎮 BÖLÜM 2: Oyun Durumu Sınıfları
# ==========================================
class FighterState:
    def __init__(self, isim, renk, baslangic_hp=100):
        self.isim = isim
        self.renk = renk
        self.hp = baslangic_hp
        self.kuyruk = deque(maxlen=PENCERE_BOYUTU)
        self.pred_history = deque(maxlen=9)
        self.prev_angles = None
        self.mevcut_sinif = "Insan Yok"
        self.guven = 0.0
        self.son_hasar_zamani = 0.0
        self.combo_sayisi = 0
        self.combo_aktif = False
        self.guncellendi = False


class FloatingText:
    def __init__(self, metin, x, y, renk):
        self.metin = metin
        self.x = x
        self.y = y
        self.renk = renk
        self.olusturma_zamani = time.time()
        self.omur = 1.5  # Ekranda 1.5 saniye kalır


floating_texts = []


# ==========================================
# ⚔️ BÖLÜM 3: Hasar ve Oyun Mekanikleri
# ==========================================
def check_hit(saldiran: FighterState, savunan: FighterState, zaman: float, frame, w):
    """Kameraya atılan yumruğun karşı tarafa hasar verip vermediğini kontrol eder."""
    if saldiran.mevcut_sinif not in SALDIRI_SINIFLARI:
        return False
    if savunan.mevcut_sinif in SAVUNMA_SINIFLARI:
        return False  # Bloklandı!
    if (zaman - saldiran.son_hasar_zamani) < HIT_COOLDOWN_SN:
        return False  # Cooldown aktif

    # İsabet! Hasar uygula
    hasar = HASAR_MIKTARI.get(saldiran.mevcut_sinif, 10)
    savunan.hp = max(0, savunan.hp - hasar)
    saldiran.son_hasar_zamani = zaman

    # Combo Güncelle
    saldiran.combo_sayisi += 1
    if saldiran.combo_sayisi >= 2:
        saldiran.combo_aktif = True

    savunan.combo_sayisi = 0
    savunan.combo_aktif = False

    # Hasar yazısını saldıranın tarafında (bilek hizası varsayılan) çıkart
    fx = (w // 4) if saldiran.isim == "P1 (SOL)" else (3 * w // 4)
    floating_texts.append(FloatingText(f"-{hasar} HP!", fx, 300, (0, 0, 255)))
    return True


# ==========================================
# 🎨 BÖLÜM 4: Görsel (UI) Çizimleri
# ==========================================
def draw_health_bars(frame, sp1: FighterState, sp2: FighterState, w: int):
    """Can barlarını sol ve sağ üst köşelere çizer."""
    # SP1 (Sol Boksör) Can Barı
    cv2.rectangle(frame, (20, 20), (320, 50), (0, 0, 0), -1)
    cv2.rectangle(frame, (20, 20), (20 + (sp1.hp * 3), 50), sp1.renk, -1)
    cv2.putText(
        frame,
        f"{sp1.isim} : {sp1.hp} HP",
        (30, 42),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
    )

    # SP2 (Sağ Boksör) Can Barı
    cv2.rectangle(frame, (w - 320, 20), (w - 20, 50), (0, 0, 0), -1)
    cv2.rectangle(
        frame, (w - 320 + ((100 - sp2.hp) * 3), 20), (w - 20, 50), sp2.renk, -1
    )
    cv2.putText(
        frame,
        f"{sp2.isim} : {sp2.hp} HP",
        (w - 310, 42),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
    )


def draw_combo_banner(frame, sporcu: FighterState, w: int, h: int, taraf: str):
    """Combo yazısını patlamayı yapan boksörün hizasında çizer."""
    if not sporcu.combo_aktif or sporcu.combo_sayisi < 2:
        return

    metin = f"{sporcu.combo_sayisi}-HIT COMBO!"
    olcek = 1.5
    kalinlik = 4
    (tw, th), _ = cv2.getTextSize(metin, cv2.FONT_HERSHEY_DUPLEX, olcek, kalinlik)

    cx = (w // 4) - (tw // 2) if taraf == "sol" else (3 * w // 4) - (tw // 2)
    cy = h // 4  # Omuz/Kafa üstü hizası

    cv2.putText(
        frame,
        metin,
        (cx + 3, cy + 3),
        cv2.FONT_HERSHEY_DUPLEX,
        olcek,
        (0, 0, 0),
        kalinlik + 2,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        metin,
        (cx, cy),
        cv2.FONT_HERSHEY_DUPLEX,
        olcek,
        sporcu.renk,
        kalinlik,
        cv2.LINE_AA,
    )


def process_floating_texts(frame, simdiki_zaman):
    """Uçuşan metinleri animasyonlu olarak çizer ve süresi biteni siler."""
    global floating_texts
    aktif_textler = []
    for ft in floating_texts:
        gecen_sure = simdiki_zaman - ft.olusturma_zamani
        if gecen_sure < ft.omur:
            # Yazı yukarı doğru süzülür
            guncel_y = int(ft.y - (gecen_sure * 50))
            cv2.putText(
                frame,
                ft.metin,
                (ft.x, guncel_y),
                cv2.FONT_HERSHEY_DUPLEX,
                1.2,
                ft.renk,
                3,
            )
            aktif_textler.append(ft)
    floating_texts = aktif_textler


def draw_upper_body_skeleton(frame, landmarks, w, h, renk):
    UST_GOVDE_BAGLANTILARI = [
        (11, 12),
        (11, 13),
        (13, 15),
        (12, 14),
        (14, 16),
        (11, 23),
        (12, 24),
        (23, 24),
    ]
    pts = {i: (int(landmarks[i].x * w), int(landmarks[i].y * h)) for i in range(33)}
    for p1, p2 in UST_GOVDE_BAGLANTILARI:
        if p1 in pts and p2 in pts:
            cv2.line(frame, pts[p1], pts[p2], renk, 3)
    for p in [11, 12, 13, 14, 15, 16, 23, 24]:
        if p in pts:
            cv2.circle(frame, pts[p], 6, (255, 255, 255), -1)
            cv2.circle(frame, pts[p], 8, renk, 2)


# ==========================================
# 🚀 BÖLÜM 5: Ana Yapay Zeka ve Oyun Döngüsü
# ==========================================
def is_idle(kuyruk: deque) -> bool:
    if len(kuyruk) < PENCERE_BOYUTU:
        return False
    stds = np.std(np.array(kuyruk)[:, :10], axis=0)
    max_sapma = np.max(stds)
    print(f"Anlık Titreme (Sapma): {max_sapma:.2f}")  # BUNU EKLE
    return max_sapma < IDLE_VARIANCE_THRESHOLD


def predict(model, scaler, kuyruk, device) -> tuple[str, float]:
    if is_idle(kuyruk):
        return "Idle", 1.0
    pencere = np.array(kuyruk).flatten()
    scaled = scaler.transform([pencere])
    with torch.no_grad():
        outputs = model(torch.FloatTensor(scaled).to(device))
        prob = torch.softmax(outputs, dim=1)[0]
        conf, idx = torch.max(prob, 0)
    return SINIFLAR[idx.item()], conf.item()

def smooth_predict(sporcu, model, scaler, device):
    sinif, guven = smooth_predict(sporcu, model, scaler, device)    
    # Idle durumu net bir barikat, oylamaya girmez
    if sinif == "Idle":
        sporcu.pred_history.clear()
        return "Idle", 1.0

    # Tahmini hafızaya at
    sporcu.pred_history.append(sinif)

    # Oylama yap (Son 9 tahminde en çok tekrar eden kim?)
    counter = Counter(sporcu.pred_history)
    kazanan_sinif = counter.most_common(1)[0][0]
    oy_orani = counter[kazanan_sinif] / len(sporcu.pred_history)

    # Eğer 9 jürinin %45'inden fazlası aynı fikirde değilse şüpheli (kararsız geçiş anı)
    if oy_orani < 0.45:
        return "Hareket: ?", 0.0

    # Jüri çoğunluğu sağlandı, güvenle söyleyebiliriz!
    return kazanan_sinif, guven


def main():
    print("🥊 RingNode AI v2.3 - ARCADE GAMIFICATION ENGINE BAŞLATILIYOR...")

    scaler = joblib.load(SCALER_PATH)
    model = BoxingMLP(INPUT_SIZE, NUM_CLASSES).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()

    base_options = python.BaseOptions(model_asset_path=MEDIAPIPE_MODEL_PATH)
    options = vision.PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.IMAGE,
        num_poses=2,
        min_pose_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    # Protokole tam uygun Fighter nesneleri (Yeşil ve Turuncu)
    sp1 = FighterState("P1 (SOL)", (50, 255, 50))
    sp2 = FighterState("P2 (SAG)", (0, 165, 255))

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    fps_queue = deque(maxlen=FPS_SMOOTHING)
    onceki_zaman = time.time()

    with vision.PoseLandmarker.create_from_options(options) as landmarker:
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break
            frame = cv2.flip(frame, 1)
            h, w, _ = frame.shape

            simdiki_zaman = time.time()
            dt = max(simdiki_zaman - onceki_zaman, 0.001)  # Gerçek dt
            fps_queue.append(1.0 / dt)
            onceki_zaman = simdiki_zaman
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = landmarker.detect(
                mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
            )

            sp1.guncellendi, sp2.guncellendi = False, False

            # --- PROTOKOL GÜN 3: X-Ekseni Tabanlı Landmark Kimliklendirme ---
            if result.pose_landmarks:
                sporcular_raw = list(
                    zip(result.pose_landmarks, result.pose_world_landmarks)
                )
                # Kalça (SOL_KALCA=23) X eksenine göre soldan sağa sırala
                sporcular_raw.sort(key=lambda item: item[0][SOL_KALCA].x)

                aktif_sporcular = [sp1, sp2][: len(sporcular_raw)]

                for idx, (l2d, l3d) in enumerate(sporcular_raw):
                    if idx > 1:
                        break  # Sadece 2 kişiyi al
                    sporcu = aktif_sporcular[idx]

                    features = extract_upper_body_angles_with_velocity(
                        l3d, sporcu.prev_angles, dt
                    )
                    sporcu.prev_angles = features[:10]
                    sporcu.kuyruk.append(features)
                    sporcu.guncellendi = True
                    draw_upper_body_skeleton(frame, l2d, w, h, sporcu.renk)

            # Tahminleri Yap ve Hasar Hesapla
            for sporcu in [sp1, sp2]:
                if not sporcu.guncellendi:
                    sporcu.kuyruk.clear()
                    sporcu.prev_angles = None
                    sporcu.mevcut_sinif = "Insan Yok"
                elif len(sporcu.kuyruk) == PENCERE_BOYUTU:
                    sinif, guven = predict(model, scaler, sporcu.kuyruk, device)
                    sporcu.mevcut_sinif = (
                        sinif if guven >= CONFIDENCE_THRESHOLD else "Hareket: ?"
                    )
                    sporcu.guven = guven
                else:
                    sporcu.mevcut_sinif = "Kalibre ediliyor..."

            # Hasar (Hit) Motoru
            if sp1.guncellendi and sp2.guncellendi:
                check_hit(sp1, sp2, simdiki_zaman, frame, w)  # SP1 SP2'ye vurur
                check_hit(sp2, sp1, simdiki_zaman, frame, w)  # SP2 SP1'e vurur

            # Oyun Bitişi
            if sp1.hp <= 0 or sp2.hp <= 0:
                kazanan = sp1.isim if sp2.hp <= 0 else sp2.isim
                cv2.putText(
                    frame,
                    f"K.O! {kazanan} KAZANDI!",
                    (w // 4, h // 2),
                    cv2.FONT_HERSHEY_DUPLEX,
                    2.0,
                    (0, 0, 255),
                    5,
                )

            # UI Çizimleri
            draw_health_bars(frame, sp1, sp2, w)
            draw_combo_banner(frame, sp1, w, h, "sol")
            draw_combo_banner(frame, sp2, w, h, "sag")
            process_floating_texts(frame, simdiki_zaman)

            # Alt Durum Çubuğu
            durum_metni = f"P1: {sp1.mevcut_sinif} | P2: {sp2.mevcut_sinif} | FPS: {int(np.mean(fps_queue))}"
            cv2.putText(
                frame,
                durum_metni,
                (w // 2 - 250, h - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
            )

            cv2.imshow("RingNode AI - ARCADE MODE", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
