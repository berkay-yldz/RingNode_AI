# RINGNODE_AI/idle_threshold.py

"""
Idle (Bekleme) Varyans Eşiği Belirleme Aracı
Bu script, kamera karşısında hareketsiz (veya hafif gardda) beklediğiniz anlardaki 
biyomekanik açıların doğal titremesini (standart sapmasını) ölçer.
Bulunan en yüksek std değeri, config.py içindeki IDLE_VARIANCE_THRESHOLD olarak kullanılacaktır.
"""

import cv2
import time
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# Kendi modülümüzden import
from utils.angle_math import extract_upper_body_angles
from utils.landmark_ids import ANGLE_NAMES

def main():
    # MediaPipe Ayarları (Vücut Takipçisi)
    base_options = python.BaseOptions(model_asset_path='pose_landmarker.task')
    options = vision.PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.IMAGE,
        num_poses=1,
        min_pose_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

    cap = cv2.VideoCapture(0)
    
    # 3 Saniye bekleme, 3 Saniye veri toplama mantığı
    hazirlik_suresi = 3.0
    toplama_suresi = 3.0
    
    toplanan_acilar = []
    durum = "HAZIRLIK"
    baslangic_zamani = time.time()

    print("--- IDLE EŞİK BELİRLEME ---")
    print("Kamera açılıyor. Lütfen kameranın karşısına geçin ve boks gardınızı alıp hafifçe bekleyin (gerçekçi bir hareketsizlik).")

    with vision.PoseLandmarker.create_from_options(options) as landmarker:
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                print("Kamera okunamadı.")
                break
                
            frame = cv2.flip(frame, 1) # Ayna görüntüsü
            h, w, _ = frame.shape
            
            gecen_zaman = time.time() - baslangic_zamani
            
            # Görüntüyü MediaPipe formatına çevir
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            result = landmarker.detect(mp_image)

            # Ekrana durum yazdır
            if durum == "HAZIRLIK":
                kalan = hazirlik_suresi - gecen_zaman
                cv2.putText(frame, f"Hazirlan... {kalan:.1f}s", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 165, 255), 2)
                if kalan <= 0:
                    durum = "TOPLAMA"
                    baslangic_zamani = time.time()
                    
            elif durum == "TOPLAMA":
                kalan = toplama_suresi - gecen_zaman
                cv2.putText(frame, f"SABIT DUR! Toplaniyor... {kalan:.1f}s", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                
                # Veri Toplama Anı
                if result.pose_world_landmarks: # DİKKAT: 3D için world_landmarks
                    landmarks = result.pose_world_landmarks[0]
                    angles = extract_upper_body_angles(landmarks)
                    toplanan_acilar.append(angles)
                
                if kalan <= 0:
                    break # Süre bitti, analiz zamanı

            # Ekranda iskeleti basitçe göster (noktalar halinde)
            if result.pose_landmarks:
                for lm in result.pose_landmarks[0]:
                    x, y = int(lm.x * w), int(lm.y * h)
                    cv2.circle(frame, (x, y), 3, (0, 255, 0), -1)

            cv2.imshow('Idle Threshold Test', frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()

    # --- Analiz ve Raporlama ---
    if not toplanan_acilar:
        print("❌ HATA: Hiç vücut algılanmadı. Lütfen aydınlık bir ortamda kameraya tekrar bakın.")
        return

    # Matrisi NumPy dizisine çevir: shape (N_frame, 10_aci)
    acilar_np = np.array(toplanan_acilar)
    
    # Her bir açının standart sapmasını (ne kadar oynadığını) hesapla
    stds = np.std(acilar_np, axis=0)
    means = np.mean(acilar_np, axis=0)
    
    print("\n" + "="*50)
    print(f"📊 {len(toplanan_acilar)} Frame Üzerinden Analiz Raporu")
    print("="*50)
    
    for i in range(10):
        print(f"{ANGLE_NAMES[i]:<25}: Ortalama = {means[i]:6.1f}° | Std Sapma = {stds[i]:5.2f}°")
        
    print("-" * 50)
    max_std = np.max(stds)
    # Güvenlik payı olarak en yüksek titremenin %20 fazlasını eşik alıyoruz
    onerilen_esik = max_std * 1.2 
    
    print(f"📈 En yüksek titreme (Max Std) : {max_std:.2f}°")
    print(f"🎯 ÖNERİLEN IDLE_VARIANCE_THRESHOLD: {onerilen_esik:.2f}")
    print("="*50)
    print(f"Lütfen config.py dosyanızı oluştururken bu değeri ({onerilen_esik:.2f}) kullanın.")

if __name__ == "__main__":
    main()
    
    
# Test 1: Önerilen Eşik = 10.80 (Max Std: 9.00° - Sol Dirsek)
# Test 2: Önerilen Eşik = 8.85 (Max Std: 7.37° - Sol Dirsek)
# Test 3: Önerilen Eşik = 9.81 (Max Std: 8.18° - Sol Dirsek)

# 👉 IDLE_VARIANCE_THRESHOLD = 11.0
