# RINGNODE_AI/video_data_extractor.py

"""
RingNode AI - Çevrimdışı Video Veri Çıkarıcı
Önceden çekilmiş MP4 videolarını işler, 3D biyomekanik açıları çıkarır
ve belirtilen sınıfa (etikete) göre CSV'ye otomatik kaydeder.
"""

import cv2
import numpy as np
import pandas as pd
import os
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# Kendi biyomekanik motorumuz
from utils.angle_math import extract_upper_body_angles_with_velocity
from utils.landmark_ids import ANGLE_NAMES

# ==========================================
# ⚙️ KULLANICI AYARLARI (BURAYI DEĞİŞTİR)
# ==========================================
VIDEO_YOLU = "./video-data/savunma_full.mp4"  # İşlenecek videonun adı/yolu
HEDEF_SINIF = 3  # 0:Aparkat, 1:Direkt, 2:Krose, 3:Savunma

CSV_DOSYA_ADI = "boks_ustgovde_aci_veri.csv"

CLASS_MAP = {0: "Aparkat", 1: "Direkt", 2: "Krose", 3: "Savunma"}


def main():
    print(f"--- VIDEO İŞLEME BAŞLIYOR: {CLASS_MAP[HEDEF_SINIF]} ---")

    if not os.path.exists(VIDEO_YOLU):
        print(
            f"❌ HATA: '{VIDEO_YOLU}' bulunamadı! Videonun klasörde olduğundan emin ol."
        )
        return

    # MediaPipe Ayarları (Video Modu)
    base_options = python.BaseOptions(model_asset_path="pose_landmarker.task")
    options = vision.PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO,  # DİKKAT: Artık VIDEO modundayız
        num_poses=1,
        min_pose_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    cap = cv2.VideoCapture(VIDEO_YOLU)

    # Videonun orijinal FPS'ini alarak kusursuz 'dt' (Delta Time) hesabı yapıyoruz
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0 or np.isnan(fps):
        fps = 30.0  # Güvenlik önlemi
    dt = 1.0 / fps
    print(f"🎥 Video FPS: {fps:.2f} | Kusursuz Zaman Farkı (dt): {dt:.4f} sn")

    veri_listesi = []
    prev_angles = None
    frame_sayaci = 0
    kaydedilen_frame = 0

    with vision.PoseLandmarker.create_from_options(options) as landmarker:
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break  # Video bitti

            frame_sayaci += 1

            # MediaPipe VIDEO modunda timestamp (milisaniye) ister
            timestamp_ms = int(cap.get(cv2.CAP_PROP_POS_MSEC))
            if timestamp_ms <= 0:
                timestamp_ms = frame_sayaci * int(dt * 1000)

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

            # Videoyu işle
            result = landmarker.detect_for_video(mp_image, timestamp_ms)

            # Ekranda İskeleti ve Durumu Göster
            cv2.putText(
                frame,
                f"Islem: {CLASS_MAP[HEDEF_SINIF]}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2,
            )
            cv2.putText(
                frame,
                f"Frame: {frame_sayaci}",
                (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 255, 0),
                2,
            )

            if result.pose_world_landmarks and result.pose_landmarks:
                landmarks_3d = result.pose_world_landmarks[0]

                # Biyomekanik hesaplama (Sabit dt ile kusursuz fizik)
                features = extract_upper_body_angles_with_velocity(
                    landmarks_3d, prev_angles, dt
                )
                prev_angles = features[:10]

                # Görsel geri bildirim (iskelet çizimi)
                h, w, _ = frame.shape
                for lm in result.pose_landmarks[0]:
                    x, y = int(lm.x * w), int(lm.y * h)
                    cv2.circle(frame, (x, y), 5, (0, 165, 255), -1)

                # Veriyi sözlük (dict) formatına çevir ve listeye ekle
                row = {"label": HEDEF_SINIF}

                # Önce 10 tane açıyı kaydet (a0, a1 ... a9)
                for i in range(10):
                    row[f"a{i}"] = features[i]

                # Sonra 10 tane hızı kaydet (v0, v1 ... v9)
                for i in range(10):
                    row[f"v{i}"] = features[i + 10]

                veri_listesi.append(row)
                kaydedilen_frame += 1

            # Sadece izlemek için pencereyi aç (İşlemi yavaşlatmasın diye çok kısa beklet)
            # Videonun yan veya düz olması ML için fark etmez ama izlerken düzeltmek istersen cv2.flip kullanabilirsin
            cv2.imshow("Video Analiz", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                print("Kullanıcı tarafından durduruldu.")
                break

    cap.release()
    cv2.destroyAllWindows()

    # --- CSV KAYIT ---
    if len(veri_listesi) > 0:
        df = pd.DataFrame(veri_listesi)
        if os.path.exists(CSV_DOSYA_ADI):
            df.to_csv(CSV_DOSYA_ADI, mode="a", header=False, index=False)
            print(
                f"\n✅ {CSV_DOSYA_ADI} dosyasına {kaydedilen_frame} frame {CLASS_MAP[HEDEF_SINIF]} eklendi."
            )
        else:
            df.to_csv(CSV_DOSYA_ADI, index=False)
            print(
                f"\n✅ YENİ DOSYA: {CSV_DOSYA_ADI} oluşturuldu ve {kaydedilen_frame} frame {CLASS_MAP[HEDEF_SINIF]} kaydedildi."
            )
    else:
        print("\n❌ Videoda hiç insan algılanmadı veya veri çıkarılamadı.")


if __name__ == "__main__":
    main()
