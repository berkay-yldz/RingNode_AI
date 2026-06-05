# RINGNODE_AI/utils/angle_math.py

"""
Biyomekanik Açı ve Hız Hesaplama Modülü
Bu modül, MediaPipe koordinatlarından üst gövde kinematik özelliklerini
(statik açılar ve açısal hızlar) çıkarır.
"""

import numpy as np
import types

# Kendi yazdığımız landmark_ids dosyasından sabitleri çekiyoruz
from utils.landmark_ids import (
    SOL_OMUZ,
    SAG_OMUZ,
    SOL_DIRSEK,
    SAG_DIRSEK,
    SOL_BILEK,
    SAG_BILEK,
    SOL_KALCA,
    SAG_KALCA,
)


def calculate_angle(a, b, c):
    """
    3 boyutlu uzayda (x, y, z) 3 nokta arasındaki GERÇEK açıyı hesaplar.
    Boksör kameraya yan dönse bile açı asla bozulmaz (Rotation Invariant).
    """
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    ba = a - b
    bc = c - b

    # 3D Uzayda iki vektör arasındaki açı formülü: cos(theta) = (A.B) / (|A|*|B|)
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    # Matematiksel taşmaları (1.000000001 gibi) engellemek için clip kullanıyoruz
    angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))

    return np.degrees(angle)


def calculate_vector_angle(p1, p2, reference_axis="horizontal"):
    """
    İki nokta arasındaki vektörün yatay veya dikey eksenle yaptığı açıyı bulur.
    Dönen değer 0-180 arasındadır.
    """
    p1 = np.array(p1)
    p2 = np.array(p2)

    diff = p2 - p1
    dx, dy = diff[0], diff[1]

    if reference_axis == "horizontal":
        angle = np.arctan2(dy, dx)
    else:  # vertical
        angle = np.arctan2(dx, dy)

    angle = np.abs(np.degrees(angle))
    return angle


def get_midpoint(p1, p2):
    """İki noktanın 3D (x,y,z) tam orta koordinatını verir."""
    p1 = np.array(p1)
    p2 = np.array(p2)
    return (p1 + p2) / 2.0

def extract_upper_body_angles(landmarks):
    """
    MediaPipe WORLD landmark listesinden 3D biyomekanik açıyı (derece) çıkarır.
    """
    # DİKKAT: Artık Z eksenini de alıyoruz!
    pts = {
        i: (landmarks[i].x, landmarks[i].y, landmarks[i].z)
        for i in [
            SOL_OMUZ,
            SAG_OMUZ,
            SOL_DIRSEK,
            SAG_DIRSEK,
            SOL_BILEK,
            SAG_BILEK,
            SOL_KALCA,
            SAG_KALCA,
        ]
    }

    omuz_orta = get_midpoint(pts[SOL_OMUZ], pts[SAG_OMUZ])
    kalca_orta = get_midpoint(pts[SOL_KALCA], pts[SAG_KALCA])

    # Açı hesaplamaları aynı kalır, calculate_angle artık 3D array aldığı için 3D çalışır
    a0 = calculate_angle(pts[SOL_OMUZ], pts[SOL_DIRSEK], pts[SOL_BILEK])
    a1 = calculate_angle(pts[SAG_OMUZ], pts[SAG_DIRSEK], pts[SAG_BILEK])
    a2 = calculate_angle(pts[SOL_KALCA], pts[SOL_OMUZ], pts[SOL_DIRSEK])
    a3 = calculate_angle(pts[SAG_KALCA], pts[SAG_OMUZ], pts[SAG_DIRSEK])

    # Not: A4-A9 (Yatay ve dikey eksenli açılar) için calculate_vector_angle
    # fonksiyonu derinlikte biraz manipüle edilebilir ama omuz kalça oranında
    # 2D izdüşümü kullanmak duruş (egim) için hala kabul edilebilirdir.
    a4 = calculate_vector_angle(omuz_orta, pts[SOL_DIRSEK], "horizontal")
    a5 = calculate_vector_angle(omuz_orta, pts[SAG_DIRSEK], "horizontal")
    a6 = calculate_vector_angle(pts[SOL_OMUZ], pts[SAG_OMUZ], "horizontal")
    a7 = calculate_vector_angle(pts[SOL_DIRSEK], pts[SOL_BILEK], "horizontal")
    a8 = calculate_vector_angle(pts[SAG_DIRSEK], pts[SAG_BILEK], "horizontal")
    a9 = calculate_vector_angle(omuz_orta, kalca_orta, "vertical")

    return np.array([a0, a1, a2, a3, a4, a5, a6, a7, a8, a9])


def calculate_angular_velocity(prev_angles, curr_angles, dt=1.0):
    """
    İki frame arasındaki açısal hızı (ivmeyi) hesaplar.
    Wrap-around (350 dereceden 10 dereceye geçerken oluşan saçmalığı) engeller.
    """
    diff = curr_angles - prev_angles
    # Wrap-around kontrolü: Açı farkı 180'den büyükse, zıt yönden gitmiştir
    diff = np.where(diff > 180, diff - 360, np.where(diff < -180, diff + 360, diff))

    angular_velocity = diff / dt
    return angular_velocity


def extract_upper_body_angles_with_velocity(landmarks, prev_angles=None, dt=1.0):
    """
    Modelin asıl kullanacağı ana özellik çıkarıcıdır.
    Hem 10 statik açıyı hem de 10 açısal hızı birleştirip (20,) boyutunda matris döner.
    """
    curr_angles = extract_upper_body_angles(landmarks)

    if prev_angles is None:
        # İlk frame ise hız yoktur, 0 kabul edilir.
        angular_velocity = np.zeros(10)
    else:
        angular_velocity = calculate_angular_velocity(prev_angles, curr_angles, dt)

    feature_vector = np.concatenate([curr_angles, angular_velocity])
    return feature_vector


# ==========================================
# TEST BLOĞU
# ==========================================
if __name__ == "__main__":
    print("--- Modül Manuel Testi Başlatılıyor ---")
    # Sahte landmark testi: Tüm eklemler farklı rastgele koordinatlarda
    np.random.seed(42)
    fake_lm_1 = [
        types.SimpleNamespace(x=np.random.rand(), y=np.random.rand(), z=0.0)
        for _ in range(33)
    ]
    fake_lm_2 = [
        types.SimpleNamespace(x=np.random.rand(), y=np.random.rand(), z=0.0)
        for _ in range(33)
    ]

    # 1. Statik Açı Testi
    angles = extract_upper_body_angles(fake_lm_1)
    print(f"Açı Sayısı: {len(angles)} (Beklenen: 10)")

    # 2. Hız Testi (Ardışık Frame simülasyonu)
    features = extract_upper_body_angles_with_velocity(
        fake_lm_2, prev_angles=angles, dt=1 / 30
    )
    print(f"Özellik Vektörü Boyutu: {len(features)} (Beklenen: 20)")

    print(f"\nAçılar (ilk 10):\n{np.round(features[:10], 2)}")
    print(f"\nAçısal Hızlar (son 10):\n{np.round(features[10:], 2)}")
    print("\n✅ angle_math.py modülü başarıyla test edildi.")
