"""
Biyomekanik Koordinat ve Açı Sabitleri
Bu dosya, MediaPipe'tan alınacak üst gövde eklem noktalarını ve 
hesaplanacak 10 temel açının tanımlarını içerir.
"""

# MediaPipe BlazePose Üst Gövde Landmark ID Sabitleri
SOL_OMUZ   = 11
SAG_OMUZ   = 12
SOL_DIRSEK = 13
SAG_DIRSEK = 14
SOL_BILEK  = 15
SAG_BILEK  = 16
SOL_KALCA  = 23
SAG_KALCA  = 24

# Hesaplanacak 10 Açının Tanımı
# DİKKAT: Bu isimler data_collection.py içinde CSV sütun başlıkları (a0, a1... a9) olarak kullanılacaktır.
ANGLE_NAMES = [
    "sol_dirsek",            # a0: 11-13-15 (Sol kol uzanma durumu - Direkt/Jab için)
    "sag_dirsek",            # a1: 12-14-16 (Sağ kol uzanma durumu - Direkt/Cross için)
    "sol_omuz_fleksion",     # a2: 23-11-13 (Sol omuz öne kaldırma)
    "sag_omuz_fleksion",     # a3: 24-12-14 (Sağ omuz öne kaldırma)
    "sol_omuz_abduksiyon",   # a4: Sol dirsekin gövde orta hattından yatay uzaklık açısı (Kroşe sallama)
    "sag_omuz_abduksiyon",   # a5: Sağ dirsekin gövde orta hattından yatay uzaklık açısı (Kroşe sallama)
    "govde_rotasyon",        # a6: Omuz hattı (11->12) ile yatay eksen arasındaki açı (Kroşe/Direkt dönüş gücü)
    "sol_onkol_oryantasyon", # a7: Sol önkol (13->15 vektörü) eğimi (Aparkatın yukarı çıkış rotası)
    "sag_onkol_oryantasyon", # a8: Sağ önkol (14->16 vektörü) eğimi (Aparkatın yukarı çıkış rotası)
    "govde_egimi",           # a9: Omuz orta noktası -> Kalça orta noktası eğimi (Savunma / Eskiv için)
]

print("✅ utils/landmark_ids.py başarıyla yüklendi. 10 kinematik açı tanımlandı.")