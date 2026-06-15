# RINGNODE_AI/video_data_extractor.py

"""
RingNode AI - Çevrimdışı Video Veri Çıkarıcı  v2.0  (TAM PARİTE / EMA Zırhlı)

AMAÇ: Eğitim verisini, canlı motorun (showcase_engine.py v3.3) ürettiği özellik
dağılımıyla %100 AYNI üretmek. Böylece model, gördüğü veriyle (train) çalıştığı
veriyi (serve) arasında dağılım kayması (Train-Serve Skew) yaşamaz.

CANLI MOTORLA BİREBİR HİZALANAN ZİNCİR
─────────────────────────────────────────────────────────────────────
  1. Frame Flip      │ cv2.flip(frame, 1) — canlı motor aynalı kare üzerinde
                     │ tespit yapar; veri de aynı şekilde aynalanır.
  2. MediaPipe       │ RunningMode.VIDEO + num_poses=2 + detection/presence/
                     │ tracking confidence = 0.7 (showcase_engine ile birebir).
  3. LandmarkSmoother│ EMA(alpha=0.6) + Landmark Freezing (vis<%60 dondur).
                     │ Sınıf showcase_engine.py'den VERBATIM kopyalandı.
                     │ smooth() kare başına TEK kez çağrılır (state tek ilerler).
  4. Özellik         │ extract_upper_body_angles_with_velocity(SMOOTHED l3d,
                     │ prev_angles, dt) — canlı ile aynı fonksiyon, aynı girdi.
  5. dt Clamp        │ clip(1/fps, 0.015, 0.100) — canlı motor RC-1 ile birebir.
  6. reset_on_lost   │ Tespit kaybolan karede smoother.reset()+prev_angles=None.
─────────────────────────────────────────────────────────────────────

⚠️  ÖNEMLİ — VERİ KARIŞTIRMA UYARISI
  Bu betik EMA'lı YENİ dağılım üretir. Eski (EMA'sız, confidence 0.5) toplanmış
  CSV ile KARIŞTIRMAYIN. "Baştan eğitim" için önce eski CSV'yi silin/yedekleyin;
  yoksa karışık dağılım scaler'ı ve modeli bozar.

────────────────────────────────────────────────────────────────────────────────
EĞİTİM (WINDOWING) SÖZLEŞMESİ — canlı flatten ile %100 parite (Bulgu #3)
────────────────────────────────────────────────────────────────────────────────
Canlı motor (showcase_engine.PredictionPipeline._infer) pencereyi şöyle kurar:

    pencere_mat = np.array(kuyruk)            # (WINDOW_SIZE, NUM_FEATURES) = (15, 20)
    pencere_mat[:, :10] = np.clip(.., 0.0, 180.0)     # OOD koruması (RC-3b)
    pencere_mat[:, 10:] = np.clip(.., -500.0, 500.0)
    pencere = pencere_mat.flatten()           # C-order, FRAME-MAJOR → (300,)
    scaler.transform([pencere])

Bu CSV'den eğitim penceresi kurarken AYNISINI yapın (aksi halde gizli shape/değer
uyuşmazlığı oluşur):

    # 1) Aynı sınıfın ARDIŞIK WINDOW_SIZE satırını al (segment sınırına dikkat: bkz. not)
    # 2) Sütun sırası MUTLAKA [a0..a9, v0..v9] olmalı (bu betik bunu garantiler)
    cols   = [f"a{i}" for i in range(10)] + [f"v{i}" for i in range(10)]
    window = df.loc[idx:idx+WINDOW_SIZE-1, cols].values   # (15, 20)
    window[:, :10] = np.clip(window[:, :10], 0.0, 180.0)  # canlı ile AYNI clip
    window[:, 10:] = np.clip(window[:, 10:], -500.0, 500.0)
    X = window.flatten()                                   # C-order → (300,)  BİREBİR
    # scaler bu 300-vektörler üzerinde fit edilmeli.

NOT (segment sınırı): Canlı motorda pencere asla tespit boşluğunu (gap) aşmaz
(reset_on_lost kuyruğu temizler). Burada da gap karelerinde prev_angles sıfırlanır,
böylece gap'ten SONRAKİ ilk karenin hızları (v0..v9) 0 olur ve doğal bir segment
sınırı işareti oluşur. Temiz (kesintisiz tespitli) klipler kullanmak en güvenlisidir.
────────────────────────────────────────────────────────────────────────────────
"""

import cv2
import numpy as np
import pandas as pd
import os
import types
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# Kendi biyomekanik motorumuz (canlı motorla AYNI fonksiyon)
from utils.angle_math import extract_upper_body_angles_with_velocity
from utils.landmark_ids import (
    SOL_OMUZ, SAG_OMUZ, SOL_DIRSEK, SAG_DIRSEK,
    SOL_BILEK, SAG_BILEK, SOL_KALCA, SAG_KALCA,
)

# Merkezi konfigürasyon (tek doğruluk kaynağı)
from config import (
    DATA_PATH,
    MEDIAPIPE_MODEL_PATH,
    SINIFLAR,
    WINDOW_SIZE,
    NUM_ANGLES,
    NUM_FEATURES,
    INPUT_SIZE,
)

# ==========================================
# ⚙️ KULLANICI AYARLARI (BURAYI DEĞİŞTİR)
# ==========================================
VIDEO_YOLU  = "./video-data/aparkat_full.mp4"  # İşlenecek videonun adı/yolu
HEDEF_SINIF = 0  # 0:Aparkat, 1:Direkt, 2:Krose, 3:Savunma

CIKTI_CSV = DATA_PATH  # config.DATA_PATH — eğitim betiğiyle aynı dosya
CLASS_MAP = SINIFLAR   # {0:'Aparkat', 1:'Direkt', 2:'Krose', 3:'Savunma'}

# ==========================================
# 🔒 PARİTE SABİTLERİ — showcase_engine.py ile BİREBİR
# ==========================================
EMA_ALPHA            = 0.6   # showcase_engine.EMA_ALPHA
VISIBILITY_THRESHOLD = 0.60  # showcase_engine.VISIBILITY_THRESHOLD (freezing eşiği)
TOPLAM_LANDMARK      = 33

# MediaPipe tespit ayarları (showcase_engine.main içindeki options ile birebir)
NUM_POSES            = 2
DETECTION_CONFIDENCE = 0.7
PRESENCE_CONFIDENCE  = 0.7
TRACKING_CONFIDENCE  = 0.7

# dt clamp (showcase_engine RC-1 ile birebir)
DT_MIN, DT_MAX = 0.015, 0.100

# Görselleştirme için üst gövde eklemleri
UST_GOVDE_IDS = [
    SOL_OMUZ, SAG_OMUZ, SOL_DIRSEK, SAG_DIRSEK,
    SOL_BILEK, SAG_BILEK, SOL_KALCA, SAG_KALCA,
]


# ═══════════════════════════════════════════════════════════════════════════════
# LandmarkSmoother  —  showcase_engine.py v3.3'ten VERBATIM kopya
# (DİKKAT: İki dosya senkron tutulmalı. Burada davranış değiştirilirse parite bozulur.)
# ═══════════════════════════════════════════════════════════════════════════════
class LandmarkSmoother:
    """
    Ham MediaPipe landmark'larına Gate 1'den ÖNCE iki katman uygular:

      1. Landmark Freezing (Görünürlük Hafızası):
         Bir eklemin visibility değeri eşiğin altına düşerse, o eklemin SON
         güvenilir pozisyonu DONDURULUR. Zıplayan/güvensiz koordinat motora
         (açılara ve hız tamponlarına) hiç girmez.

      2. EMA (Exponential Moving Average, alpha=EMA_ALPHA):
         Görünür eklemlerde kalan titreşim üssel hareketli ortalamayla
         yumuşatılır:  sm = alpha * current + (1 - alpha) * sm_prev

    2D (normalize ekran) ve 3D (world/metre) landmark'lar için AYRI EMA durumu
    tutar. Freezing kararı her iki temsil için de 2D visibility'den verilir
    (görünürlük eklemin özelliğidir; iki temsilde de aynıdır).

    ÇIKTI: SimpleNamespace listesi (x, y, z, visibility). visibility alanına
    HAM (current) değer yazılır → Gate 1 gerçek görünürlüğü görür, donmuş/sahte
    bir değeri değil.
    """

    def __init__(self, alpha: float = EMA_ALPHA,
                 vis_threshold: float = VISIBILITY_THRESHOLD,
                 n: int = TOPLAM_LANDMARK):
        self.alpha = alpha
        self.vis_threshold = vis_threshold
        self.n = n
        self.sm_2d: list[list[float]] | None = None
        self.sm_3d: list[list[float]] | None = None

    def reset(self) -> None:
        """Özne kaybolunca EMA / freezing hafızasını sıfırlar."""
        self.sm_2d = None
        self.sm_3d = None

    def _step(self, sm, raw):
        """
        raw: [(x, y, z, vis), ...] uzunluğu n.
        sm : önceki yumuşatılmış [[x,y,z], ...] veya None (ilk kare).
        """
        a = self.alpha
        if sm is None:
            # İlk kare: elimizdeki en iyi veri → ham pozisyonla başlat.
            return [[r[0], r[1], r[2]] for r in raw]

        for i in range(self.n):
            if raw[i][3] >= self.vis_threshold:
                # Görünür → EMA güncellemesi
                sm[i][0] = a * raw[i][0] + (1.0 - a) * sm[i][0]
                sm[i][1] = a * raw[i][1] + (1.0 - a) * sm[i][1]
                sm[i][2] = a * raw[i][2] + (1.0 - a) * sm[i][2]
            # else: FREEZE → son pozisyonu koru (güncelleme yok)
        return sm

    def smooth(self, l2d_raw, l3d_raw):
        """Ham 2D ve 3D landmark listelerini alır, yumuşatılmış kopyalarını döner."""
        raw2d = [(lm.x, lm.y, lm.z, lm.visibility) for lm in l2d_raw]
        # 3D world landmark'larda freezing kararı 2D visibility'den verilir.
        raw3d = [(lm.x, lm.y, lm.z, l2d_raw[i].visibility)
                 for i, lm in enumerate(l3d_raw)]

        self.sm_2d = self._step(self.sm_2d, raw2d)
        self.sm_3d = self._step(self.sm_3d, raw3d)

        out2d = [
            types.SimpleNamespace(
                x=self.sm_2d[i][0], y=self.sm_2d[i][1],
                z=self.sm_2d[i][2], visibility=raw2d[i][3],  # HAM visibility
            )
            for i in range(self.n)
        ]
        out3d = [
            types.SimpleNamespace(
                x=self.sm_3d[i][0], y=self.sm_3d[i][1],
                z=self.sm_3d[i][2], visibility=raw3d[i][3],
            )
            for i in range(self.n)
        ]
        return out2d, out3d


# ═══════════════════════════════════════════════════════════════════════════════
# ANA İŞLEM
# ═══════════════════════════════════════════════════════════════════════════════
def main():
    # Parite/şekil tutarlılığı için erken doğrulama (gizli shape riskini yok et)
    assert NUM_FEATURES == NUM_ANGLES * 2, "config: NUM_FEATURES, NUM_ANGLES*2 olmalı"
    assert INPUT_SIZE == NUM_FEATURES * WINDOW_SIZE, "config: INPUT_SIZE tutarsız"

    print(f"--- VIDEO İŞLEME (TAM PARİTE): {CLASS_MAP[HEDEF_SINIF]} ---")
    print(f"   Smoothing : EMA(alpha={EMA_ALPHA}) + Freezing(vis<{VISIBILITY_THRESHOLD})")
    print(f"   MediaPipe : VIDEO | num_poses={NUM_POSES} | conf={DETECTION_CONFIDENCE}")
    print(f"   Flip      : cv2.flip(frame, 1) (canli motorla ayni)")
    print(f"   Sema      : label + a0..a{NUM_ANGLES-1} + v0..v{NUM_ANGLES-1}  "
          f"({NUM_FEATURES} ozellik/kare)")
    print(f"   Cikti     : {CIKTI_CSV}")

    if not os.path.exists(VIDEO_YOLU):
        print(f"❌ HATA: '{VIDEO_YOLU}' bulunamadı! Videonun klasörde olduğundan emin ol.")
        return

    if os.path.exists(CIKTI_CSV):
        print(
            f"⚠️  UYARI: '{CIKTI_CSV}' zaten var → bu klibin verisi SONUNA EKLENECEK.\n"
            f"    Eski (EMA'sız) veriyle karıştırmadığından emin ol. Baştan eğitim için\n"
            f"    önce bu dosyayı sil/yedekle."
        )

    # ── MediaPipe Ayarları (showcase_engine ile BİREBİR) ─────────────────────
    base_options = python.BaseOptions(model_asset_path=MEDIAPIPE_MODEL_PATH)
    options = vision.PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO,
        num_poses=NUM_POSES,
        min_pose_detection_confidence=DETECTION_CONFIDENCE,
        min_pose_presence_confidence=PRESENCE_CONFIDENCE,
        min_tracking_confidence=TRACKING_CONFIDENCE,
    )

    cap = cv2.VideoCapture(VIDEO_YOLU)

    # Videonun FPS'inden 'dt' (Delta Time) → canlı RC-1 ile aynı clamp
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0 or np.isnan(fps):
        fps = 30.0  # Güvenlik önlemi
    dt = float(np.clip(1.0 / fps, DT_MIN, DT_MAX))
    print(f"🎥 Video FPS: {fps:.2f} | dt (clamp'li): {dt:.4f} sn\n")

    veri_listesi     = []
    prev_angles      = None
    smoother         = LandmarkSmoother(EMA_ALPHA, VISIBILITY_THRESHOLD)
    frame_sayaci     = 0
    kaydedilen_frame = 0
    son_ts_ms        = -1   # VIDEO modu: kesinlikle artan timestamp garantisi

    with vision.PoseLandmarker.create_from_options(options) as landmarker:
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break  # Video bitti

            frame_sayaci += 1

            # ── PARİTE: canlı motor kareyi yatay çevirip (mirror) tespit eder ──
            frame = cv2.flip(frame, 1)

            # ── VIDEO modu için kesinlikle artan ms timestamp (canlı guard) ───
            ts_ms = int(cap.get(cv2.CAP_PROP_POS_MSEC))
            if ts_ms <= son_ts_ms:
                ts_ms = son_ts_ms + 1
            son_ts_ms = ts_ms

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image  = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            result    = landmarker.detect_for_video(mp_image, ts_ms)

            # ── Ekran bilgisi ─────────────────────────────────────────────────
            cv2.putText(frame, f"Islem: {CLASS_MAP[HEDEF_SINIF]}", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, f"Frame: {frame_sayaci} | Kayit: {kaydedilen_frame}",
                        (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

            if result.pose_world_landmarks and result.pose_landmarks:
                # Veri toplama tek-özneliktir → birincil pozu (index 0) al.
                l2d_raw = result.pose_landmarks[0]
                l3d_raw = result.pose_world_landmarks[0]

                # ── Bulgu #4 paritesi: EMA + Freezing — smooth() TEK çağrı ────
                # (state kare başına bir kez ilerler; çift çağrı EMA'yı bozardı)
                l2d, l3d = smoother.smooth(l2d_raw, l3d_raw)

                # ── Özellik çıkarımı: SMOOTHED l3d + clamp'li dt (canlı ile aynı)
                features = extract_upper_body_angles_with_velocity(l3d, prev_angles, dt)
                assert len(features) == NUM_FEATURES, "Beklenmeyen özellik boyutu!"
                prev_angles = features[:NUM_ANGLES].copy()

                # ── CSV satırı: a0..a9, v0..v9 (canlı flatten ile BİREBİR) ────
                row = {"label": HEDEF_SINIF}
                for i in range(NUM_ANGLES):
                    row[f"a{i}"] = float(features[i])
                for i in range(NUM_ANGLES):
                    row[f"v{i}"] = float(features[i + NUM_ANGLES])
                veri_listesi.append(row)
                kaydedilen_frame += 1

                # ── Görsel geri bildirim: YUMUŞATILMIŞ 2D iskelet noktaları ───
                h, w, _ = frame.shape
                for j in UST_GOVDE_IDS:
                    px, py = int(l2d[j].x * w), int(l2d[j].y * h)
                    cv2.circle(frame, (px, py), 5, (0, 165, 255), -1)
            else:
                # ── PARİTE: canlı motorun reset_on_lost davranışı ─────────────
                # Tespit kaybolunca EMA hafızası + hız sürekliliği sıfırlanır.
                # Böylece gap'ten sonraki ilk karenin hızı 0 olur (doğal segment sınırı).
                smoother.reset()
                prev_angles = None

            cv2.imshow("Video Analiz (EMA Parite)", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                print("Kullanıcı tarafından durduruldu.")
                break

    cap.release()
    cv2.destroyAllWindows()

    # ── CSV KAYIT (sütun sırası AÇIKÇA sabitlenir → gizli shape riski yok) ────
    if len(veri_listesi) > 0:
        sutunlar = (
            ["label"]
            + [f"a{i}" for i in range(NUM_ANGLES)]
            + [f"v{i}" for i in range(NUM_ANGLES)]
        )
        df = pd.DataFrame(veri_listesi, columns=sutunlar)

        dosya_var = os.path.exists(CIKTI_CSV)
        df.to_csv(CIKTI_CSV, mode="a" if dosya_var else "w",
                  header=not dosya_var, index=False)

        durum = "eklendi" if dosya_var else "oluşturuldu ve kaydedildi"
        print(
            f"\n✅ {CIKTI_CSV} dosyasına {kaydedilen_frame} frame "
            f"{CLASS_MAP[HEDEF_SINIF]} {durum}."
        )
        print(f"   Sütun sırası: {sutunlar}")
    else:
        print("\n❌ Videoda hiç insan algılanmadı veya veri çıkarılamadı.")


if __name__ == "__main__":
    main()
