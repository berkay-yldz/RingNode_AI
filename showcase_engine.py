# RINGNODE_AI/showcase_engine.py
"""
RingNode AI - Showcase Engine v3.3  (Gold Master — Ghost Punch Anti-Cheat Mimarisi)

SORUN: Laptop kamerası titremeyi 'Direkt'/'Kroşe' olarak sınıflandırıyor → Ghost Punches.

ÇÖZÜM: Yukarı-akış yumuşatma + 4 Katmanlı Bariyer (Gate) Sistemi
─────────────────────────────────────────────────────────────────────
  Pre    │ Landmark Freezing │ Görünürlük < %60 olan eklemde SON güvenilir
         │ + EMA (alpha=0.6) │ pozisyon dondurulur; kalan titreşim üssel
         │ (Bulgu #4)        │ hareketli ortalama ile bastırılır.
─────────────────────────────────────────────────────────────────────
  Gate 1 │ Visibility Check  │ Bilek landmark görünürlüğü < %60 → model
         │ (Bulgu #2)        │ çağrılmaz, kare 'Idle' sayılır VE tüm
         │                   │ hız/vektör tamponları temizlenir (stale peak yok).
─────────────────────────────────────────────────────────────────────
  Gate 2 │ Wrist Speed Gate  │ Model 'saldırı' dese bile bilek hızı
         │                   │ WRIST_SPEED_THRESHOLD altındaysa hit iptal.
─────────────────────────────────────────────────────────────────────
  Gate   │ 3D Vektör Doğrul. │ Direkt → Z-atılımı, Kroşe → X-savrulması,
  Z/X/Y  │ (Bulgu #5)        │ Aparkat → Y-atılımı (aşağıdan yukarı) zorunlu.
─────────────────────────────────────────────────────────────────────
  Gate 3 │ State Machine     │ Hasar verdikten sonra hit_armed = False.
         │ (Hit Arming)      │ Sadece 'Idle'/'Savunma' sonrası yeniden
         │                   │ hit_armed = True olur. Combo spam yok.
─────────────────────────────────────────────────────────────────────

Train-Serve Uyumu (Bulgu #1):
  - Eğitim verisi VIDEO modunda (temporal tracking) toplandı.
  - Bu motor da artık RunningMode.VIDEO + detect_for_video() kullanır;
    böylece landmark dağılımı eğitimle hizalanır ve min_tracking_confidence
    gerçekten etkin olur.
  - NOT (parite): EMA yalnızca servis tarafında uygulanıyor. %100 dağılım
    pariteyi için ideal olan, veri toplama hattına da aynı EMA'yı eklemek
    veya modeli EMA'lı veriyle yeniden eğitmektir.

OOP Tasarım Kararları:
  - LandmarkSmoother    → Görünürlük Hafızası (freezing) + EMA
  - PredictionPipeline  → Gate 1 + varyans idle + model + majority voting
  - HitDetector         → Gate 2 + Gate Z/X/Y + Gate 3 + combo yönetimi
  - FighterState        → Pipeline'ın tüm per-oyuncu durumunu taşır
  - BoxingMLP           → State_dict uyumluluğu için katman isimleri KORUNDU
"""

import cv2
import numpy as np
import torch
import torch.nn as nn
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from collections import deque, Counter
import types
import warnings
import joblib
import time

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

# ─── ARCADE SABİTLERİ ────────────────────────────────────────────────────────
FPS_SMOOTHING     = 10
MAX_HP            = 250
HASAR_MIKTARI     = {"Direkt": 10, "Krose": 15, "Aparkat": 20}
SALDIRI_SINIFLARI = frozenset({"Direkt", "Krose", "Aparkat"})
SAVUNMA_BLOK      = frozenset({"Savunma"})            # Hasarı tamamen iptal eder
RESET_SINIFLARI   = frozenset({"Idle", "Savunma"})    # Gate 3: hit_armed yeniden kuşanır

# ─── BARİYER EŞİK DEĞERLERİ ──────────────────────────────────────────────────
VISIBILITY_THRESHOLD  = 0.60   # Gate 1 + Freezing : MediaPipe bilek görünürlük alt sınırı
WRIST_SPEED_THRESHOLD = 0.015  # Gate 2 : Normalize ekran koor./frame min hız

# ─── BULGU #4: YUKARI-AKIŞ YUMUŞATMA ─────────────────────────────────────────
# EMA (üssel hareketli ortalama) ham MediaPipe titreşimini bastırır.
# alpha = anlık karenin ağırlığı (yüksek = daha az gecikme, daha az yumuşatma).
EMA_ALPHA = 0.6

# ─── SINIFA ÖZEL 3D VEKTÖR EŞİKLERİ ─────────────────────────────────────────
# MediaPipe world landmarks metre cinsindendir.
# Kamera yönü = negatif Z ekseni; pozitif X = kişinin solu; +Y = aşağı.
#
# Direkt (Jab/Cross): Yumruk DERİNLİK yönünde (Z−) ilerler.
#   30cm atılım / 6 frame ≈ 0.05 m/frame → eşik 0.04 konservatiflir.
#
# Kroşe (Hook): Bilek YATAY (X) yaylanır, Z atılımı ikincildir.
#   20cm lateral / 6 frame ≈ 0.033 m/frame → eşik 0.025 konservatiflir.
#
# Aparkat (Uppercut): Bilek aşağıdan yukarı (Y−) çıkar.
#   25cm dikey / 6 frame ≈ 0.042 m/frame → eşik 0.030 konservatiflir.
#
# Kalibrasyon notu: Eşikler laptop kamerasına göre ±%20 ayarlanabilir.
Z_THRUST_THRESHOLD    = 0.040  # Gate Z: Direkt için min kameraya Z atılımı (m/frame)
X_SWEEP_THRESHOLD     = 0.025  # Gate X: Kroşe için min yatay X savrulması  (m/frame)
Y_THRUST_THRESHOLD    = 0.030  # Gate Y: Aparkat için min yukarı Y atılımı  (m/frame)
HIT_COOLDOWN_SN       = 1.2    # Gate 3+ : Saniye bazlı yedek spam koruması
CONFIDENCE_THRESHOLD  = 0.72   # Model güven alt sınırı
VOTING_MIN_RATIO      = 0.50   # Çoğunluk oylama min oranı

# ─── MediaPipe Landmark ID'leri ───────────────────────────────────────────────
SOL_OMUZ_ID  = 11
SAG_OMUZ_ID  = 12
SOL_BILEK_ID = 15
SAG_BILEK_ID = 16
SOL_KALCA_ID = 23
TOPLAM_LANDMARK = 33

DEBUG_OVERLAY = True  # Gate durumlarını ekranda göster (prod'da False yapılabilir)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ═══════════════════════════════════════════════════════════════════════════════
# BÖLÜM 0: LandmarkSmoother  —  Görünürlük Hafızası (Freezing) + EMA  (Bulgu #4)
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
        """Oyuncu kaybolunca EMA / freezing hafızasını sıfırlar."""
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
# BÖLÜM 1: BoxingMLP
# NOT: Katman isimleri (layer1, layer2, out) bilinçli olarak korundu.
#      nn.Sequential'a geçilseydi state_dict anahtarları değişir, ağırlık
#      dosyası yüklenemezdi. Model ağırlıklarına DOKUNULMADI.
# ═══════════════════════════════════════════════════════════════════════════════
class BoxingMLP(nn.Module):
    def __init__(self, input_size, num_classes):
        super().__init__()
        self.layer1  = nn.Linear(input_size, 128)
        self.relu    = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        self.layer2  = nn.Linear(128, 64)
        self.out     = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.relu(self.layer1(x))
        x = self.dropout(x)
        x = self.relu(self.layer2(x))
        return self.out(x)


# ═══════════════════════════════════════════════════════════════════════════════
# BÖLÜM 2: FloatingText
# ═══════════════════════════════════════════════════════════════════════════════
class FloatingText:
    def __init__(self, metin: str, x: int, y: int, renk: tuple, omur: float = 1.5):
        self.metin = metin
        self.x = x
        self.y = y
        self.renk = renk
        self.olusturma_zamani = time.time()
        self.omur = omur


# ═══════════════════════════════════════════════════════════════════════════════
# BÖLÜM 3: FighterState  —  Pipeline'ın Tüm Per-Oyuncu Durumunu Taşır
# ═══════════════════════════════════════════════════════════════════════════════
class FighterState:
    """
    Bir oyuncunun hem oyun durumunu (hp, combo) hem de tahmin pipeline'ının
    durumunu (kuyruk, pred_history, Gate 2/Z/X/Y-3 değişkenleri) tek yerde tutar.
    Ayrıca kendi LandmarkSmoother örneğini (EMA + freezing) barındırır.
    """

    def __init__(self, isim: str, renk: tuple, baslangic_hp: int = MAX_HP):
        self.isim  = isim
        self.renk  = renk
        self.hp    = baslangic_hp

        # ── Bulgu #4: Görünürlük Hafızası + EMA ──────────────────────────────
        self.smoother = LandmarkSmoother(EMA_ALPHA, VISIBILITY_THRESHOLD)

        # ── Tahmin Pipeline ──────────────────────────────────────────────────
        self.kuyruk      = deque(maxlen=PENCERE_BOYUTU)
        self.pred_history = deque(maxlen=9)
        self.prev_angles = None
        self.mevcut_sinif = "Insan Yok"
        self.guven        = 0.0
        self.guncellendi  = False

        # ── Gate 2: Bilek Hızı Takibi ────────────────────────────────────────
        # Normalize ekran koordinatları (0-1) üzerinden per-frame öklidyen mesafe.
        # Titreme < 0.005, gerçek yumruk pik frame > 0.02 tipik değerler.
        self.prev_wrist_l: tuple | None = None
        self.prev_wrist_r: tuple | None = None
        self._hiz_tamponu = deque(maxlen=PENCERE_BOYUTU)
        self.peak_wrist_speed: float = 0.0

        # ── Gate Z/X/Y: 3D Vektörel Hareket Takibi (World Landmark) ───────────
        # Koordinat sistemi: origin = kalça merkezi, birim = metre.
        #   Z negatif  → kameraya yaklaşıyor (Direkt'in baskın yönü)
        #   X mutlak   → yatay savrulma     (Kroşe'nin baskın yönü)
        #   Y negatif  → yukarı atılım       (Aparkat'ın baskın yönü)
        # Her frame'de per-frame delta hesaplanır; rolling max pencerede tutulur.
        self.prev_wrist_l_3d: tuple | None = None
        self.prev_wrist_r_3d: tuple | None = None
        self._z_thrust_buf = deque(maxlen=PENCERE_BOYUTU)  # negatif Z delta büyüklüğü
        self._x_sweep_buf  = deque(maxlen=PENCERE_BOYUTU)  # mutlak X delta
        self._y_thrust_buf = deque(maxlen=PENCERE_BOYUTU)  # negatif Y delta büyüklüğü
        self.peak_z_thrust: float = 0.0   # pencerede maks kameraya Z atılımı
        self.peak_x_sweep:  float = 0.0   # pencerede maks yatay X savrulması
        self.peak_y_thrust: float = 0.0   # pencerede maks yukarı Y atılımı

        # ── Gate 3: State Machine ────────────────────────────────────────────
        # hit_armed = False → Idle/Savunma görülene kadar hasar verilemez.
        # HitDetector hasar verdikten sonra False yapar.
        # update_arm_state, RESET_SINIFLARI görünce True'ya döndürür.
        self.hit_armed        = True
        self.son_hasar_zamani = 0.0

        # ── Debug: Son red sebebi ─────────────────────────────────────────────
        self.hit_reject_reason: str = ""

        # ── Combo ─────────────────────────────────────────────────────────────
        self.combo_sayisi = 0
        self.combo_aktif  = False

    # ── Gate 2 Yardımcısı ────────────────────────────────────────────────────
    def update_wrist_speed(self, landmarks_2d) -> None:
        """
        Her frame'de 2D normalize bilek pozisyonlarından anlık hızı hesaplar.
        peak_wrist_speed = son PENCERE_BOYUTU frame içindeki maksimum.
        prev None ise (Gate 1 sonrası recovery) delta hesaplanmaz → sahte pik yok.
        """
        wl = (landmarks_2d[SOL_BILEK_ID].x, landmarks_2d[SOL_BILEK_ID].y)
        wr = (landmarks_2d[SAG_BILEK_ID].x, landmarks_2d[SAG_BILEK_ID].y)

        if self.prev_wrist_l is not None:
            hiz_l = float(np.linalg.norm(np.array(wl) - np.array(self.prev_wrist_l)))
            hiz_r = float(np.linalg.norm(np.array(wr) - np.array(self.prev_wrist_r)))
            self._hiz_tamponu.append(max(hiz_l, hiz_r))
            self.peak_wrist_speed = max(self._hiz_tamponu)

        self.prev_wrist_l = wl
        self.prev_wrist_r = wr

    # ── Gate Z/X/Y Yardımcısı ────────────────────────────────────────────────
    def update_wrist_3d(self, landmarks_3d) -> None:
        """
        Her frame'de world landmark'lardan 3D bilek delta vektörünü hesaplar.

        Z atılımı    → delta_z negatif olmalı (kameraya geliyor).  max(0, −Δz)
        X savrulması  → abs(Δx), Kroşe'nin yatay dönüşü
        Y atılımı    → delta_y negatif olmalı (yukarı = küçük y).   max(0, −Δy)

        Her iki bilek de hesaplanır; büyük olan rolling tampona yazılır.
        prev None ise (recovery) delta üretilmez → bayat/sahte pik engellenir.
        """
        wl = (landmarks_3d[SOL_BILEK_ID].x,
              landmarks_3d[SOL_BILEK_ID].y,
              landmarks_3d[SOL_BILEK_ID].z)
        wr = (landmarks_3d[SAG_BILEK_ID].x,
              landmarks_3d[SAG_BILEK_ID].y,
              landmarks_3d[SAG_BILEK_ID].z)

        if self.prev_wrist_l_3d is not None:
            # Z: negatif delta = kameraya atılım; pozitif delta = geri çekilme (önemsiz)
            thrust_l = max(0.0, -(wl[2] - self.prev_wrist_l_3d[2]))
            thrust_r = max(0.0, -(wr[2] - self.prev_wrist_r_3d[2]))
            self._z_thrust_buf.append(max(thrust_l, thrust_r))
            self.peak_z_thrust = max(self._z_thrust_buf)

            # X: mutlak lateral hareket (Kroşe'de dominant; Direkt'te küçük)
            sweep_l = abs(wl[0] - self.prev_wrist_l_3d[0])
            sweep_r = abs(wr[0] - self.prev_wrist_r_3d[0])
            self._x_sweep_buf.append(max(sweep_l, sweep_r))
            self.peak_x_sweep = max(self._x_sweep_buf)

            # Y: negatif delta = yukarı atılım (MediaPipe'ta aşağı = +y).
            #    Aparkat'ın aşağıdan-yukarı çıkışını ölçer.
            #    (Kameranız ters konumdaysa işaret tek satırda çevrilebilir.)
            ythrust_l = max(0.0, -(wl[1] - self.prev_wrist_l_3d[1]))
            ythrust_r = max(0.0, -(wr[1] - self.prev_wrist_r_3d[1]))
            self._y_thrust_buf.append(max(ythrust_l, ythrust_r))
            self.peak_y_thrust = max(self._y_thrust_buf)

        self.prev_wrist_l_3d = wl
        self.prev_wrist_r_3d = wr

    # ── Gate 3 State Machine ──────────────────────────────────────────────────
    def update_arm_state(self, sinif: str) -> None:
        """
        Tahmin sonucuna göre state machine geçişi yapar.
        Sadece RESET_SINIFLARI (Idle / Savunma) hit_armed'ı yeniden kuşandırır.
        hit_armed'ı False'a almak HitDetector'ın görevidir.
        """
        if sinif in RESET_SINIFLARI:
            self.hit_armed = True
        self.mevcut_sinif = sinif

    # ── Bulgu #2: Hız/Vektör Tamponu Temizliği ────────────────────────────────
    def clear_motion_buffers(self) -> None:
        """
        Gate 1 (görünürlük) tetiklendiğinde TÜM hız/vektör tamponlarını ve pik
        değerlerini sıfırlar → düşük görünürlükteki titreşimden gelen 'stale peak'
        bir sonraki saldırıda Gate 2/Z/X/Y'yi otomatik geçemez (Ghost Punch önlemi).

        Ayrıca hız sürekliliğini (prev_* = None) keser: görünürlük geri geldiğinde
        donmuş→gerçek pozisyon sıçraması sahte bir delta/pik üretmez (recovery guard).
        """
        # Gate 2 — 2D bilek hızı
        self._hiz_tamponu.clear()
        self.peak_wrist_speed = 0.0
        self.prev_wrist_l = None
        self.prev_wrist_r = None

        # Gate Z / X / Y — 3D vektör
        self._z_thrust_buf.clear()
        self._x_sweep_buf.clear()
        self._y_thrust_buf.clear()
        self.peak_z_thrust = 0.0
        self.peak_x_sweep  = 0.0
        self.peak_y_thrust = 0.0
        self.prev_wrist_l_3d = None
        self.prev_wrist_r_3d = None

        # Açısal hız sürekliliği de kesilir (recovery'de yapay velocity spike önlemi)
        self.prev_angles = None

    # ── Temizlik ─────────────────────────────────────────────────────────────
    def reset_on_lost(self) -> None:
        """Sporcu görüntüden kaybolduğunda pipeline'ı tamamen sıfırlar."""
        self.kuyruk.clear()
        self.pred_history.clear()
        self.clear_motion_buffers()   # hız/vektör tamponları + prev_* (prev_angles dahil)
        self.smoother.reset()         # EMA / freezing hafızası
        self.hit_reject_reason = ""
        self.mevcut_sinif      = "Insan Yok"
        self.guncellendi       = False


# ═══════════════════════════════════════════════════════════════════════════════
# BÖLÜM 4: PredictionPipeline  —  Gate 1 + Idle + Model + Majority Voting
# ═══════════════════════════════════════════════════════════════════════════════
class PredictionPipeline:
    """
    Bir frame için tahmin üretir. Sorumluluk zinciri:
      1. Gate 1 (Visibility)     → bilek görünmüyorsa Idle döner, model yok,
                                    tüm hız/vektör tamponları temizlenir (Bulgu #2)
      2. Pencere dolu mu?        → dolmadıysa "Kalibre ediliyor..." döner
      3. Varyans Idle kontrolü   → hareket yoksa Idle döner, model yok
      4. Model Inference         → MLP softmax çıktısı
      5. Majority Voting         → güvensiz tahminleri filtreler
    """

    def __init__(self, model: BoxingMLP, scaler, device):
        self.model  = model
        self.scaler = scaler
        self.device = device

    # ── Varyans tabanlı Idle ─────────────────────────────────────────────────
    @staticmethod
    def _is_idle(kuyruk: deque) -> bool:
        stds = np.std(np.array(kuyruk)[:, :10], axis=0)
        return float(np.max(stds)) < IDLE_VARIANCE_THRESHOLD

    # ── Model çağrısı ────────────────────────────────────────────────────────
    def _infer(self, kuyruk: deque) -> tuple[str, float]:
        pencere_mat = np.array(kuyruk)           # (PENCERE_BOYUTU, 20)

        # RC-3a: NaN / Inf erken çıkış
        # dt patlamasından gelen tek kirli frame tüm pencereyi bozar.
        # np.isfinite False döndürürse modeli hiç yormadan Idle dön.
        if not np.all(np.isfinite(pencere_mat)):
            return "Idle", 1.0

        # RC-3b: OOD (Out-of-Distribution) koruması
        # Açılar biyomekanik olarak [0°, 180°] dışına çıkamaz.
        pencere_mat[:, :10] = np.clip(pencere_mat[:, :10], 0.0,   180.0)
        pencere_mat[:, 10:] = np.clip(pencere_mat[:, 10:], -500.0, 500.0)

        pencere = pencere_mat.flatten()
        scaled  = self.scaler.transform([pencere])
        with torch.no_grad():
            logits = self.model(torch.FloatTensor(scaled).to(self.device))
            prob   = torch.softmax(logits, dim=1)[0]
            conf, idx = torch.max(prob, 0)
        return SINIFLAR[idx.item()], float(conf.item())

    # ── Ana süreç ────────────────────────────────────────────────────────────
    def process(self, sporcu: FighterState, landmarks_2d) -> tuple[str, float]:
        """
        landmarks_2d: (yumuşatılmış) 2D landmark listesi (visibility erişimi için).
        Döner: (sinif_adi, guven_skoru)
        """

        # ── GATE 1: Görünürlük Kontrolü (Bulgu #2) ───────────────────────────
        vis_l = landmarks_2d[SOL_BILEK_ID].visibility
        vis_r = landmarks_2d[SAG_BILEK_ID].visibility
        if vis_l < VISIBILITY_THRESHOLD or vis_r < VISIBILITY_THRESHOLD:
            # Bilek(ler) yeterince görünmüyor → gürültü riski yüksek.
            # Güvenli yol: Idle döndür + geçmişi VE tüm hız/vektör tamponlarını
            # temizle ki bayat pik sonraki saldırıyı bedavaya geçirmesin.
            sporcu.pred_history.clear()
            sporcu.clear_motion_buffers()
            return "Idle", 1.0

        # ── Pencere dolmadıysa model çağırma ─────────────────────────────────
        if len(sporcu.kuyruk) < PENCERE_BOYUTU:
            return "Kalibre ediliyor...", 0.0

        # ── Varyans tabanlı Idle ─────────────────────────────────────────────
        if self._is_idle(sporcu.kuyruk):
            sporcu.pred_history.clear()
            return "Idle", 1.0

        # ── Model Inference ──────────────────────────────────────────────────
        sinif, guven = self._infer(sporcu.kuyruk)

        # RC-4: Agresif pred_history temizliği
        # Saldırı dışı her kesin sonuç geçmişi sıfırlar (Savunma dahil).
        if sinif in RESET_SINIFLARI:          # "Idle" veya "Savunma"
            sporcu.pred_history.clear()
            return sinif, (1.0 if sinif == "Idle" else guven)

        # ── Majority Voting (Anti-Flicker) ───────────────────────────────────
        sporcu.pred_history.append(sinif)
        counter    = Counter(sporcu.pred_history)
        en_cok, oy = counter.most_common(1)[0]
        oy_orani   = oy / len(sporcu.pred_history)

        # RC-4: Şüpheli (?) durumunda da geçmişi sıfırla
        if oy_orani < VOTING_MIN_RATIO or guven < CONFIDENCE_THRESHOLD:
            sporcu.pred_history.clear()
            return "?", 0.0

        return en_cok, guven


# ═══════════════════════════════════════════════════════════════════════════════
# BÖLÜM 5: HitDetector  —  Gate 2 + Gate Z/X/Y + Gate 3 + Combo Yönetimi
# ═══════════════════════════════════════════════════════════════════════════════
class HitDetector:
    """
    Hasar (hit) kararını ve yan etkilerini (HP düşürme, combo, floating text)
    yönetir. PredictionPipeline çıktısını alır, kalan bariyerlerden geçirir.

    Gate 2     – Bilek Hızı   : peak_wrist_speed eşiğin altındaysa → Ghost Punch, iptal.
    Gate Z/X/Y – 3D Vektör    : sınıfa özel yön doğrulaması (Direkt→Z, Kroşe→X, Aparkat→Y).
    Gate 3     – State Machine : hit_armed False ise → henüz Idle/Savunma görmedi, iptal.
    """

    def __init__(self):
        self.floating_texts: list[FloatingText] = []

    def check_and_apply(
        self,
        saldiran: FighterState,
        savunan:  FighterState,
        zaman:    float,
        w:        int,
    ) -> bool:
        """
        True döner → isabet gerçekleşti.
        False döner → en az bir bariyer blok koydu; saldiran.hit_reject_reason sebebi yazar.

        Bariyer sırası (önce ucuz kontroller, sonra 3D hesaplar):
          1. Saldırı sınıfı mı?
          2. Savunma bloğu?
          3. Gate 3a — Cooldown
          4. Gate 3b — State Machine (hit_armed)
          5. Gate 2  — Genel bilek hızı (2D)
          6. Gate Z/X/Y — Sınıfa özel 3D vektör
        """

        # Saldırı sınıfında değilse red sebebi yazmaya gerek yok
        sinif = saldiran.mevcut_sinif
        if sinif not in SALDIRI_SINIFLARI:
            saldiran.hit_reject_reason = ""
            return False

        # ── Savunma Bloğu ────────────────────────────────────────────────────
        if savunan.mevcut_sinif in SAVUNMA_BLOK:
            saldiran.hit_reject_reason = "Bloklandi: Savunma"
            return False

        # ── Gate 3a: Cooldown ─────────────────────────────────────────────────
        if (zaman - saldiran.son_hasar_zamani) < HIT_COOLDOWN_SN:
            saldiran.hit_reject_reason = "Reddedildi: Cooldown"
            return False

        # ── Gate 3b: State Machine ────────────────────────────────────────────
        if not saldiran.hit_armed:
            saldiran.hit_reject_reason = "Reddedildi: Reset Bekle"
            return False

        # ── Gate 2: Genel Bilek Hızı (2D) ────────────────────────────────────
        if saldiran.peak_wrist_speed < WRIST_SPEED_THRESHOLD:
            saldiran.hit_reject_reason = "Reddedildi: Hiz Yetersiz"
            return False

        # ── Gate Z/X/Y: Sınıfa Özel 3D Vektör Doğrulaması ────────────────────
        # Direkt → kameraya doğru Z atılımı zorunlu.
        if sinif == "Direkt":
            if saldiran.peak_z_thrust < Z_THRUST_THRESHOLD:
                saldiran.hit_reject_reason = "Reddedildi: Z-Atilimi Yok"
                return False

        # Kroşe → yatay X savrulması zorunlu.
        elif sinif == "Krose":
            if saldiran.peak_x_sweep < X_SWEEP_THRESHOLD:
                saldiran.hit_reject_reason = "Reddedildi: X-Savrulmasi Yok"
                return False

        # Aparkat → aşağıdan-yukarı Y atılımı zorunlu (Bulgu #5).
        # En yüksek hasarlı (20) hareket artık en zayıf kapı değil:
        # düz veya yatay bir el hareketi Aparkat'a sınıflansa bile,
        # net dikey çıkış olmadan isabet sayılmaz.
        elif sinif == "Aparkat":
            if saldiran.peak_y_thrust < Y_THRUST_THRESHOLD:
                saldiran.hit_reject_reason = "Reddedildi: Y-Atilimi Yok"
                return False

        # ── Tüm bariyerler geçildi → İSABET ──────────────────────────────────
        saldiran.hit_reject_reason = ""   # Başarılı hit → sıfırla

        hasar = HASAR_MIKTARI.get(sinif, 10)
        savunan.hp                = max(0, savunan.hp - hasar)
        saldiran.son_hasar_zamani = zaman

        # Gate 3: Silahı kapat; Idle/Savunma döngüsü tetikleyene kadar beklenir
        saldiran.hit_armed = False

        # RC-6: Rolling max tamponlarını anında sıfırla.
        # Eski yumruğun yüksek peak değerleri deque(maxlen) içinde yaşamaya devam
        # ederse, hit_armed yeniden True olduğunda stale peak Gate Z/X/Y'yi otomatik
        # geçirir. Sıfırlama, bir sonraki hareketin kendi verisini sıfırdan
        # kazanmasını zorunlu kılar.
        saldiran._z_thrust_buf.clear()
        saldiran._x_sweep_buf.clear()
        saldiran._y_thrust_buf.clear()
        saldiran.peak_z_thrust = 0.0
        saldiran.peak_x_sweep  = 0.0
        saldiran.peak_y_thrust = 0.0

        # Combo
        saldiran.combo_sayisi += 1
        saldiran.combo_aktif   = saldiran.combo_sayisi >= 2
        savunan.combo_sayisi   = 0
        savunan.combo_aktif    = False

        fx = (w // 4) if "SOL" in saldiran.isim else (3 * w // 4)
        self.floating_texts.append(
            FloatingText(f"-{hasar} HP!", fx, 300, (0, 0, 255))
        )
        return True

    def process_floating_texts(self, frame, simdiki_zaman: float) -> None:
        aktif = []
        for ft in self.floating_texts:
            gecen = simdiki_zaman - ft.olusturma_zamani
            if gecen < ft.omur:
                y = int(ft.y - gecen * 50)
                cv2.putText(frame, ft.metin, (ft.x, y),
                            cv2.FONT_HERSHEY_DUPLEX, 1.2, ft.renk, 3)
                aktif.append(ft)
        self.floating_texts = aktif


# ═══════════════════════════════════════════════════════════════════════════════
# BÖLÜM 6: UI Çizim Fonksiyonları
# ═══════════════════════════════════════════════════════════════════════════════
def draw_health_bars(frame, sp1: FighterState, sp2: FighterState, w: int, h: int):
    cx    = w // 2
    bar_w = cx - 60
    cv2.line(frame, (cx, 0), (cx, h), (0, 255, 255), 4)

    for bar_x, sporcu in [(20, sp1), (cx + 40, sp2)]:
        cv2.rectangle(frame, (bar_x, 30), (bar_x + bar_w, 60), (0, 0, 255), -1)
        px = int((sporcu.hp / MAX_HP) * bar_w)
        if px > 0:
            cv2.rectangle(frame, (bar_x, 30), (bar_x + px, 60), (0, 255, 0), -1)
        cv2.putText(frame, sporcu.isim, (bar_x, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)


def draw_combo_banner(frame, sporcu: FighterState, w: int, h: int, taraf: str):
    if not sporcu.combo_aktif or sporcu.combo_sayisi < 2:
        return
    metin              = f"{sporcu.combo_sayisi}-HIT COMBO!"
    olcek, kalinlik    = 1.5, 4
    (tw, _), _         = cv2.getTextSize(metin, cv2.FONT_HERSHEY_DUPLEX, olcek, kalinlik)
    cx = (w // 4 - tw // 2) if taraf == "sol" else (3 * w // 4 - tw // 2)
    cy = h // 4
    cv2.putText(frame, metin, (cx + 3, cy + 3),
                cv2.FONT_HERSHEY_DUPLEX, olcek, (0, 0, 0), kalinlik + 2, cv2.LINE_AA)
    cv2.putText(frame, metin, (cx, cy),
                cv2.FONT_HERSHEY_DUPLEX, olcek, sporcu.renk, kalinlik, cv2.LINE_AA)


def draw_upper_body_skeleton(frame, landmarks, w: int, h: int, renk: tuple):
    BAGLANTLAR = [
        (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),
        (11, 23), (12, 24), (23, 24),
    ]
    pts = {i: (int(landmarks[i].x * w), int(landmarks[i].y * h)) for i in range(33)}
    for a, b in BAGLANTLAR:
        cv2.line(frame, pts[a], pts[b], renk, 3)
    for p in [11, 12, 13, 14, 15, 16, 23, 24]:
        cv2.circle(frame, pts[p], 6, (255, 255, 255), -1)
        cv2.circle(frame, pts[p], 8, renk, 2)


def draw_gate_debug(frame, sporcu: FighterState, w: int, h: int, taraf: str):
    """
    Geliştirme overlay'i — 6 satır bilgi:
      Satır 1: G2 Genel bilek hızı (2D)
      Satır 2: GZ Z-atılımı   (Direkt için)
      Satır 3: GX X-savrulması (Kroşe için)
      Satır 4: GY Y-atılımı   (Aparkat için)
      Satır 5: G3 State Machine (hit_armed)
      Satır 6: Son red sebebi (saldırı sınıfındayken kırmızı yazı)

    Renk kodu: Yeşil = eşik geçildi / EVET, Kırmızı = eşik geçilemedi / HAYIR
    DEBUG_OVERLAY = False ile tüm panel kapatılır.
    """
    x      = 20 if taraf == "sol" else w // 2 + 20
    y      = h - 115
    yesil  = (0, 220, 0)
    sari   = (0, 200, 220)   # bilgi rengi (saldırı sınıfı dışındayken nötr)
    kirmizi = (0, 60, 255)
    font   = cv2.FONT_HERSHEY_SIMPLEX
    olcek  = 0.40
    kalinlik = 1

    hiz_ok = sporcu.peak_wrist_speed >= WRIST_SPEED_THRESHOLD
    z_ok   = sporcu.peak_z_thrust    >= Z_THRUST_THRESHOLD
    x_ok   = sporcu.peak_x_sweep     >= X_SWEEP_THRESHOLD
    y_ok   = sporcu.peak_y_thrust    >= Y_THRUST_THRESHOLD
    armed  = sporcu.hit_armed
    saldiri_aktif = sporcu.mevcut_sinif in SALDIRI_SINIFLARI

    satirlar = [
        (f"G2 Hiz :{sporcu.peak_wrist_speed:.3f} (esik {WRIST_SPEED_THRESHOLD})",
         yesil if hiz_ok else kirmizi),

        (f"GZ Atil:{sporcu.peak_z_thrust:.3f} (esik {Z_THRUST_THRESHOLD})",
         yesil if z_ok else (kirmizi if saldiri_aktif else sari)),

        (f"GX Savr:{sporcu.peak_x_sweep:.3f} (esik {X_SWEEP_THRESHOLD})",
         yesil if x_ok else (kirmizi if saldiri_aktif else sari)),

        (f"GY Atil:{sporcu.peak_y_thrust:.3f} (esik {Y_THRUST_THRESHOLD})",
         yesil if y_ok else (kirmizi if saldiri_aktif else sari)),

        (f"G3 Armed: {'EVET' if armed else 'RESET BEKLE'}",
         yesil if armed else kirmizi),
    ]

    for i, (metin, renk) in enumerate(satirlar):
        cv2.putText(frame, metin, (x, y + i * 15), font, olcek, renk, kalinlik)

    # Son red sebebi: yalnızca saldırı sınıfındayken ve sebep varsa göster
    if saldiri_aktif and sporcu.hit_reject_reason:
        cv2.putText(
            frame,
            sporcu.hit_reject_reason,
            (x, y + len(satirlar) * 15),
            font, olcek, kirmizi, kalinlik,
        )


# ═══════════════════════════════════════════════════════════════════════════════
# BÖLÜM 7: Ana Oyun Döngüsü
# ═══════════════════════════════════════════════════════════════════════════════
def main():
    print("🥊 RingNode AI v3.3 — GOLD MASTER (Bulgu #1/#2/#4/#5 düzeltmeleri aktif)")
    print(f"   Pre    : Freezing<{VISIBILITY_THRESHOLD} + EMA(alpha={EMA_ALPHA})")
    print(f"   Mode   : RunningMode.VIDEO (egitimle hizalandi, tracking ETKIN)")
    print(f"   Gate 1 : Visibility    < {VISIBILITY_THRESHOLD}  (+ tampon temizligi)")
    print(f"   Gate 2 : Wrist Speed   < {WRIST_SPEED_THRESHOLD}")
    print(f"   Gate Z : Z Atilimi     < {Z_THRUST_THRESHOLD}  (Direkt)")
    print(f"   Gate X : X Savrulmasi  < {X_SWEEP_THRESHOLD}  (Krose)")
    print(f"   Gate Y : Y Atilimi     < {Y_THRUST_THRESHOLD}  (Aparkat)")
    print(f"   Gate 3 : State Machine  hit_armed")
    print(f"   Cihaz  : {device}\n")

    # ── Model & Scaler Yükle ──────────────────────────────────────────────────
    scaler = joblib.load(SCALER_PATH)
    model  = BoxingMLP(INPUT_SIZE, NUM_CLASSES).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
    model.eval()

    # ── MediaPipe Landmarker (Bulgu #1: VIDEO modu) ───────────────────────────
    base_options = python.BaseOptions(model_asset_path=MEDIAPIPE_MODEL_PATH)
    options = vision.PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO,   # eğitim verisi de VIDEO modunda toplandı
        num_poses=2,
        min_pose_detection_confidence=0.7,
        min_pose_presence_confidence=0.7,
        min_tracking_confidence=0.7,             # VIDEO modunda artık ETKİN
    )

    # ── Sistem Bileşenleri ────────────────────────────────────────────────────
    sp1          = FighterState("P1 (SOL)", (50, 255, 50))
    sp2          = FighterState("P2 (SAG)", (0, 165, 255))
    pipeline     = PredictionPipeline(model, scaler, device)
    hit_detector = HitDetector()

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    fps_queue       = deque(maxlen=FPS_SMOOTHING)
    baslangic_zamani = time.time()
    prev_time       = baslangic_zamani
    son_ts_ms       = -1   # VIDEO modu: kesinlikle artan timestamp garantisi

    with vision.PoseLandmarker.create_from_options(options) as landmarker:
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break

            frame = cv2.flip(frame, 1)
            h, w, _ = frame.shape

            now = time.time()
            # RC-1: dt Clamping — taban VE tavan.
            dt  = float(np.clip(now - prev_time, 0.015, 0.100))
            fps_queue.append(1.0 / dt)
            prev_time = now

            # VIDEO modu için kesinlikle artan milisaniye timestamp
            ts_ms = int((now - baslangic_zamani) * 1000)
            if ts_ms <= son_ts_ms:
                ts_ms = son_ts_ms + 1
            son_ts_ms = ts_ms

            # ── MediaPipe Pose Tespiti (VIDEO modu = temporal tracking) ───────
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result    = landmarker.detect_for_video(
                mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb),
                ts_ms,
            )

            sp1.guncellendi = False
            sp2.guncellendi = False

            if result.pose_landmarks:
                for l2d_raw, l3d_raw in zip(
                    result.pose_landmarks, result.pose_world_landmarks
                ):
                    # ── Bulgu #8: MUTLAK ekran konumuna göre kimlik ataması ────
                    # Omuz orta noktasının X'i ekranın sol yarısındaysa (<0.5)
                    # oyuncu P1 (SOL), değilse P2 (SAG). Sıralama/indeks YOK; bu
                    # sayede tek oyuncu kalsa ya da oyuncular kameraya girip çıksa
                    # da kimlikler (Sol/Sağ) takla atmaz.
                    center_x = (l2d_raw[SOL_OMUZ_ID].x + l2d_raw[SAG_OMUZ_ID].x) / 2.0
                    sporcu = sp1 if center_x < 0.5 else sp2

                    # İki tespit de aynı yarıya düşerse çakışmayı önle
                    # (bu karede o tarafı ilk dolduran tespit geçerlidir).
                    if sporcu.guncellendi:
                        continue

                    # ── Bulgu #4: Görünürlük Hafızası (freezing) + EMA ────────
                    # Ham landmark'lar motora girmeden önce yumuşatılır.
                    l2d, l3d = sporcu.smoother.smooth(l2d_raw, l3d_raw)

                    # Gate 2: 2D normalize bilek hızını güncelle
                    sporcu.update_wrist_speed(l2d)

                    # Gate Z/X/Y: 3D world landmark'lardan vektörel hareketi güncelle
                    sporcu.update_wrist_3d(l3d)

                    # 3D world landmark'lardan 20 özellik çıkar → kuyruğa ekle
                    features = extract_upper_body_angles_with_velocity(
                        l3d, sporcu.prev_angles, dt
                    )
                    sporcu.prev_angles = features[:10].copy()
                    sporcu.kuyruk.append(features)
                    sporcu.guncellendi = True

                    draw_upper_body_skeleton(frame, l2d, w, h, sporcu.renk)

                    # Bariyerli Tahmin Pipeline (Gate 1 burada)
                    sinif, guven = pipeline.process(sporcu, l2d)

                    # Gate 3 State Machine geçişi
                    sporcu.update_arm_state(sinif)
                    sporcu.guven = guven

            # ── Kayıp Oyuncu Temizliği ────────────────────────────────────────
            for sporcu in (sp1, sp2):
                if not sporcu.guncellendi:
                    sporcu.reset_on_lost()

            # ── Hit Detection (Gate 2 + Gate Z/X/Y + Gate 3 burada) ──────────
            if sp1.guncellendi and sp2.guncellendi:
                hit_detector.check_and_apply(sp1, sp2, now, w)
                hit_detector.check_and_apply(sp2, sp1, now, w)

            # ── Oyun Bitişi ───────────────────────────────────────────────────
            if sp1.hp <= 0 or sp2.hp <= 0:
                kazanan = sp1.isim if sp2.hp <= 0 else sp2.isim
                cv2.putText(
                    frame, f"K.O! {kazanan} KAZANDI!",
                    (w // 4, h // 2),
                    cv2.FONT_HERSHEY_DUPLEX, 1.5, (0, 0, 255), 4,
                )

            # ── UI Katmanları ─────────────────────────────────────────────────
            draw_health_bars(frame, sp1, sp2, w, h)
            draw_combo_banner(frame, sp1, w, h, "sol")
            draw_combo_banner(frame, sp2, w, h, "sag")
            hit_detector.process_floating_texts(frame, now)

            if DEBUG_OVERLAY:
                draw_gate_debug(frame, sp1, w, h, "sol")
                draw_gate_debug(frame, sp2, w, h, "sag")

            # ── Alt Durum Çubuğu ──────────────────────────────────────────────
            fps_val = int(np.mean(fps_queue)) if fps_queue else 0
            durum   = (
                f"P1: {sp1.mevcut_sinif}({sp1.guven:.2f}) | "
                f"P2: {sp2.mevcut_sinif}({sp2.guven:.2f}) | "
                f"FPS: {fps_val}"
            )
            cv2.putText(frame, durum, (10, h - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)

            cv2.imshow("RingNode AI — ARCADE v3.3 (Gold Master)", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
