"""
RingNode AI - Merkezi Konfigürasyon Dosyası
Tüm sistem ayarları, eşik değerleri ve sınıflandırma kuralları buradan yönetilir.
"""

# ==========================================
# 1. IDLE (BEKLEME) AYARLARI
# ==========================================
# idle_threshold.py testlerinden elde edilen en yüksek titreme sınırı.
IDLE_VARIANCE_THRESHOLD = 11.0

# ==========================================
# 2. MODEL VE BİYOMEKANİK AYARLARI
# ==========================================
WINDOW_SIZE = 15                          # Sliding window frame sayısı (Pencere Boyutu)
NUM_ANGLES = 10                           # extract_upper_body_angles() çıktı boyutu
NUM_FEATURES = NUM_ANGLES * 2             # Açı + Açısal Hız = 20 feature/frame
INPUT_SIZE = NUM_FEATURES * WINDOW_SIZE   # = 300 (10 açı + 10 hız × 15 frame)

# ==========================================
# 3. SINIFLANDIRMA (CLASS) AYARLARI
# ==========================================
NUM_CLASSES = 4                           # Aparkat, Direkt, Krose, Savunma
SINIFLAR = {0: 'Aparkat', 1: 'Direkt', 2: 'Krose', 3: 'Savunma'}

# ==========================================
# 4. DOSYA VE YOL (PATH) AYARLARI
# ==========================================
DATA_PATH = 'boks_ustgovde_aci_veri.csv'  # Topladığımız veri setinin adı
MODEL_PATH = 'boxing_mlp_model.pth'       # Eğitilecek PyTorch modelinin kayıt yeri
SCALER_PATH = 'scaler.pkl'                # Standardizasyon dosyasının kayıt yeri
MEDIAPIPE_MODEL_PATH = 'pose_landmarker.task' # Yapay Zeka ağırlık dosyası