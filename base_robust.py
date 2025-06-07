import os
import numpy as np
import torch
import librosa
from torchaudio.transforms import TimeMasking, FrequencyMasking
from sklearn.metrics import f1_score, accuracy_score
from tensorflow.keras.models import load_model  # or joblib.load
from tqdm import tqdm

# ── CONFIG ─────────────────────────────────────────────────────────────────────
DATA_ROOT      = "data/gtzan"                   # root of genres/ subdir
SPLIT_DIR      = "data/gtzan-splits"            # contains test_filtered.txt
GTZAN_CLASSES  = ['blues','classical','country','disco','hiphop',
                  'jazz','metal','pop','reggae','rock']
SR             = 22050
DURATION       = 30
N_MELS         = 128
HOP_LENGTH     = 512
NUM_SEGMENTS   = 10
BATCH_SIZE     = 32

# 1) Compute some derived constants
samples_per_track = SR * DURATION
samples_per_seg   = samples_per_track // NUM_SEGMENTS
fixed_frames      = int(np.ceil(samples_per_seg / HOP_LENGTH))  # should be 130

# 2) Load your baseline CNN
baseline_cnn_model = load_model("basic_cnn.joblib")  # or use joblib.load if you prefer

# 3) Build the clean (un–corrupted) test set
test_list = []
with open(os.path.join(SPLIT_DIR, "test_filtered.txt")) as f:
    test_list = [line.strip() for line in f]

X_test = []
y_test = []
for rel_path in tqdm(test_list, desc="Building test set"):
    genre = rel_path.split("/")[0]
    label = GTZAN_CLASSES.index(genre)
    wav_path = os.path.join(DATA_ROOT, "genres_original", *rel_path.split("/"))
    
    # load + pad to 30s
    y_audio, _ = librosa.load(wav_path, sr=SR, duration=DURATION)
    if len(y_audio) < samples_per_track:
        y_audio = np.pad(y_audio, (0, samples_per_track-len(y_audio)), mode="constant")
    
    # segment + extract log-Mel, to dB, clamp, normalize
    for seg in range(NUM_SEGMENTS):
        start = seg * samples_per_seg
        end   = start + samples_per_seg
        chunk = y_audio[start:end]
        
        S    = librosa.feature.melspectrogram(
                   y=chunk, sr=SR, n_mels=N_MELS, hop_length=HOP_LENGTH
               )
        logS = librosa.power_to_db(S, ref=np.max, top_db=80.0)  # [-80..0]
        
        # pad/truncate to fixed_frames
        if logS.shape[1] < fixed_frames:
            pad = fixed_frames - logS.shape[1]
            logS = np.pad(logS, ((0,0),(0,pad)), mode="constant")
        else:
            logS = logS[:, :fixed_frames]
        
        # normalize to [0,1]
        spec_norm = (logS + 80.0) / 80.0
        
        X_test.append(spec_norm[..., np.newaxis])  # → (128,130,1)
        y_test.append(label)

X_test = np.array(X_test, dtype="float32")
y_test = np.array(y_test, dtype="int64")

# 4) Prepare your corruption transforms
time_mask = TimeMasking(time_mask_param=0)
freq_mask = FrequencyMasking(freq_mask_param=0)

# 5) Loop over corruption types & levels
corruption_tests = {
    'additive_noise': [0.0, 0.25, 0.5, 0.75, 1.0, 1.25],
    'time_mask':       [0, 50, 100, 150, 200, 250],
    'freq_mask':       [0, 20, 40, 60, 80, 100],
}

results = {}
for corr, levels in corruption_tests.items():
    results[corr] = []
    for lvl in levels:
        print(f"Baseline → {corr} @ {lvl}")
        
        # copy clean data
        Xc = X_test.copy()
        
        if corr == 'additive_noise':
            Xc += np.random.randn(*Xc.shape) * lvl
        
        else:
            # we need a (N,1,128,130) torch tensor to apply masks
            t = torch.from_numpy(Xc.transpose(0,3,1,2))  # (N,1,128,130)
            if corr == 'time_mask':
                time_mask.time_mask_param = int(lvl)
                t = time_mask(t)
            else:
                freq_mask.freq_mask_param = int(lvl)
                t = freq_mask(t)
            # back to NHWC
            Xc = t.numpy().transpose(0,2,3,1)
        
        # 6) Predict & score
        preds = baseline_cnn_model.predict(Xc, batch_size=BATCH_SIZE)
        if preds.ndim > 1 and preds.shape[1] > 1:
            preds = preds.argmax(axis=1)
        f1  = f1_score(y_test, preds, average='macro', zero_division=0)
        acc = accuracy_score(y_test, preds)
        
        results[corr].append({'level': lvl, 'f1': f1, 'acc': acc})
        print(f"  → F1 {f1:.4f}, Acc {acc:.4f}")

# 7) (Optionally) dump to JSON for plotting
import json
with open("baseline_robustness.json", "w") as f:
    json.dump(results, f, indent=2)
print("Done.")
