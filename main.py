import torch.multiprocessing as mp
# IMPORTANT: set_start_method should be called ONLY in the if __name__ == '__main__': block

import os
os.environ["AUDIOREAD_PRIORITY_BACKENDS"] = "ffmpeg"

import torch
torch.backends.cudnn.benchmark = True

import sys
import contextlib
import json
import numpy as np
import pandas as pd
import librosa
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchaudio.transforms as T
from torch.utils.data import Dataset, DataLoader
from torchvision import models
from torch.amp import GradScaler, autocast
from collections import Counter
from sklearn.metrics import f1_score, confusion_matrix
from tqdm import tqdm
from multiprocessing import Pool, cpu_count


SEED = 805
torch.manual_seed(SEED)
np.random.seed(SEED)

# ── CONFIGURATION ───────────────────────────────────────────────────────────────
GTZAN_ROOT       = "data/gtzan/genres"
FMA_ROOT         = "data/fma_small"
FMA_METADATA_CSV = "data/fma_metadata/tracks.csv"

# GTZAN Sturm Split definition files
GTZAN_SPLIT_DIR        = "data/gtzan-splits" 
GTZAN_TRAIN_SPLIT_FILE = os.path.join(GTZAN_SPLIT_DIR, "train-filtered.txt") 
GTZAN_VAL_SPLIT_FILE   = os.path.join(GTZAN_SPLIT_DIR, "valid-filtered.txt") 
GTZAN_TEST_SPLIT_FILE  = os.path.join(GTZAN_SPLIT_DIR, "test-filtered.txt") 

PRECOMP_DIR       = "precomputed_std_splits_specaug" # New dir for this run
PRECOMP_GTZAN_TRAIN_DIR = os.path.join(PRECOMP_DIR, "gtzan_train")
PRECOMP_GTZAN_VAL_DIR   = os.path.join(PRECOMP_DIR, "gtzan_val")
PRECOMP_GTZAN_TEST_DIR  = os.path.join(PRECOMP_DIR, "gtzan_test")

PRECOMP_FMA_TRAIN_DIR = os.path.join(PRECOMP_DIR, "fma_train")
PRECOMP_FMA_VAL_DIR   = os.path.join(PRECOMP_DIR, "fma_val")
PRECOMP_FMA_TEST_DIR  = os.path.join(PRECOMP_DIR, "fma_test") 

FMA_PRETRAIN_BATCH_SIZE   = 64
GTZAN_FINETUNE_BATCH_SIZE = 32
NUM_WORKERS    = 0      # For DataLoader during epoch iteration
CACHE_LOAD_WORKERS = 12 # For initial parallel caching into RAM (set to 0 for sequential)
PIN_MEMORY     = True
FMA_EPOCHS     = 25
GTZ_EPOCHS     = 45
FREEZE_EPOCHS  = 5 # Can increase if needed, e.g. 5
DEVICE         = torch.device("cuda" if torch.cuda.is_available() else "cpu")

FMA_WEIGHT_DECAY = 2e-3
FMA_LR = 5e-4
FMA_SCHEDULER_PATIENCE = 7
FMA_EARLY_STOPPING_PATIENCE = 10


GTZAN_WEIGHT_DECAY_FR = 1e-3
GTZAN_LR_FR = 1e-3

GTZAN_WEIGHT_DECAY_FT = 1.2e-3
GTZAN_LR_FT = 1e-5

SR          = 22050
N_FFT       = 2048
HOP_LENGTH  = 512
N_MELS      = 128
DURATION    = 30.0

# SpecAugment Configuration
APPLY_SPECAUGMENT_FMA   = True 
APPLY_SPECAUGMENT_GTZAN = False
SPECAUGMENT_FREQ_MASK_PARAM = 27   # Max width of frequency mask
SPECAUGMENT_TIME_MASK_PARAM = 70   # Max width of time mask
# ===================================

# ── 1. PARALLEL PREPROCESSING WORKER FUNCTION ───────────────────────────────────
def worker_preprocess_audio(args_tuple):
    audio_path, save_path_full, sr_const, duration_const, n_fft_const, hop_length_const, n_mels_const = args_tuple
    mel_transform_cpu = T.MelSpectrogram(
        sample_rate=sr_const, n_fft=n_fft_const, hop_length=hop_length_const, n_mels=n_mels_const, power=2.0
    ).cpu()
    db_transform_cpu = T.AmplitudeToDB(stype="power").cpu()
    try:
        with open(os.devnull, 'w') as devnull_file, contextlib.redirect_stderr(devnull_file):
            y_np, _ = librosa.load(path=audio_path, sr=sr_const, mono=True, duration=duration_const)
    except Exception:
        return None 
    target_len = int(duration_const * sr_const)
    if y_np.shape[0] < target_len:
        pad_amt = target_len - y_np.shape[0]
        y_np = np.concatenate([y_np, np.zeros(pad_amt)], axis=0)
    else:
        y_np = y_np[:target_len]
    waveform = torch.from_numpy(y_np).unsqueeze(0).float()
    with torch.no_grad():
        mel_spec = mel_transform_cpu(waveform)
        mel_db   = db_transform_cpu(mel_spec)
        mean = mel_db.mean()
        std  = mel_db.std(unbiased=False) + 1e-6
        mel_norm = (mel_db - mean) / std
        spec_3ch = mel_norm.repeat(1, 3, 1, 1).squeeze(0)
    os.makedirs(os.path.dirname(save_path_full), exist_ok=True)
    torch.save(spec_3ch, save_path_full)
    return save_path_full

# ── 1.5 ORCHESTRATOR FOR PARALLEL PREPROCESSING ───────────────────────────────────
def run_parallel_preprocessing(samples_with_labels, classes_list, save_root_dir, dataset_name_with_split, num_processes=None):
    print(f"Checking files for {dataset_name_with_split} precomputation...")
    os.makedirs(save_root_dir, exist_ok=True) 
    for genre_name in classes_list:
        os.makedirs(os.path.join(save_root_dir, genre_name), exist_ok=True)

    tasks_to_process = []
    if not samples_with_labels: # Handle empty input list
        print(f"No samples provided for {dataset_name_with_split} precomputation. Skipping.")
        return

    for audio_path, label_idx in samples_with_labels: 
        genre_name = classes_list[label_idx] 
        base_fname = os.path.basename(audio_path)
        name_no_ext, _ = os.path.splitext(base_fname)
        output_save_path = os.path.join(save_root_dir, genre_name, name_no_ext + ".pt")

        if os.path.exists(output_save_path):
            continue
        tasks_to_process.append(
            (audio_path, output_save_path, SR, DURATION, N_FFT, HOP_LENGTH, N_MELS)
        )
    if not tasks_to_process:
        print(f"All {dataset_name_with_split} files appear to be precomputed. Nothing to do.")
        return

    effective_num_processes = num_processes
    if effective_num_processes is None:
        effective_num_processes = CACHE_LOAD_WORKERS # Use same workers as caching for consistency
    effective_num_processes = min(effective_num_processes, cpu_count())
    if effective_num_processes <= 0: effective_num_processes = 1

    print(f"Starting parallel preprocessing for {len(tasks_to_process)} {dataset_name_with_split} files using {effective_num_processes} workers...")
    with Pool(processes=effective_num_processes) as pool:
        results = list(tqdm(pool.imap_unordered(worker_preprocess_audio, tasks_to_process),
                            total=len(tasks_to_process),
                            desc=f"Precomputing {dataset_name_with_split}", ncols=100))
    successful_count = sum(1 for r in results if r is not None)
    print(f"Finished precomputing {dataset_name_with_split}. {successful_count} successful, {len(tasks_to_process) - successful_count} failed/skipped.")

# ── 2. SAMPLE BUILDERS (Using Standard Splits) ───────────────────────────
def build_gtzan_samples_from_sturm_splits(gtzan_audio_root, gtzan_classes_list):
    class_to_idx = {genre: i for i, genre in enumerate(gtzan_classes_list)}
    def load_split_file_entries(filepath):
        parsed_entries = []
        try:
            with open(filepath, 'r') as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line: continue
                    parts = line.replace('\\', '/').split('/')
                    if len(parts) == 2:
                        genre_dirname_from_split = parts[0]
                        filename_with_ext_from_split = parts[1]
                        base_filename_from_split, _ = os.path.splitext(filename_with_ext_from_split)
                        parsed_entries.append((genre_dirname_from_split, base_filename_from_split))
                    else:
                        print(f"Warning [File: {os.path.basename(filepath)}, Line: {line_num}]: Unexpected format '{line}'. Expected 'genre/filename.ext'. Skipping.", file=sys.stderr)
        except FileNotFoundError:
            print(f"ERROR: GTZAN Split file not found: {filepath}. Please download/place it in '{GTZAN_SPLIT_DIR}'.", file=sys.stderr)
            sys.exit(1)
        return parsed_entries

    sturm_train_entries = load_split_file_entries(GTZAN_TRAIN_SPLIT_FILE)
    sturm_val_entries   = load_split_file_entries(GTZAN_VAL_SPLIT_FILE)
    sturm_test_entries  = load_split_file_entries(GTZAN_TEST_SPLIT_FILE)

    def create_sample_list_for_gtzan(split_entries, audio_root_path, all_classes_map):
        split_samples = []
        missing_files_count = 0
        for genre_dirname, base_filename in split_entries:
            if genre_dirname not in all_classes_map: continue
            label = all_classes_map[genre_dirname]
            actual_audio_filename = base_filename + ".au"
            constructed_audio_path = os.path.join(audio_root_path, genre_dirname, actual_audio_filename)
            if os.path.isfile(constructed_audio_path):
                split_samples.append((constructed_audio_path, label))
            else:
                if missing_files_count < 5: # Log first few misses
                     print(f"DEBUG: GTZAN audio file NOT FOUND for Sturm entry: {constructed_audio_path}", file=sys.stderr)
                missing_files_count += 1
        if missing_files_count > 0:
            print(f"WARNING: Total {missing_files_count} GTZAN audio files from split definition were NOT found on disk.", file=sys.stderr)
        return split_samples

    train_samples = create_sample_list_for_gtzan(sturm_train_entries, gtzan_audio_root, class_to_idx)
    val_samples   = create_sample_list_for_gtzan(sturm_val_entries,   gtzan_audio_root, class_to_idx)
    test_samples  = create_sample_list_for_gtzan(sturm_test_entries,  gtzan_audio_root, class_to_idx)
    return train_samples, val_samples, test_samples

def build_fma_samples_with_official_splits(fma_audio_root, metadata_csv):
    df_meta = pd.read_csv(metadata_csv, header=[0,1], index_col=0)
    if ("set", "subset") not in df_meta.columns: raise ValueError("Metadata CSV missing ('set', 'subset') column.")
    df_meta_small = df_meta[df_meta[("set", "subset")] == "small"].copy()
    genre_col_name = ('track', 'genre_top')
    if genre_col_name not in df_meta_small.columns:
        genre_cols = [col for col in df_meta_small.columns if isinstance(col, tuple) and col[1] == "genre_top"]
        if not genre_cols: raise ValueError("Metadata CSV missing 'genre_top' column.")
        genre_col_name = genre_cols[0]
    split_col_name = ('set', 'split')
    if split_col_name not in df_meta_small.columns: raise ValueError("Metadata CSV missing ('set', 'split') column.")
    relevant_cols_df = df_meta_small[[genre_col_name, split_col_name]].copy()
    relevant_cols_df.columns = ["genre_flat", "split_flat"]
    unique_genres = sorted(relevant_cols_df["genre_flat"].dropna().unique())
    class_to_idx = {g: i for i, g in enumerate(unique_genres)}
    train_samples, val_samples, test_samples = [], [], []
    missing_files_count = 0
    for track_id, row_data in relevant_cols_df.iterrows():
        genre, split_type = row_data["genre_flat"], row_data["split_flat"]
        if pd.isna(genre) or pd.isna(split_type): continue
        track_id_str = str(track_id).zfill(6) 
        subdir, filename = track_id_str[:3], track_id_str + ".mp3"
        audio_path = os.path.join(fma_audio_root, subdir, filename)
        if os.path.isfile(audio_path):
            label = class_to_idx[genre]
            sample_tuple = (audio_path, label) 
            if split_type == 'training': train_samples.append(sample_tuple)
            elif split_type == 'validation': val_samples.append(sample_tuple)
            elif split_type == 'test': test_samples.append(sample_tuple)
        else:
            if missing_files_count < 5:
                print(f"DEBUG: FMA audio file NOT FOUND: {audio_path}", file=sys.stderr)
            missing_files_count += 1
    if missing_files_count > 0:
        print(f"WARNING: Total {missing_files_count} FMA audio files from metadata were NOT found on disk.", file=sys.stderr)
    return train_samples, val_samples, test_samples, unique_genres

# ── 3. IN-MEMORY DATASET & CACHING WORKER (with SpecAugment) ───────────────────
def worker_load_pt_file(path):
    try: return torch.load(path, map_location='cpu')
    except Exception as e:
        print(f"Warning: worker_load_pt_file failed for {path}: {e}", file=sys.stderr)
        return None

class InMemorySpectrogramDataset(Dataset):
    def __init__(self, samples_paths_labels, dataset_name="Dataset", 
                 augment=False, num_load_workers=None):
        self.data, self.labels, self.augment = [], [], augment
        effective_load_workers = num_load_workers if num_load_workers is not None else CACHE_LOAD_WORKERS
        effective_load_workers = min(effective_load_workers, cpu_count())
        if effective_load_workers <= 0: effective_load_workers = 1
        
        print(f"Caching {dataset_name} into memory using {effective_load_workers} workers (Augment: {self.augment})...")
        paths_to_load = [p for p, l in samples_paths_labels]
        original_labels = [l for p, l in samples_paths_labels]

        if effective_load_workers > 1 and len(paths_to_load) >= effective_load_workers:
            with Pool(processes=effective_load_workers) as pool:
                results = list(tqdm(pool.imap(worker_load_pt_file, paths_to_load), 
                                    total=len(paths_to_load), desc=f"Caching {dataset_name}", ncols=100))
        else:
            print(f"Using sequential caching for {dataset_name} (files: {len(paths_to_load)})")
            results = [worker_load_pt_file(p) for p in tqdm(paths_to_load, desc=f"Caching {dataset_name}", ncols=100)]

        for i, loaded_tensor in enumerate(results):
            if loaded_tensor is not None:
                self.data.append(loaded_tensor); self.labels.append(original_labels[i])
        
        if not self.data: raise ValueError(f"No data loaded for {dataset_name}.")
        print(f"Finished caching {len(self.data)} samples for {dataset_name} into memory.")

        if self.augment:
            print(f"Initializing SpecAugment for {dataset_name}: FreqM={SPECAUGMENT_FREQ_MASK_PARAM}, TimeM={SPECAUGMENT_TIME_MASK_PARAM}")
            self.freq_masking = T.FrequencyMasking(freq_mask_param=SPECAUGMENT_FREQ_MASK_PARAM)
            self.time_masking = T.TimeMasking(time_mask_param=SPECAUGMENT_TIME_MASK_PARAM)

    def __len__(self): return len(self.data)
    def __getitem__(self, idx):
        spec, label = self.data[idx], self.labels[idx]
        if self.augment:
            spec_aug = spec.unsqueeze(0) 
            spec_aug = self.freq_masking(spec_aug)
            spec_aug = self.time_masking(spec_aug)
            spec = spec_aug.squeeze(0)
        return spec, label

# ── 4. MODEL AND TRAIN/VAL FUNCTIONS ─────────────────────────────────────────────
class FeatExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.feature = nn.Sequential(*list(resnet.children())[:-1])
    def forward(self, x):
        x = self.feature(x); return x.view(x.size(0), -1)

class Classifier(nn.Module):
    def __init__(self, in_dim, n_cls, dropout1=0.6, dropout2=0.3):
        super().__init__()
        self.dropout1  = nn.Dropout(dropout1); self.fc1 = nn.Linear(in_dim, 256)
        self.relu = nn.ReLU(); self.dropout2 = nn.Dropout(dropout2)
        self.fc2 = nn.Linear(256, n_cls)
    def forward(self, x):
        x = self.dropout1(x); x = self.fc1(x); x = self.relu(x)
        x = self.dropout2(x); x = self.fc2(x); return x

scaler = GradScaler()

def train_epoch(loader, feature_extractor, classifier, optimizer, class_weights, device_obj):
    feature_extractor.train(); classifier.train()
    total_loss, correct, num_samples = 0.0, 0, 0
    for x_cpu, y in tqdm(loader, desc="[TrainBatches]", leave=False, ncols=100):
        x = x_cpu.to(device_obj, non_blocking=PIN_MEMORY) 
        y = y.to(device_obj, non_blocking=PIN_MEMORY)
        batch_s = x.size(0); num_samples += batch_s
        optimizer.zero_grad()
        is_cuda = (device_obj.type == 'cuda')
        with autocast(device_type=device_obj.type, enabled=is_cuda):
            z = feature_extractor(x); out = classifier(z)
            loss = F.cross_entropy(out, y, weight=class_weights)
        if is_cuda: scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update()
        else: loss.backward(); optimizer.step()
        total_loss += loss.item() * batch_s
        preds = out.argmax(dim=1); correct += (preds == y).sum().item()
    return (total_loss / num_samples if num_samples > 0 else 0), \
           (correct / num_samples if num_samples > 0 else 0)

def validate_epoch(loader, feature_extractor, classifier, device_obj):
    feature_extractor.eval(); classifier.eval()
    total_loss, correct, num_samples = 0.0, 0, 0
    all_preds, all_gts = [], []
    with torch.no_grad():
        for x_cpu, y in tqdm(loader, desc="[ValBatches]", leave=False, ncols=100):
            x = x_cpu.to(device_obj, non_blocking=PIN_MEMORY)
            y = y.to(device_obj, non_blocking=PIN_MEMORY)
            batch_s = x.size(0); num_samples += batch_s
            is_cuda = (device_obj.type == 'cuda')
            with autocast(device_type=device_obj.type, enabled=is_cuda):
                z = feature_extractor(x); out = classifier(z)
                loss = F.cross_entropy(out, y)
            total_loss += loss.item() * batch_s
            preds = out.argmax(dim=1); correct += (preds == y).sum().item()
            all_preds.extend(preds.cpu().numpy()); all_gts.extend(y.cpu().numpy())
    avg_loss = total_loss / num_samples if num_samples > 0 else 0
    accuracy = correct / num_samples if num_samples > 0 else 0
    f1 = f1_score(all_gts, all_preds, average="macro", zero_division=0)
    cm = confusion_matrix(all_gts, all_preds) if all_gts and all_preds else None
    return avg_loss, accuracy, f1, cm

# ── 5. MAIN TRAINING PIPELINE ───────────────────────────────────────────────────
def main():
    print(f"Using device: {DEVICE}")
    print(f"DataLoader epochs: num_workers={NUM_WORKERS}, pin_memory={PIN_MEMORY}")
    print(f"Initial RAM Caching workers: {CACHE_LOAD_WORKERS if CACHE_LOAD_WORKERS > 0 else 'Sequential'}")
    print(f"SpecAugment: FMA={APPLY_SPECAUGMENT_FMA}, GTZAN={APPLY_SPECAUGMENT_GTZAN}")

    _gtzan_all_possible_classes = sorted([d for d in os.listdir(GTZAN_ROOT) if os.path.isdir(os.path.join(GTZAN_ROOT, d))])
    gtzan_raw_train, gtzan_raw_val, gtzan_raw_test = build_gtzan_samples_from_sturm_splits(GTZAN_ROOT, _gtzan_all_possible_classes)
    n_gtz_classes = len(_gtzan_all_possible_classes)
    print(f"GTZAN Sturm splits (raw): Tr={len(gtzan_raw_train)}, Vl={len(gtzan_raw_val)}, Ts={len(gtzan_raw_test)}. Classes: {n_gtz_classes}")

    fma_raw_train, fma_raw_val, fma_raw_test, fma_classes_list = build_fma_samples_with_official_splits(FMA_ROOT, FMA_METADATA_CSV)
    n_fma_classes = len(fma_classes_list)
    print(f"FMA Official splits (raw): Tr={len(fma_raw_train)}, Vl={len(fma_raw_val)}, Ts={len(fma_raw_test)}. Classes: {n_fma_classes}")

    run_parallel_preprocessing(gtzan_raw_train, _gtzan_all_possible_classes, PRECOMP_GTZAN_TRAIN_DIR, "GTZAN Train")
    run_parallel_preprocessing(gtzan_raw_val,   _gtzan_all_possible_classes, PRECOMP_GTZAN_VAL_DIR,   "GTZAN Val")
    run_parallel_preprocessing(gtzan_raw_test,  _gtzan_all_possible_classes, PRECOMP_GTZAN_TEST_DIR,  "GTZAN Test")
    run_parallel_preprocessing(fma_raw_train, fma_classes_list, PRECOMP_FMA_TRAIN_DIR, "FMA Train")
    run_parallel_preprocessing(fma_raw_val,   fma_classes_list, PRECOMP_FMA_VAL_DIR,   "FMA Val")
    run_parallel_preprocessing(fma_raw_test,  fma_classes_list, PRECOMP_FMA_TEST_DIR,  "FMA Test")

    def build_precomp_paths_list(precomp_split_dir, original_classes_list):
        precomp_paths = []
        for genre_idx, genre_name in enumerate(original_classes_list):
            spec_dir = os.path.join(precomp_split_dir, genre_name)
            if os.path.isdir(spec_dir):
                for fname in os.listdir(spec_dir):
                    if fname.endswith(".pt"): precomp_paths.append((os.path.join(spec_dir, fname), genre_idx))
        return precomp_paths

    gtzan_train_paths = build_precomp_paths_list(PRECOMP_GTZAN_TRAIN_DIR, _gtzan_all_possible_classes)
    gtzan_val_paths   = build_precomp_paths_list(PRECOMP_GTZAN_VAL_DIR,   _gtzan_all_possible_classes)
    gtzan_test_paths  = build_precomp_paths_list(PRECOMP_GTZAN_TEST_DIR,  _gtzan_all_possible_classes)
    print(f"Precomp GTZAN: Tr={len(gtzan_train_paths)}, Vl={len(gtzan_val_paths)}, Ts={len(gtzan_test_paths)}")

    fma_train_paths = build_precomp_paths_list(PRECOMP_FMA_TRAIN_DIR, fma_classes_list)
    fma_val_paths   = build_precomp_paths_list(PRECOMP_FMA_VAL_DIR,   fma_classes_list)
    print(f"Precomp FMA: Tr={len(fma_train_paths)}, Vl={len(fma_val_paths)}")

    if not all([gtzan_train_paths, gtzan_val_paths, gtzan_test_paths]): sys.exit("ERR: Missing GTZAN precomp splits.")
    if not all([fma_train_paths, fma_val_paths]): sys.exit("ERR: Missing FMA precomp train/val.")

    gtz_train_ds = InMemorySpectrogramDataset(gtzan_train_paths, "GTZAN Train", APPLY_SPECAUGMENT_GTZAN, CACHE_LOAD_WORKERS)
    gtz_val_ds   = InMemorySpectrogramDataset(gtzan_val_paths,   "GTZAN Val",   False, CACHE_LOAD_WORKERS)
    gtz_test_ds  = InMemorySpectrogramDataset(gtzan_test_paths,  "GTZAN Test",  False, CACHE_LOAD_WORKERS)
    gtz_train_loader = DataLoader(gtz_train_ds, GTZAN_FINETUNE_BATCH_SIZE, shuffle=True,  num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY, drop_last=True)
    gtz_val_loader   = DataLoader(gtz_val_ds,   GTZAN_FINETUNE_BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)
    gtz_test_loader  = DataLoader(gtz_test_ds,  GTZAN_FINETUNE_BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)

    fma_train_ds = InMemorySpectrogramDataset(fma_train_paths, "FMA Train", APPLY_SPECAUGMENT_FMA, CACHE_LOAD_WORKERS)
    fma_val_ds   = InMemorySpectrogramDataset(fma_val_paths,   "FMA Val",   False, CACHE_LOAD_WORKERS)
    fma_train_loader = DataLoader(fma_train_ds, FMA_PRETRAIN_BATCH_SIZE, shuffle=True,  num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY, drop_last=True)
    fma_val_loader   = DataLoader(fma_val_ds,   FMA_PRETRAIN_BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)

    class_weights_gtz = None
    if gtz_train_ds and len(gtz_train_ds) > 0:
        counts_gtz = Counter(gtz_train_ds.labels)
        if len(counts_gtz) <= n_gtz_classes:
            class_weights_list = [1.0 / counts_gtz[i] if counts_gtz.get(i,0) > 0 else 0.0 for i in range(n_gtz_classes)]
            present_classes_sum_inv = sum(w for w in class_weights_list if w > 0)
            num_present_classes = sum(1 for w in class_weights_list if w > 0)
            if present_classes_sum_inv > 0 and num_present_classes > 0:
                class_weights_list = [w * num_present_classes / present_classes_sum_inv for w in class_weights_list]
            class_weights_gtz = torch.tensor(class_weights_list, dtype=torch.float).to(DEVICE)
            print(f"GTZAN Class weights (Sturm train): {class_weights_gtz.cpu().numpy()}")
        else: print("WARN: GTZAN class count/weight issue. No weights used.")
    else: print("WARN: GTZAN train split empty. No weights used.")
    
    # FMA Pretraining
    feature_extractor = FeatExtractor().to(DEVICE)
    classifier_fma = Classifier(512, n_fma_classes, 0.5, 0.2).to(DEVICE)
    optimizer_fma = optim.AdamW(list(feature_extractor.parameters()) + list(classifier_fma.parameters()), lr=FMA_LR, weight_decay=FMA_WEIGHT_DECAY)
    scheduler_fma = optim.lr_scheduler.ReduceLROnPlateau(optimizer_fma, "max", factor=0.5, patience=FMA_SCHEDULER_PATIENCE)
    best_f1_fma, patience_fma, best_fma_epoch = 0.0, 0, 0
    history = {"fma_pretrain": [], "gtz_freeze": [], "gtz_finetune": [], "gtz_test": {}}
    model_save_name_fma = "fma_best_model_std_splits_specaug.pt"
    print(f"\n--- FMA Pretrain (SpecAugment: {APPLY_SPECAUGMENT_FMA}) ---")
    for epoch in range(1, FMA_EPOCHS + 1):
        print(f"FMA Epoch {epoch}/{FMA_EPOCHS}")
        tr_loss, tr_acc = train_epoch(fma_train_loader, feature_extractor, classifier_fma, optimizer_fma, None, DEVICE)
        vl_loss, vl_acc, vl_f1, _ = validate_epoch(fma_val_loader, feature_extractor, classifier_fma, DEVICE)
        scheduler_fma.step(vl_f1)
        print(f"  Tr Ls: {tr_loss:.4f}, Acc: {tr_acc:.4f} | Vl Ls: {vl_loss:.4f}, Acc: {vl_acc:.4f}, F1: {vl_f1:.4f}")
        history["fma_pretrain"].append({"epoch":epoch,"train_loss":tr_loss,"train_acc":tr_acc,"val_loss":vl_loss,"val_acc":vl_acc,"val_f1":vl_f1})
        if vl_f1 > best_f1_fma:
            best_f1_fma, best_fma_epoch, patience_fma = vl_f1, epoch, 0
            torch.save({"feature_extractor":feature_extractor.state_dict(),"classifier":classifier_fma.state_dict()}, model_save_name_fma)
            print(f"  New best FMA model saved (F1: {best_f1_fma:.4f})")
        else: patience_fma += 1
        if patience_fma >= FMA_EARLY_STOPPING_PATIENCE: print(f"  Early stopping FMA pretrain @ epoch {epoch}."); break
    print(f"Best FMA pretrain F1: {best_f1_fma:.4f} @ epoch {best_fma_epoch}")

    # GTZAN Fine-tuning
    print(f"\n--- Loading FMA model for GTZAN fine-tune (SpecAugment: {APPLY_SPECAUGMENT_GTZAN}) ---")
    if not os.path.exists(model_save_name_fma): sys.exit(f"ERR: {model_save_name_fma} not found.")
    checkpoint = torch.load(model_save_name_fma, map_location=DEVICE)
    feature_extractor = FeatExtractor().to(DEVICE); feature_extractor.load_state_dict(checkpoint["feature_extractor"])
    classifier_gtz = Classifier(512, n_gtz_classes, 0.5, 0.2).to(DEVICE)
    
    print("\n--- GTZAN Freeze Backbone ---")
    for p in feature_extractor.parameters(): p.requires_grad = False
    opt_gtz_fr = optim.AdamW(classifier_gtz.parameters(), lr=GTZAN_LR_FR, weight_decay=GTZAN_WEIGHT_DECAY_FR)
    for epoch in range(1, FREEZE_EPOCHS + 1):
        print(f"GTZAN Freeze Epoch {epoch}/{FREEZE_EPOCHS}")
        tr_loss, tr_acc = train_epoch(gtz_train_loader,feature_extractor,classifier_gtz,opt_gtz_fr,class_weights_gtz,DEVICE)
        vl_loss, vl_acc, vl_f1, _ = validate_epoch(gtz_val_loader,feature_extractor,classifier_gtz,DEVICE)
        print(f"  Tr Ls: {tr_loss:.4f}, Acc: {tr_acc:.4f} | Vl Ls: {vl_loss:.4f}, Acc: {vl_acc:.4f}, F1: {vl_f1:.4f}")
        history["gtz_freeze"].append({"epoch":epoch,"train_loss":tr_loss,"train_acc":tr_acc,"val_loss":vl_loss,"val_acc":vl_acc,"val_f1":vl_f1})

    print("\n--- GTZAN Full Fine-Tune ---")
    for p in feature_extractor.parameters(): p.requires_grad = True
    opt_gtz_ft = optim.AdamW(list(feature_extractor.parameters())+list(classifier_gtz.parameters()),lr=GTZAN_LR_FT,weight_decay=GTZAN_WEIGHT_DECAY_FT)
    sched_gtz_ft = optim.lr_scheduler.ReduceLROnPlateau(opt_gtz_ft,"max",factor=0.5,patience=10)
    best_f1_gtz,patience_gtz,best_gtz_epoch = 0.0,0,0
    model_save_name_gtz = "gtzan_best_model_std_splits_specaug.pt"
    for epoch in range(1, GTZ_EPOCHS + 1):
        print(f"GTZAN Fine-Tune Epoch {epoch}/{GTZ_EPOCHS}")
        tr_loss,tr_acc = train_epoch(gtz_train_loader,feature_extractor,classifier_gtz,opt_gtz_ft,class_weights_gtz,DEVICE)
        vl_loss,vl_acc,vl_f1,cm = validate_epoch(gtz_val_loader,feature_extractor,classifier_gtz,DEVICE)
        sched_gtz_ft.step(vl_f1)
        print(f"  Tr Ls: {tr_loss:.4f}, Acc: {tr_acc:.4f} | Vl Ls: {vl_loss:.4f}, Acc: {vl_acc:.4f}, F1: {vl_f1:.4f}")
        history["gtz_finetune"].append({"epoch":epoch,"train_loss":tr_loss,"train_acc":tr_acc,"val_loss":vl_loss,"val_acc":vl_acc,"val_f1":vl_f1,"confusion_matrix":cm.tolist() if cm is not None else None})
        if vl_f1 > best_f1_gtz:
            best_f1_gtz,best_gtz_epoch,patience_gtz = vl_f1,epoch,0
            torch.save({"feature_extractor":feature_extractor.state_dict(),"classifier":classifier_gtz.state_dict()},model_save_name_gtz)
            print(f"  New best GTZAN model saved (F1: {best_f1_gtz:.4f})")
        else: patience_gtz +=1
        if patience_gtz >=15: print(f"  Early stopping GTZAN fine-tune @ epoch {epoch}."); break
    print(f"Best GTZAN fine-tune F1: {best_f1_gtz:.4f} @ epoch {best_gtz_epoch}")

    print("\n--- Evaluating on GTZAN Test Set ---")
    if os.path.exists(model_save_name_gtz):
        ckpt_test = torch.load(model_save_name_gtz,map_location=DEVICE)
        feat_ext_test = FeatExtractor().to(DEVICE); clf_gtz_test = Classifier(512,n_gtz_classes).to(DEVICE)
        feat_ext_test.load_state_dict(ckpt_test["feature_extractor"])
        clf_gtz_test.load_state_dict(ckpt_test["classifier"])
        ts_loss,ts_acc,ts_f1,ts_cm = validate_epoch(gtz_test_loader,feat_ext_test,clf_gtz_test,DEVICE)
        print(f"\nGTZAN Test (Sturm,SpecAug) – Ls: {ts_loss:.4f}, Acc: {ts_acc:.4f}, F1: {ts_f1:.4f}")
        if ts_cm is not None: print("GTZAN Test CM:\n",ts_cm)
        history["gtz_test"] = {"test_loss":ts_loss,"test_acc":ts_acc,"test_f1":ts_f1,"confusion_matrix":ts_cm.tolist() if ts_cm is not None else None}
    else: print(f"WARN: {model_save_name_gtz} not found. Skipping test eval.")

    print("\n--- Saving metrics to metrics_std_splits_specaug.json ---") 
    with open("metrics_std_splits_specaug.json","w") as jf: json.dump(history,jf,indent=2)
    print("\nScript finished.")

if __name__ == "__main__":
    try:
        mp.set_start_method('spawn', force=True)
        print("Multiprocessing start method set to 'spawn'.")
    except RuntimeError as e:
        print(f"Warning: Could not set multiprocessing start method to 'spawn': {e}")
        current_method = mp.get_start_method(allow_none=True)
        if current_method is None:
            try:
                mp.set_start_method('forkserver', force=True)
                print("Multiprocessing start method set to 'forkserver' as fallback.")
            except RuntimeError as e2:
                print(f"Warning: Could not set start method to 'forkserver': {e2}")
                if mp.get_start_method(allow_none=True) is None: 
                    mp.set_start_method('fork', force=True) 
                    print("Multiprocessing start method set to 'fork' as last resort.")
        else:
            print(f"Current start method already set to: {current_method}")
    main()