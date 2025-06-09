import torch.multiprocessing as mp
import os
import sys
import contextlib
import json
import numpy as np
import pandas as pd
import librosa
import torch
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
import time
from data_prep.downloader import ensure_gtzan_splits, verify_kaggle_dataset
# Import individual variables from the config file
from config import (
    GTZAN_ROOT, FMA_ROOT, FMA_METADATA,
    PRECOMP_GTZAN_TRAIN_DIR, PRECOMP_GTZAN_VAL_DIR, PRECOMP_GTZAN_TEST_DIR,
    PRECOMP_FMA_TRAIN_DIR, PRECOMP_FMA_VAL_DIR, PRECOMP_FMA_TEST_DIR,
    DEVICE, NUM_WORKERS, CACHE_LOAD_WORKERS, PIN_MEMORY,
    APPLY_SPECAUGMENT_FMA, APPLY_SPECAUGMENT_GTZAN,
    FMA_PRETRAIN_BATCH_SIZE, FMA_EPOCHS, FMA_WEIGHT_DECAY, FMA_LR,
    FMA_SCHEDULER_PATIENCE, FMA_EARLY_STOPPING_PATIENCE,
    FREEZE_EPOCHS, GTZAN_FINETUNE_BATCH_SIZE, GTZAN_EPOCHS,
    GTZAN_WEIGHT_DECAY_FR, GTZAN_LR_FR,
    GTZAN_WEIGHT_DECAY_FT, GTZAN_LR_FT,
    GTZAN_SPECAUG_FREQ_MASK_PARAM, GTZAN_SPECAUG_TIME_MASK_PARAM,
    SPECAUGMENT_FREQ_MASK_PARAM, SPECAUGMENT_TIME_MASK_PARAM
)
from data_prep.build_splits import build_gtzan_samples_from_sturm_splits, build_fma_samples_with_official_splits
from data_prep.preprocess import run_parallel_preprocessing
from data_prep.dataset import InMemorySpectrogramDataset
from model.classifier import Classifier, FeatExtractor, train_epoch, validate_epoch
from utils import set_seed, seed_worker

# Commented out for now since we already have the precomputed spectrograms

def main(device):
    set_seed(805)
    ensure_gtzan_splits()
    if not verify_kaggle_dataset():
        print("ERR: Kaggle dataset not found or incomplete. Please download it from Kaggle and place it into /data.")
        sys.exit(1)
    print_model_info(device)

    # The original code was trying to find genre folders in the raw audio directory (GTZAN_ROOT).
    # Since we are using precomputed data, we should get the class list directly from the
    # precomputed training directory's subfolders.
    # _gtzan_all_possible_classes = sorted([d for d in os.listdir(PRECOMP_GTZAN_TRAIN_DIR) if os.path.isdir(os.path.join(GTZAN_ROOT, d))])
    if not os.path.exists(PRECOMP_GTZAN_TRAIN_DIR) or not os.listdir(PRECOMP_GTZAN_TRAIN_DIR):
        sys.exit(f"ERR: Precomputed GTZAN training directory not found or is empty: {PRECOMP_GTZAN_TRAIN_DIR}")
    _gtzan_all_possible_classes = sorted([d for d in os.listdir(PRECOMP_GTZAN_TRAIN_DIR) if os.path.isdir(os.path.join(PRECOMP_GTZAN_TRAIN_DIR, d))])
    
    n_gtz_classes = len(_gtzan_all_possible_classes)

    # The following lines are correctly commented out as they deal with raw audio files.
    # gtzan_raw_train, gtzan_raw_val, gtzan_raw_test = build_gtzan_samples_from_sturm_splits(GTZAN_ROOT, _gtzan_all_possible_classes)
    # print(f"GTZAN Sturm splits (raw): Tr={len(gtzan_raw_train)}, Vl={len(gtzan_raw_val)}, Ts={len(gtzan_raw_test)}. Classes: {n_gtz_classes}")
    # fma_raw_train, fma_raw_val, fma_raw_test, fma_classes_list = build_fma_samples_with_official_splits(FMA_ROOT, FMA_METADATA)
    # n_fma_classes = len(fma_classes_list)
    # print(f"FMA Official splits (raw): Tr={len(fma_raw_train)}, Vl={len(fma_raw_val)}, Ts={len(fma_raw_test)}. Classes: {n_fma_classes}")

    # The preprocessing steps are also correctly commented out.
    # run_parallel_preprocessing(gtzan_raw_train, _gtzan_all_possible_classes, PRECOMP_GTZAN_TRAIN_DIR, "GTZAN Train")
    # run_parallel_preprocessing(gtzan_raw_val,   _gtzan_all_possible_classes, PRECOMP_GTZAN_VAL_DIR,   "GTZAN Val")
    # run_parallel_preprocessing(gtzan_raw_test,  _gtzan_all_possible_classes, PRECOMP_GTZAN_TEST_DIR,  "GTZAN Test")
    # run_parallel_preprocessing(fma_raw_train, fma_classes_list, PRECOMP_FMA_TRAIN_DIR, "FMA Train")
    # run_parallel_preprocessing(fma_raw_val,   fma_classes_list, PRECOMP_FMA_VAL_DIR,   "FMA Val")
    # run_parallel_preprocessing(fma_raw_test,  fma_classes_list, PRECOMP_FMA_TEST_DIR,  "FMA Test")

    # Now we build the file paths from the precomputed directories. This will now work correctly.
    gtzan_train_paths = build_precomp_paths_list(PRECOMP_GTZAN_TRAIN_DIR, _gtzan_all_possible_classes)
    gtzan_val_paths   = build_precomp_paths_list(PRECOMP_GTZAN_VAL_DIR,   _gtzan_all_possible_classes)
    gtzan_test_paths  = build_precomp_paths_list(PRECOMP_GTZAN_TEST_DIR,  _gtzan_all_possible_classes)
    print(f"Precomp GTZAN: Tr={len(gtzan_train_paths)}, Vl={len(gtzan_val_paths)}, Ts={len(gtzan_test_paths)}")

    # Same fix as for GTZAN: get the FMA class list from the precomputed directory.
    # fma_classes_list = sorted([d for d in os.listdir(PRECOMP_FMA_TRAIN_DIR) if os.path.isdir(os.path.join(FMA_ROOT, d))])
    if not os.path.exists(PRECOMP_FMA_TRAIN_DIR) or not os.listdir(PRECOMP_FMA_TRAIN_DIR):
        sys.exit(f"ERR: Precomputed FMA training directory not found or is empty: {PRECOMP_FMA_TRAIN_DIR}")
    fma_classes_list = sorted([d for d in os.listdir(PRECOMP_FMA_TRAIN_DIR) if os.path.isdir(os.path.join(PRECOMP_FMA_TRAIN_DIR, d))])
    n_fma_classes = len(fma_classes_list)
    
    fma_train_paths = build_precomp_paths_list(PRECOMP_FMA_TRAIN_DIR, fma_classes_list)
    fma_val_paths   = build_precomp_paths_list(PRECOMP_FMA_VAL_DIR,   fma_classes_list)
    print(f"Precomp FMA: Tr={len(fma_train_paths)}, Vl={len(fma_val_paths)}. Classes: {n_fma_classes}")

    if not all([gtzan_train_paths, gtzan_val_paths, gtzan_test_paths]): sys.exit("ERR: Missing GTZAN precomp splits.")
    if not all([fma_train_paths, fma_val_paths]): sys.exit("ERR: Missing FMA precomp train/val.")

    gtz_train_ds = InMemorySpectrogramDataset(gtzan_train_paths, "GTZAN Train", APPLY_SPECAUGMENT_GTZAN, CACHE_LOAD_WORKERS, freq_mask_param=GTZAN_SPECAUG_FREQ_MASK_PARAM, time_mask_param=GTZAN_SPECAUG_TIME_MASK_PARAM)
    gtz_val_ds   = InMemorySpectrogramDataset(gtzan_val_paths,   "GTZAN Val",   False, CACHE_LOAD_WORKERS)
    gtz_test_ds  = InMemorySpectrogramDataset(gtzan_test_paths,  "GTZAN Test",  False, CACHE_LOAD_WORKERS)
    gtz_train_loader = DataLoader(gtz_train_ds, GTZAN_FINETUNE_BATCH_SIZE, shuffle=True,  num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY, drop_last=True, worker_init_fn=seed_worker)
    gtz_val_loader   = DataLoader(gtz_val_ds,   GTZAN_FINETUNE_BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY, worker_init_fn=seed_worker)
    gtz_test_loader  = DataLoader(gtz_test_ds,  GTZAN_FINETUNE_BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY, worker_init_fn=seed_worker)

    fma_train_ds = InMemorySpectrogramDataset(fma_train_paths, "FMA Train", APPLY_SPECAUGMENT_FMA, CACHE_LOAD_WORKERS, freq_mask_param=SPECAUGMENT_FREQ_MASK_PARAM, time_mask_param=SPECAUGMENT_TIME_MASK_PARAM)
    fma_val_ds   = InMemorySpectrogramDataset(fma_val_paths,   "FMA Val",   False, CACHE_LOAD_WORKERS)
    fma_train_loader = DataLoader(fma_train_ds, FMA_PRETRAIN_BATCH_SIZE, shuffle=True,  num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY, drop_last=True, worker_init_fn=seed_worker)
    fma_val_loader   = DataLoader(fma_val_ds,   FMA_PRETRAIN_BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY, worker_init_fn=seed_worker)

    class_weights_gtz = None
    if gtz_train_ds and len(gtz_train_ds) > 0:
        counts_gtz = Counter(gtz_train_ds.labels)
        if len(counts_gtz) <= n_gtz_classes:
            class_weights_list = [1.0 / counts_gtz[i] if counts_gtz.get(i,0) > 0 else 0.0 for i in range(n_gtz_classes)]
            present_classes_sum_inv = sum(w for w in class_weights_list if w > 0)
            num_present_classes = sum(1 for w in class_weights_list if w > 0)
            if present_classes_sum_inv > 0 and num_present_classes > 0:
                class_weights_list = [w * num_present_classes / present_classes_sum_inv for w in class_weights_list]
            class_weights_gtz = torch.tensor(class_weights_list, dtype=torch.float).to(device)
            print(f"GTZAN Class weights (Sturm train): {class_weights_gtz.cpu().numpy()}")
        else: print("WARN: GTZAN class count/weight issue. No weights used.")
    else: print("WARN: GTZAN train split empty. No weights used.")
    
    start = time.time()

    # FMA Pretraining
    feature_extractor = FeatExtractor().to(device)
    classifier_fma = Classifier(512, n_fma_classes, 0.4, 0.4).to(device)
    optimizer_fma = optim.AdamW(list(feature_extractor.parameters()) + list(classifier_fma.parameters()), lr=FMA_LR, weight_decay=FMA_WEIGHT_DECAY)
    scheduler_fma = optim.lr_scheduler.ReduceLROnPlateau(optimizer_fma, "max", factor=0.5, patience=FMA_SCHEDULER_PATIENCE)
    best_f1_fma, patience_fma, best_fma_epoch = 0.0, 0, 0
    history = {"fma_pretrain": [], "gtz_freeze": [], "gtz_finetune": [], "gtz_test": {}}
    model_save_name_fma = "fma_model.pt"
    print(f'\n--- FMA Pretrain (SpecAugment: {APPLY_SPECAUGMENT_FMA}) ---')
    for epoch in range(1, FMA_EPOCHS + 1):
        print(f'FMA Epoch {epoch}/{FMA_EPOCHS}')
        tr_loss, tr_acc = train_epoch(fma_train_loader, feature_extractor, classifier_fma, optimizer_fma, None, device)
        vl_loss, vl_acc, vl_f1, _ = validate_epoch(fma_val_loader, feature_extractor, classifier_fma, device)
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
    checkpoint = torch.load(model_save_name_fma, map_location=device)
    feature_extractor = FeatExtractor().to(device); feature_extractor.load_state_dict(checkpoint["feature_extractor"])
    classifier_gtz = Classifier(512, n_gtz_classes, 0.2, 0.6).to(device)
    
    print("\n--- GTZAN Freeze Backbone ---")
    for p in feature_extractor.parameters(): p.requires_grad = False
    opt_gtz_fr = optim.AdamW(classifier_gtz.parameters(), lr=GTZAN_LR_FR, weight_decay=GTZAN_WEIGHT_DECAY_FR)
    for epoch in range(1, FREEZE_EPOCHS + 1):
        print(f"GTZAN Freeze Epoch {epoch}/{FREEZE_EPOCHS}")
        tr_loss, tr_acc = train_epoch(gtz_train_loader,feature_extractor,classifier_gtz,opt_gtz_fr,class_weights_gtz,device)
        vl_loss, vl_acc, vl_f1, _ = validate_epoch(gtz_val_loader,feature_extractor,classifier_gtz,device)
        print(f"  Tr Ls: {tr_loss:.4f}, Acc: {tr_acc:.4f} | Vl Ls: {vl_loss:.4f}, Acc: {vl_acc:.4f}, F1: {vl_f1:.4f}")
        history["gtz_freeze"].append({"epoch":epoch,"train_loss":tr_loss,"train_acc":tr_acc,"val_loss":vl_loss,"val_acc":vl_acc,"val_f1":vl_f1})

    print("\n--- GTZAN Full Fine-Tune ---")
    for p in feature_extractor.parameters(): p.requires_grad = True
    opt_gtz_ft = optim.AdamW(list(feature_extractor.parameters())+list(classifier_gtz.parameters()),lr=GTZAN_LR_FT,weight_decay=GTZAN_WEIGHT_DECAY_FT)
    sched_gtz_ft = optim.lr_scheduler.ReduceLROnPlateau(opt_gtz_ft,"max",factor=0.5,patience=10)
    best_f1_gtz,patience_gtz,best_gtz_epoch = 0.0,0,0
    model_save_name_gtz = "best_model.pt"
    for epoch in range(1, GTZAN_EPOCHS + 1):
        print(f"GTZAN Fine-Tune Epoch {epoch}/{GTZAN_EPOCHS}")
        tr_loss,tr_acc = train_epoch(gtz_train_loader,feature_extractor,classifier_gtz,opt_gtz_ft,class_weights_gtz,device)
        vl_loss,vl_acc,vl_f1,cm = validate_epoch(gtz_val_loader,feature_extractor,classifier_gtz,device)
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
        ckpt_test = torch.load(model_save_name_gtz,map_location=device)
        feat_ext_test = FeatExtractor().to(device); clf_gtz_test = Classifier(512,n_gtz_classes).to(device)
        feat_ext_test.load_state_dict(ckpt_test["feature_extractor"])
        clf_gtz_test.load_state_dict(ckpt_test["classifier"])
        ts_loss,ts_acc,ts_f1,ts_cm = validate_epoch(gtz_test_loader,feat_ext_test,clf_gtz_test,device)
        print(f"\nGTZAN Test (Sturm,SpecAug) – Ls: {ts_loss:.4f}, Acc: {ts_acc:.4f}, F1: {ts_f1:.4f}")
        if ts_cm is not None: print("GTZAN Test CM:\n",ts_cm)
        history["gtz_test"] = {"test_loss":ts_loss,"test_acc":ts_acc,"test_f1":ts_f1,"confusion_matrix":ts_cm.tolist() if ts_cm is not None else None}
    else: print(f"WARN: {model_save_name_gtz} not found. Skipping test eval.")

    end = time.time()

    print("\n--- Saving metrics to metrics.json ---") 
    with open("metrics.json","w") as jf: json.dump(history,jf,indent=2)
    print("\nScript finished.")

    print(f"Total runtime: {end - start:.2f} seconds")

def print_model_info(device):
    print(f"Using device: {device}")
    print(f"DataLoader settings: num_workers={NUM_WORKERS}, pin_memory={PIN_MEMORY}")
    print(f"Initial RAM Caching workers: {CACHE_LOAD_WORKERS if CACHE_LOAD_WORKERS > 0 else 'Sequential'}")
    print(f"SpecAugment: FMA={APPLY_SPECAUGMENT_FMA}, GTZAN={APPLY_SPECAUGMENT_GTZAN}")

def build_precomp_paths_list(precomp_split_dir, original_classes_list):
    precomp_paths = []
    for genre_idx, genre_name in enumerate(original_classes_list):
        spec_dir = os.path.join(precomp_split_dir, genre_name)
        if os.path.isdir(spec_dir):
            for fname in os.listdir(spec_dir):
                if fname.endswith(".pt"): precomp_paths.append((os.path.join(spec_dir, fname), genre_idx))
    return precomp_paths

if __name__ == "__main__":
    # The DEVICE string ("cuda" or "cpu") is imported from the config file.
    # We create the actual torch.device object here.
    device = torch.device(DEVICE)

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
            
    # Pass the torch.device object to the main function
    main(device)