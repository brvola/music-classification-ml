# FILE: test_all_model_robustness.py (Replaces test_baseline_robustness.py)

import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torchaudio.transforms as T
from torch.utils.data import Dataset, DataLoader
from torchvision import models
from sklearn.metrics import f1_score, accuracy_score
from tqdm import tqdm
import json
import joblib

# --- Define Classes (Must match main.py) ---
class FeatExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        resnet = models.resnet18(weights=None) 
        self.feature = nn.Sequential(*list(resnet.children())[:-1])
    def forward(self, x):
        x = self.feature(x); return x.view(x.size(0), -1)

# ### <<< ADD CLASSIFIER CLASS FROM main.py
class Classifier(nn.Module):
    def __init__(self, in_dim, n_cls, dropout1=0.5, dropout2=0.2): # Match params in main.py
        super().__init__()
        self.dropout1  = nn.Dropout(dropout1); self.fc1 = nn.Linear(in_dim, 256)
        self.relu = nn.ReLU(); self.dropout2 = nn.Dropout(dropout2)
        self.fc2 = nn.Linear(256, n_cls)
    def forward(self, x):
        x = self.dropout1(x); x = self.fc1(x); x = self.relu(x)
        x = self.dropout2(x); x = self.fc2(x); return x

class PrecomputedPTDataset(Dataset):
    def __init__(self, file_paths_with_labels, dataset_name="Dataset"):
        self.file_list = file_paths_with_labels
        self.dataset_name = dataset_name
    def __len__(self): return len(self.file_list)
    def __getitem__(self, idx):
        spec_path, label = self.file_list[idx]
        return torch.load(spec_path, map_location='cpu'), label

class CorruptedOnTheFlyDataset(Dataset):
    """Wraps a dataset to apply corruption to the spectrogram in __getitem__."""
    def __init__(self, original_dataset, corruption_type, intensity):
        self.original_dataset = original_dataset
        self.corruption_type = corruption_type
        self.intensity = intensity
        
        if self.corruption_type in ['freq_mask', 'time_mask']:
            self.freq_masking = T.FrequencyMasking(freq_mask_param=int(intensity))
            self.time_masking = T.TimeMasking(time_mask_param=int(intensity))

    def __len__(self): return len(self.original_dataset)
    def __getitem__(self, idx):
        spec, label = self.original_dataset[idx]
        spec_corrupted = spec.clone()

        if self.corruption_type == 'noise':
            noise = torch.randn_like(spec_corrupted) * self.intensity
            spec_corrupted += noise
        elif self.corruption_type == 'freq_mask':
            spec_corrupted = self.freq_masking(spec_corrupted.unsqueeze(0)).squeeze(0)
        elif self.corruption_type == 'time_mask':
            spec_corrupted = self.time_masking(spec_corrupted.unsqueeze(0)).squeeze(0)

        return spec_corrupted, label

# --- CONFIGURATION (Sync with main.py) ---
BEST_DL_RUN_RESULTS_DIR = "./"
GTZAN_MODEL_FILENAME = "gtzan_best_model_std_splits_specaug.pt" # ### <<< ADD GTZAN MODEL
FMA_MODEL_FILENAME = "fma_best_model_std_splits_specaug.pt" 
PRECOMP_DIR_FOR_BASELINES = "precomputed_std_splits_specaug"
PRECOMP_GTZAN_TEST_DIR  = os.path.join(PRECOMP_DIR_FOR_BASELINES, "gtzan_test")
GTZAN_CLASSES = ['blues', 'classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock']
N_GTZAN_CLASSES = len(GTZAN_CLASSES) # ### <<< ADD
BASELINE_RESULTS_SUBDIR = "baseline_comparisons"
SAVED_MODELS_DIR = os.path.join(BASELINE_RESULTS_SUBDIR, "saved_models")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 64 # ### <<< BATCH SIZE FOR DL MODEL
SEED = 805

# --- Helper functions ---
def build_precomp_paths_list(precomp_split_dir, original_classes_list):
    precomp_paths = []
    for genre_idx, genre_name in enumerate(original_classes_list):
        spec_dir = os.path.join(precomp_split_dir, genre_name)
        if os.path.isdir(spec_dir):
            for fname in os.listdir(spec_dir):
                if fname.endswith(".pt"): 
                    precomp_paths.append((os.path.join(spec_dir, fname), genre_idx))
    return precomp_paths

def extract_features_for_baselines(dataset, feature_extractor_model, device, batch_size):
    feature_extractor_model.eval() 
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0) 
    all_features_list, all_labels_list = [], []
    with torch.no_grad():
        for specs, labels in tqdm(dataloader, desc="Feature Extraction (Baselines)", ncols=100, leave=False):
            specs = specs.to(device)
            features = feature_extractor_model(specs)
            all_features_list.append(features.cpu().numpy())
            all_labels_list.append(labels.numpy())
    return np.concatenate(all_features_list, axis=0), np.concatenate(all_labels_list, axis=0)

# ### <<< NEW FUNCTION TO EVALUATE THE DEEP LEARNING MODEL
def evaluate_deep_model(dataset, feature_extractor, classifier, device, batch_size):
    feature_extractor.eval(); classifier.eval()
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    all_preds, all_gts = [], []
    with torch.no_grad():
        for specs, labels in tqdm(dataloader, desc="DL Model Evaluation", ncols=100, leave=False):
            specs = specs.to(device)
            features = feature_extractor(specs)
            outputs = classifier(features)
            preds = outputs.argmax(dim=1)
            all_preds.append(preds.cpu().numpy())
            all_gts.append(labels.numpy())

    all_preds = np.concatenate(all_preds)
    all_gts = np.concatenate(all_gts)
    
    f1 = f1_score(all_gts, all_preds, average='macro', zero_division=0)
    acc = accuracy_score(all_gts, all_preds)
    return f1, acc

def main():
    torch.manual_seed(SEED); np.random.seed(SEED)
    print(f"Using device: {DEVICE}")

    # 1. LOAD ALL MODELS (DEEP LEARNING + BASELINES)
    # --- Load the final fine-tuned GTZAN model ---
    gtzan_model_path = os.path.join(BEST_DL_RUN_RESULTS_DIR, GTZAN_MODEL_FILENAME)
    if not os.path.exists(gtzan_model_path):
        print(f"ERROR: Fine-tuned GTZAN model not found at {gtzan_model_path}. Run main.py first."); sys.exit(1)
    
    gtzan_checkpoint = torch.load(gtzan_model_path, map_location=DEVICE)
    dl_feature_extractor = FeatExtractor().to(DEVICE)
    dl_classifier = Classifier(512, N_GTZAN_CLASSES).to(DEVICE)
    dl_feature_extractor.load_state_dict(gtzan_checkpoint["feature_extractor"])
    dl_classifier.load_state_dict(gtzan_checkpoint["classifier"])
    print("Loaded fine-tuned GTZAN Deep Learning model.")

    # --- Load the feature extractor for baselines (from FMA pre-training) ---
    fma_model_path = os.path.join(BEST_DL_RUN_RESULTS_DIR, FMA_MODEL_FILENAME)
    if not os.path.exists(fma_model_path):
        print(f"ERROR: FMA model for baselines not found at {fma_model_path}."); sys.exit(1)
    fma_checkpoint = torch.load(fma_model_path, map_location=DEVICE)
    baseline_feature_extractor = FeatExtractor().to(DEVICE)
    baseline_feature_extractor.load_state_dict(fma_checkpoint["feature_extractor"])
    print("Loaded FMA feature extractor for baselines.")

    # --- Load the saved scikit-learn baseline models ---
    full_models_dir = os.path.join(BEST_DL_RUN_RESULTS_DIR, SAVED_MODELS_DIR)
    model_paths = [os.path.join(full_models_dir, f) for f in os.listdir(full_models_dir) if f.endswith('.joblib') and 'Sturm' in f]
    if not model_paths:
        print(f"ERROR: No saved baseline models found in {full_models_dir}. Run baseline.py first."); sys.exit(1)
    
    loaded_baseline_models = {}
    for path in model_paths:
        model_name = os.path.basename(path).replace('.joblib', '')
        loaded_baseline_models[model_name] = joblib.load(path)
    print(f"Loaded {len(loaded_baseline_models)} baseline models.")

    # 2. LOAD THE CLEAN TEST DATA
    gtzan_test_paths = build_precomp_paths_list(PRECOMP_GTZAN_TEST_DIR, GTZAN_CLASSES)
    if not gtzan_test_paths:
        print(f"ERROR: No precomputed test files found in {PRECOMP_GTZAN_TEST_DIR}"); sys.exit(1)
    clean_test_dataset = PrecomputedPTDataset(gtzan_test_paths, "GTZAN Clean Test")

    # 3. SETUP FOR ROBUSTNESS TESTING
    all_model_names = ["Fine-Tuned DL Model"] + list(loaded_baseline_models.keys())
    robustness_results = {model_name: {} for model_name in all_model_names}
    
    corruption_tests = {
        'additive_noise': [0.0, 0.1, 0.25, 0.5, 0.75, 1.0],
        'time_mask': [0, 50, 100, 150, 200, 250],
        'freq_mask': [0, 20, 40, 60, 80, 100]
    }

    # 4. RUN THE ROBUSTNESS EVALUATION LOOP
    for corruption_type, levels in corruption_tests.items():
        print(f"\n--- TESTING ROBUSTNESS TO: {corruption_type.upper()} ---")
        for level in levels:
            print(f"  Level: {level}")
            
            # Create a new dataset with the current corruption level
            corrupted_dataset = CorruptedOnTheFlyDataset(clean_test_dataset, corruption_type, level)

            # --- Evaluate the Fine-Tuned Deep Learning Model ---
            f1_dl, acc_dl = evaluate_deep_model(
                corrupted_dataset, dl_feature_extractor, dl_classifier, DEVICE, BATCH_SIZE
            )
            if corruption_type not in robustness_results["Fine-Tuned DL Model"]:
                robustness_results["Fine-Tuned DL Model"][corruption_type] = []
            robustness_results["Fine-Tuned DL Model"][corruption_type].append({'level': level, 'f1': f1_dl, 'acc': acc_dl})
            print(f"    - Fine-Tuned DL Model: F1={f1_dl:.4f}")

            # --- Evaluate the Baseline Models ---
            # Extract features from the corrupted data just once for all baselines
            X_test_corrupted, y_test_corrupted = extract_features_for_baselines(
                corrupted_dataset, baseline_feature_extractor, DEVICE, BATCH_SIZE
            )
            for model_name, model in loaded_baseline_models.items():
                y_pred = model.predict(X_test_corrupted)
                f1_base = f1_score(y_test_corrupted, y_pred, average='macro', zero_division=0)
                acc_base = accuracy_score(y_test_corrupted, y_pred)
                
                if corruption_type not in robustness_results[model_name]:
                    robustness_results[model_name][corruption_type] = []
                robustness_results[model_name][corruption_type].append({'level': level, 'f1': f1_base, 'acc': acc_base})
                print(f"    - {model_name}: F1={f1_base:.4f}")

    # 5. SAVE THE COMBINED RESULTS
    output_dir = os.path.join(BEST_DL_RUN_RESULTS_DIR, BASELINE_RESULTS_SUBDIR)
    results_path = os.path.join(output_dir, "all_models_robustness_results.json")
    with open(results_path, 'w') as f:
        json.dump(robustness_results, f, indent=2)
    print(f"\nComprehensive robustness results for all models saved to: {results_path}")

if __name__ == "__main__":
    main()