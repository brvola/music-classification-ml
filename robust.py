# FILE: test_advanced_model_robustness.py
# PURPOSE: To evaluate the robustness of the final, fine-tuned Advanced CNN model.
#          This script tests the model against various on-the-fly audio corruptions
#          on the standardized Sturm test set.

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

# --- PyTorch Class Definitions (for your Advanced CNN) ---
class FeatExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        resnet = models.resnet18(weights=None)
        self.feature = nn.Sequential(*list(resnet.children())[:-1])
    def forward(self, x):
        return self.feature(x).view(x.size(0), -1)

class Classifier(nn.Module):
    def __init__(self, in_dim, n_cls, dropout1=0.5, dropout2=0.2):
        super().__init__()
        self.drop1 = nn.Dropout(dropout1)
        self.fc1   = nn.Linear(in_dim, 256)
        self.relu  = nn.ReLU()
        self.drop2 = nn.Dropout(dropout2)
        self.fc2   = nn.Linear(256, n_cls)
    def forward(self, x):
        x = self.drop1(x); x = self.fc1(x); x = self.relu(x); x = self.drop2(x); return self.fc2(x)

# --- Dataset Definitions ---
class PrecomputedPTDataset(Dataset):
    def __init__(self, file_paths_with_labels):
        self.file_list = file_paths_with_labels
    def __len__(self): return len(self.file_list)
    def __getitem__(self, idx):
        spec_path, label = self.file_list[idx]
        return torch.load(spec_path, map_location='cpu'), label

class CorruptedOnTheFlyDataset(Dataset):
    """Wraps a dataset to apply corruption to the spectrogram."""
    def __init__(self, original_dataset, corruption_type, intensity):
        self.original_dataset = original_dataset
        self.corruption_type = corruption_type
        self.intensity = float(intensity)
        self.freq_masking = T.FrequencyMasking(freq_mask_param=int(self.intensity))
        self.time_masking = T.TimeMasking(time_mask_param=int(self.intensity))
    def __len__(self): return len(self.original_dataset)
    def __getitem__(self, idx):
        spec, label = self.original_dataset[idx]
        spec_corrupted = spec.clone()
        if self.corruption_type == 'noise':
            spec_corrupted.add_(torch.randn_like(spec_corrupted) * self.intensity)
        elif self.corruption_type == 'freq_mask':
            spec_corrupted = self.freq_masking(spec_corrupted)
        elif self.corruption_type == 'time_mask':
            spec_corrupted = self.time_masking(spec_corrupted)
        return spec_corrupted, label

# --- CONFIGURATION ---
RESULTS_DIR = "./"
ADVANCED_CNN_FILENAME = "gtzan_best_model_std_splits_specaug.pt"
PRECOMP_DIR = "precomputed_std_splits_specaug"
PRECOMP_GTZAN_TEST_DIR  = os.path.join(PRECOMP_DIR, "gtzan_test")
GTZAN_CLASSES = ['blues', 'classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock']
N_GTZAN_CLASSES = len(GTZAN_CLASSES)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 32
SEED = 805

# --- Helper & Evaluation Functions ---
def build_precomp_paths_list(split_dir, classes_list):
    paths = []
    for i, g in enumerate(classes_list):
        d = os.path.join(split_dir, g)
        if os.path.isdir(d):
            for fn in os.listdir(d):
                if fn.endswith(".pt"): paths.append((os.path.join(d, fn), i))
    return paths

def evaluate_advanced_cnn(dataset, feat_ext, clf, device, batch_size):
    feat_ext.eval(); clf.eval()
    loader = DataLoader(dataset, batch_size, shuffle=False, num_workers=0)
    all_p, all_g = [], []
    with torch.no_grad():
        for specs, labels in tqdm(loader, desc="Evaluating Model", ncols=100, leave=False):
            specs = specs.to(device)
            p = clf(feat_ext(specs)).argmax(dim=1).cpu().numpy()
            all_p.append(p); all_g.append(labels.numpy())
    p = np.concatenate(all_p); g = np.concatenate(all_g)
    return f1_score(g, p, average='macro', zero_division=0), accuracy_score(g, p)

# --- MAIN SCRIPT ---
def main():
    torch.manual_seed(SEED); np.random.seed(SEED)
    print(f"Using device: {DEVICE}")

    # 1. Load your Advanced CNN model
    adv_cnn_path = os.path.join(RESULTS_DIR, ADVANCED_CNN_FILENAME)
    if not os.path.exists(adv_cnn_path):
        print(f"ERROR: {ADVANCED_CNN_FILENAME} not found. Please run main.py first."); sys.exit(1)
    
    checkpoint = torch.load(adv_cnn_path, map_location=DEVICE)
    adv_feat_extractor = FeatExtractor().to(DEVICE)
    adv_classifier = Classifier(512, N_GTZAN_CLASSES).to(DEVICE)
    adv_feat_extractor.load_state_dict(checkpoint["feature_extractor"])
    adv_classifier.load_state_dict(checkpoint["classifier"])
    print("Loaded Advanced CNN (PyTorch) model.")

    # 2. Prepare the clean test dataset from your pre-computed files
    adv_test_paths = build_precomp_paths_list(PRECOMP_GTZAN_TEST_DIR, GTZAN_CLASSES)
    if not adv_test_paths:
        print(f"ERROR: No precomputed test files found in {PRECOMP_GTZAN_TEST_DIR}"); sys.exit(1)
    clean_test_dataset = PrecomputedPTDataset(adv_test_paths)
    print(f"Loaded {len(clean_test_dataset)} test spectrograms for evaluation.")

    # 3. Define corruption tests and run the evaluation sweep
    corruption_tests = {
        'additive_noise': [0.0, 0.25, 0.5, 0.75, 1.0, 1.25],
        'time_mask':       [0, 50, 100, 150, 200, 250],
        'freq_mask':       [0, 20, 40, 60, 80, 100],
    }
    
    results = {"Advanced CNN (Ours)": {}}

    for corruption_type, levels in corruption_tests.items():
        print(f"\n--- TESTING ROBUSTNESS TO: {corruption_type.upper()} ---")
        results["Advanced CNN (Ours)"][corruption_type] = []

        for level in levels:
            print(f"  Corruption Level: {level}")
            corrupted_dataset = CorruptedOnTheFlyDataset(clean_test_dataset, corruption_type, level)
            
            f1_adv, acc_adv = evaluate_advanced_cnn(
                corrupted_dataset, adv_feat_extractor, adv_classifier, DEVICE, BATCH_SIZE
            )
            results["Advanced CNN (Ours)"][corruption_type].append({'level': level, 'f1': f1_adv, 'acc': acc_adv})
            print(f"    - F1-Score: {f1_adv:.4f}, Accuracy: {acc_adv:.4f}")

    # 4. Save the results to a JSON file
    results_path = os.path.join(RESULTS_DIR, "advanced_model_robustness_results.json")
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nRobustness results for the advanced model saved to: {results_path}")

if __name__ == "__main__":
    main()