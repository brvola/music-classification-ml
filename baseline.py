import os
import sys
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split # Added random_split
from torchvision import models
import torch.nn as nn
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report, ConfusionMatrixDisplay
from tqdm import tqdm
import json
import matplotlib.pyplot as plt
import random # For shuffling full dataset before splitting

# --- Define FeatExtractor (Must match the one used for pretraining) ---
class FeatExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        resnet = models.resnet18(weights=None) 
        self.feature = nn.Sequential(*list(resnet.children())[:-1])
    def forward(self, x):
        x = self.feature(x); return x.view(x.size(0), -1)

# --- Simplified Dataset to load precomputed .pt files ---
class PrecomputedPTDataset(Dataset):
    def __init__(self, file_paths_with_labels, dataset_name="Dataset"):
        self.file_list = file_paths_with_labels
        self.dataset_name = dataset_name
        if not self.file_list:
            print(f"Warning: File list for {dataset_name} is empty.")
        # Determine the number of unique labels to help with CM display later if needed
        self.num_classes = 0
        if self.file_list:
             self.num_classes = len(set(label for _, label in file_paths_with_labels))


    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        spec_path, label = self.file_list[idx]
        try:
            spectrogram = torch.load(spec_path, map_location='cpu')
        except Exception as e:
            print(f"Error loading {spec_path} for {self.dataset_name}: {e}")
            raise e 
        return spectrogram, label

# --- CONFIGURATION FOR BASELINES ---
BEST_DL_RUN_RESULTS_DIR = "./" # <<< UPDATE THIS to your champion DL run directory
FMA_MODEL_FILENAME = "fma_best_model_std_splits_specaug.pt" 

PRECOMP_DIR_FOR_BASELINES = "precomputed_std_splits_specaug" # <<< UPDATE IF DIFFERENT
PRECOMP_GTZAN_TRAIN_DIR = os.path.join(PRECOMP_DIR_FOR_BASELINES, "gtzan_train")
PRECOMP_GTZAN_VAL_DIR   = os.path.join(PRECOMP_DIR_FOR_BASELINES, "gtzan_val") # Not directly used for training baselines but path defined
PRECOMP_GTZAN_TEST_DIR  = os.path.join(PRECOMP_DIR_FOR_BASELINES, "gtzan_test")

GTZAN_CLASSES = ['blues', 'classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock'] 
N_GTZAN_CLASSES = len(GTZAN_CLASSES)

DEVICE_STR = "cuda" if torch.cuda.is_available() else "cpu"
DEVICE = torch.device(DEVICE_STR)
BATCH_SIZE_FEATURES = 128 
SEED = 805 

BASELINE_RESULTS_SUBDIR = "baseline_comparisons"

# --- HELPER FUNCTIONS (MOVED HERE) ---
def build_precomp_paths_list(precomp_split_dir, original_classes_list):
    """
    Builds a list of (filepath, label_idx) for precomputed .pt files.
    Assumes precomp_split_dir contains subdirectories named after genres.
    """
    precomp_paths = []
    for genre_idx, genre_name in enumerate(original_classes_list):
        spec_dir = os.path.join(precomp_split_dir, genre_name)
        if os.path.isdir(spec_dir):
            for fname in os.listdir(spec_dir):
                if fname.endswith(".pt"): 
                    precomp_paths.append((os.path.join(spec_dir, fname), genre_idx))
    return precomp_paths

def build_all_gtzan_precomp_paths(precomp_base_dir, original_classes_list):
    """
    Collects all GTZAN .pt files from train, val, and test Sturm split precomputed directories.
    Shuffles them for creating a random split.
    """
    all_gtzan_paths = []
    sturm_split_subdirs = ["gtzan_train", "gtzan_val", "gtzan_test"] # Relative to precomp_base_dir
    
    for subdir_name in sturm_split_subdirs:
        split_dir_path = os.path.join(precomp_base_dir, subdir_name)
        for genre_idx, genre_name in enumerate(original_classes_list):
            spec_dir = os.path.join(split_dir_path, genre_name)
            if os.path.isdir(spec_dir):
                for fname in os.listdir(spec_dir):
                    if fname.endswith(".pt"): 
                        all_gtzan_paths.append((os.path.join(spec_dir, fname), genre_idx))
    
    if not all_gtzan_paths:
        print(f"ERROR: No GTZAN .pt files found by pooling from {precomp_base_dir} subdirectories.")
        print("Ensure 'PRECOMP_DIR_FOR_BASELINES' is correct and points to the parent of gtzan_train, gtzan_val, gtzan_test.")
        sys.exit(1)
    
    random.seed(SEED) 
    random.shuffle(all_gtzan_paths) 
    print(f"Collected and shuffled {len(all_gtzan_paths)} total GTZAN precomputed files for random splitting.")
    return all_gtzan_paths

# --- FEATURE EXTRACTION FUNCTION --- (Same as before)
def extract_features(dataset, feature_extractor_model, device, batch_size):
    feature_extractor_model.eval() 
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0) 
    all_features_list, all_labels_list = [], []
    print(f"Extracting features for {dataset.dataset_name}...")
    with torch.no_grad():
        for specs, labels in tqdm(dataloader, desc=f"Feature Extraction ({dataset.dataset_name})", ncols=80):
            specs = specs.to(device)
            features = feature_extractor_model(specs)
            all_features_list.append(features.cpu().numpy())
            all_labels_list.append(labels.numpy())
    all_features = np.concatenate(all_features_list, axis=0)
    all_labels = np.concatenate(all_labels_list, axis=0)
    print(f"Extracted {all_features.shape[0]} features, each of dimension {all_features.shape[1]}.")
    return all_features, all_labels

# --- Plotting Confusion Matrix --- (Same as before)
def plot_cm_for_baseline(cm_data, class_names, model_name, save_dir, normalize='true'):
    if cm_data is None: return
    cm_np = np.array(cm_data)
    # Ensure labels match the size of cm_np if it's smaller than full class list (e.g. if some classes not predicted)
    effective_labels = class_names
    if cm_np.shape[0] < len(class_names):
        effective_labels = class_names[:cm_np.shape[0]] # Simple truncation, might need smarter handling

    disp = ConfusionMatrixDisplay(confusion_matrix=cm_np, display_labels=effective_labels)
    fig, ax = plt.subplots(figsize=(max(8,len(effective_labels)*0.8), max(7,len(effective_labels)*0.7)))
    disp.plot(ax=ax, cmap=plt.cm.Blues, values_format='d' if normalize is None else '.2f', xticks_rotation='vertical')
    plt.title(f'GTZAN Test CM - {model_name} ({"Norm" if normalize else "Counts"})')
    plt.tight_layout()
    norm_suffix = "_norm" if normalize else "_counts"
    safe_model_name = model_name.lower().replace(' ', '_').replace('(', '').replace(')', '').replace('=', '').replace('.', '')
    plot_filename = os.path.join(save_dir, f"baseline_{safe_model_name}_cm{norm_suffix}.png")
    plt.savefig(plot_filename)
    plt.close()
    print(f"Saved {model_name} CM plot: {plot_filename}")

# --- Function to run baselines on given splits ---
def evaluate_baselines_on_splits(X_train, y_train, X_test, y_test, class_names, output_dir, split_type_name=""):
    svm_linear_params = {'kernel': 'linear', 'C': 0.1, 'random_state': SEED, 'max_iter': 5000}
    models_to_test = {
        f"LogisticRegression (C=0.1){split_type_name}": Pipeline([
            ('scaler', StandardScaler()), 
            ('logreg', LogisticRegression(solver='liblinear', multi_class='ovr', random_state=SEED, C=0.1, max_iter=2000))
        ]),
        f"KNN (k=5){split_type_name}": Pipeline([
            ('scaler', StandardScaler()),
            ('knn', KNeighborsClassifier(n_neighbors=5, n_jobs=-1))
        ]),
        f"RandomForest (100 trees){split_type_name}": Pipeline([
            ('scaler', StandardScaler()), 
            ('rf', RandomForestClassifier(n_estimators=100, random_state=SEED, n_jobs=-1, class_weight='balanced'))
        ]),
        f"SVM (Linear, C=0.1){split_type_name}": Pipeline([
            ('scaler', StandardScaler()),
            ('svm', SVC(**svm_linear_params)) 
        ]),
    }
    split_results = {}
    for model_name, model_pipeline in models_to_test.items():
        print(f"\n--- Model: {model_name} ---")
        try:
            model_pipeline.fit(X_train, y_train)
            y_test_pred = model_pipeline.predict(X_test)
            test_accuracy = accuracy_score(y_test, y_test_pred)
            test_f1_macro = f1_score(y_test, y_test_pred, average='macro', zero_division=0)
            # Ensure labels for classification_report and confusion_matrix match the actual unique labels in y_test and y_test_pred
            unique_labels_in_data = sorted(list(np.unique(np.concatenate((y_test, y_test_pred)))))
            current_class_names = [class_names[i] for i in unique_labels_in_data if i < len(class_names)]
            
            test_report_dict = classification_report(y_test, y_test_pred, labels=unique_labels_in_data, target_names=current_class_names, zero_division=0, output_dict=True)
            test_cm = confusion_matrix(y_test, y_test_pred, labels=unique_labels_in_data)


            print(f"  Test Results - Accuracy: {test_accuracy:.4f}, Macro F1: {test_f1_macro:.4f}")
            split_results[model_name] = {
                "test_accuracy": test_accuracy,
                "test_f1_macro": test_f1_macro,
                "test_classification_report_dict": test_report_dict,
                "test_confusion_matrix": test_cm.tolist()
            }
        except Exception as e:
            print(f"ERROR training or evaluating {model_name}: {e}")
            split_results[model_name] = {"error": str(e)}
    return split_results

# --- MAIN SCRIPT FOR BASELINES ---
def main():
    torch.manual_seed(SEED); np.random.seed(SEED)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(SEED)

    print(f"Using device for feature extraction: {DEVICE}")
    output_dir_base = os.path.join(BEST_DL_RUN_RESULTS_DIR, BASELINE_RESULTS_SUBDIR)
    os.makedirs(output_dir_base, exist_ok=True)
    print(f"Baseline results will be saved in: {output_dir_base}")

    fma_model_path = os.path.join(BEST_DL_RUN_RESULTS_DIR, FMA_MODEL_FILENAME)
    if not os.path.exists(fma_model_path):
        print(f"ERROR: FMA model not found: {fma_model_path}. Update 'BEST_DL_RUN_RESULTS_DIR'."); sys.exit(1)
    checkpoint = torch.load(fma_model_path, map_location=DEVICE)
    feature_extractor = FeatExtractor().to(DEVICE)
    try: feature_extractor.load_state_dict(checkpoint["feature_extractor"])
    except Exception as e: print(f"ERR loading FE state_dict: {e}"); sys.exit(1)
    print(f"Loaded FMA feature extractor from: {fma_model_path}")

    all_baseline_results = {}

    # === Scenario 1: Baselines on GTZAN Sturm Splits ===
    print("\n=== Scenario 1: Evaluating Baselines on GTZAN Sturm Splits ===")
    gtzan_sturm_train_paths = build_precomp_paths_list(PRECOMP_GTZAN_TRAIN_DIR, GTZAN_CLASSES)
    gtzan_sturm_test_paths  = build_precomp_paths_list(PRECOMP_GTZAN_TEST_DIR,  GTZAN_CLASSES)

    if not gtzan_sturm_train_paths or not gtzan_sturm_test_paths:
        print("ERR: Missing GTZAN Sturm precomp .pt files."); sys.exit(1)
    
    gtzan_sturm_train_ds = PrecomputedPTDataset(gtzan_sturm_train_paths, "GTZAN Sturm Train")
    # Make sure labels_set attribute exists for CM plotting
    gtzan_sturm_train_ds.labels_set = GTZAN_CLASSES 
    gtzan_sturm_test_ds  = PrecomputedPTDataset(gtzan_sturm_test_paths,  "GTZAN Sturm Test")
    gtzan_sturm_test_ds.labels_set = GTZAN_CLASSES

    X_train_sturm, y_train_sturm = extract_features(gtzan_sturm_train_ds, feature_extractor, DEVICE, BATCH_SIZE_FEATURES)
    X_test_sturm,  y_test_sturm  = extract_features(gtzan_sturm_test_ds,  feature_extractor, DEVICE, BATCH_SIZE_FEATURES)
    
    sturm_split_results = evaluate_baselines_on_splits(
        X_train_sturm, y_train_sturm, X_test_sturm, y_test_sturm, 
        GTZAN_CLASSES, output_dir_base, split_type_name=" (Sturm Split)"
    )
    all_baseline_results["sturm_split"] = sturm_split_results

    # === Scenario 2: Baselines on GTZAN Random 80/20 Split ===
    print("\n=== Scenario 2: Evaluating Baselines on GTZAN Random 80/20 Split ===")
    # This will pool files from PRECOMP_GTZAN_TRAIN_DIR, _VAL_DIR, _TEST_DIR
    # and then shuffle and re-split them.
    all_gtzan_pt_file_tuples = build_all_gtzan_precomp_paths(PRECOMP_DIR_FOR_BASELINES, GTZAN_CLASSES)
        
    full_gtzan_for_random_split = PrecomputedPTDataset(all_gtzan_pt_file_tuples, "GTZAN Full for Random Split")
    full_gtzan_for_random_split.labels_set = GTZAN_CLASSES # For CM plotting

    total_len = len(full_gtzan_for_random_split)
    train_len = int(0.8 * total_len)
    test_len = total_len - train_len
    
    generator = torch.Generator().manual_seed(SEED)
    train_subset_random, test_subset_random = random_split(
        full_gtzan_for_random_split, [train_len, test_len], generator=generator
    )
    print(f"GTZAN Random Split: Train={len(train_subset_random)}, Test={len(test_subset_random)}")
    
    # Create datasets from these subsets for feature extraction
    # These subsets need to be converted back to lists of (path,label) for PrecomputedPTDataset
    # OR we modify PrecomputedPTDataset to accept Subset directly (more complex for this script)
    # OR we create Dataset wrappers around Subsets
    
    # Simpler: just use the Subset directly if DataLoader handles it for feature extraction
    # For this script, extract features from the full_dataset and then split the features array.
    # This is easier than creating new Dataset objects from Subsets just for this baseline script.
    
    print("Extracting features from the *entire pooled* GTZAN set for random splitting...")
    # We need to be careful here. We want to train on 80% and test on 20% of features.
    # So, extract features from the full set, then split X and y.
    X_all_gtzan_random, y_all_gtzan_random = extract_features(full_gtzan_for_random_split, feature_extractor, DEVICE, BATCH_SIZE_FEATURES)
    
    # Now apply the same indices from random_split to X_all_gtzan_random and y_all_gtzan_random
    X_train_random = X_all_gtzan_random[train_subset_random.indices]
    y_train_random = y_all_gtzan_random[train_subset_random.indices]
    X_test_random  = X_all_gtzan_random[test_subset_random.indices]
    y_test_random  = y_all_gtzan_random[test_subset_random.indices]

    print(f"Random Split Features: X_train={X_train_random.shape}, X_test={X_test_random.shape}")

    random_split_results = evaluate_baselines_on_splits(
        X_train_random, y_train_random, X_test_random, y_test_random,
        GTZAN_CLASSES, output_dir_base, split_type_name=" (Random 80-20 Split)"
    )
    all_baseline_results["random_80_20_split"] = random_split_results
    
    results_save_path = os.path.join(output_dir_base, "all_baseline_results_comparison.json")
    try:
        with open(results_save_path, 'w') as f: json.dump(all_baseline_results, f, indent=2)
        print(f"\nAll baseline model results saved to: {results_save_path}")
    except Exception as e: print(f"Error saving all baseline results: {e}")

if __name__ == "__main__":
    if "YOUR_CHAMPION_RUN_ID_HERE" in BEST_DL_RUN_RESULTS_DIR:
        print("ERROR: Please update 'BEST_DL_RUN_RESULTS_DIR' at the top of the script.")
        print("       It should point to the specific run directory from your main DL script,")
        print("       e.g., 'results_v2/20231028_120000_seed805_fma_bs64_saTrue_gtzan_bs32_saFalse_fr10'")
        sys.exit(1)
    
    main()