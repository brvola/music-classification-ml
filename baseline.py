# FILE: baseline.py (Final Version)
# PURPOSE: To train and save baseline ML models (RF, SVM, etc.) on the 
#          features extracted from the STURM SPLITS of GTZAN. This creates
#          a fair comparison against the final fine-tuned deep learning model.

import os
import sys
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import models
import torch.nn as nn
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
from tqdm import tqdm
import json
import joblib

# --- Define FeatExtractor (Must match main.py) ---
class FeatExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        # Using weights=None as we will load our own pretrained weights
        resnet = models.resnet18(weights=None) 
        self.feature = nn.Sequential(*list(resnet.children())[:-1])
    def forward(self, x):
        x = self.feature(x)
        return x.view(x.size(0), -1)

# --- Simplified Dataset to load precomputed .pt files ---
class PrecomputedPTDataset(Dataset):
    def __init__(self, file_paths_with_labels, dataset_name="Dataset"):
        self.file_list = file_paths_with_labels
        self.dataset_name = dataset_name
        if not self.file_list:
            print(f"Warning: File list for {dataset_name} is empty.")

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

# --- CONFIGURATION ---
# This directory should contain the results of your main.py run
BEST_DL_RUN_RESULTS_DIR = "./" 
# The feature extractor to use for the baselines (from FMA pretraining)
FMA_MODEL_FILENAME = "fma_best_model_std_splits_specaug.pt" 

# The directory containing the precomputed spectrograms from the Sturm splits
PRECOMP_DIR_FOR_BASELINES = "precomputed_std_splits_specaug"
PRECOMP_GTZAN_TRAIN_DIR = os.path.join(PRECOMP_DIR_FOR_BASELINES, "gtzan_train")
PRECOMP_GTZAN_TEST_DIR  = os.path.join(PRECOMP_DIR_FOR_BASELINES, "gtzan_test")

# GTZAN configuration
GTZAN_CLASSES = ['blues', 'classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock'] 

# System and paths configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE_FEATURES = 128 
SEED = 805 
BASELINE_RESULTS_SUBDIR = "baseline_comparisons"
SAVED_MODELS_DIR = os.path.join(BASELINE_RESULTS_SUBDIR, "saved_models")

# --- HELPER FUNCTIONS ---
def build_precomp_paths_list(precomp_split_dir, original_classes_list):
    """Builds a list of (filepath, label_idx) for precomputed .pt files."""
    precomp_paths = []
    for genre_idx, genre_name in enumerate(original_classes_list):
        spec_dir = os.path.join(precomp_split_dir, genre_name)
        if os.path.isdir(spec_dir):
            for fname in os.listdir(spec_dir):
                if fname.endswith(".pt"): 
                    precomp_paths.append((os.path.join(spec_dir, fname), genre_idx))
    return precomp_paths

def extract_features(dataset, feature_extractor_model, device, batch_size):
    """Extracts deep features from a dataset of spectrograms."""
    feature_extractor_model.eval() 
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0) 
    all_features_list, all_labels_list = [], []
    
    print(f"Extracting features for {dataset.dataset_name}...")
    with torch.no_grad():
        for specs, labels in tqdm(dataloader, desc=f"Feature Extraction ({dataset.dataset_name})", ncols=100):
            specs = specs.to(device)
            features = feature_extractor_model(specs)
            all_features_list.append(features.cpu().numpy())
            all_labels_list.append(labels.numpy())
            
    all_features = np.concatenate(all_features_list, axis=0)
    all_labels = np.concatenate(all_labels_list, axis=0)
    print(f"Extracted {all_features.shape[0]} features, each of dimension {all_features.shape[1]}.")
    return all_features, all_labels

# --- MAIN BASELINE TRAINING FUNCTION ---
def train_and_evaluate_baselines(X_train, y_train, X_test, y_test, class_names, saved_models_path):
    """
    Trains and saves baseline models based on the provided features from the Sturm splits.
    Hyperparameters are set to match the group's paper for consistency.
    """
    os.makedirs(saved_models_path, exist_ok=True)
    
    # Define models with hyperparameters matching the group paper
    models_to_test = {
        "RandomForest": Pipeline([
            ('scaler', StandardScaler()), 
            ('rf', RandomForestClassifier(n_estimators=100, random_state=SEED, n_jobs=-1))
        ]),
        "SVM_RBF": Pipeline([
            ('scaler', StandardScaler()),
            ('svm', SVC(kernel='rbf', C=10, random_state=SEED, probability=True)) 
        ]),
        "KNN": Pipeline([
            ('scaler', StandardScaler()),
            ('knn', KNeighborsClassifier(n_neighbors=5, n_jobs=-1))
        ]),
        "LogisticRegression": Pipeline([
            ('scaler', StandardScaler()), 
            ('logreg', LogisticRegression(solver='liblinear', multi_class='ovr', C=1.0, random_state=SEED, max_iter=2000))
        ]),
    }
    
    split_results = {}
    for model_name, model_pipeline in models_to_test.items():
        # The split name is hardcoded to Sturm for clarity
        full_model_name = f"{model_name}(Sturm_Split_Features)"
        print(f"\n--- Training Model: {full_model_name} ---")
        
        try:
            # 1. Train the model
            model_pipeline.fit(X_train, y_train)
            
            # 2. Save the trained model
            model_filename = f"{model_name}_SturmSplit.joblib"
            model_save_path = os.path.join(saved_models_path, model_filename)
            joblib.dump(model_pipeline, model_save_path)
            print(f"  -> Model saved to: {model_save_path}")

            # 3. Evaluate on the clean Sturm test set
            y_test_pred = model_pipeline.predict(X_test)
            test_accuracy = accuracy_score(y_test, y_test_pred)
            test_f1_macro = f1_score(y_test, y_test_pred, average='macro', zero_division=0)
            
            print(f"  Clean Test Results - Accuracy: {test_accuracy:.4f}, Macro F1: {test_f1_macro:.4f}")
            
            report = classification_report(y_test, y_test_pred, target_names=class_names, zero_division=0, output_dict=True)
            cm = confusion_matrix(y_test, y_test_pred)
            
            split_results[full_model_name] = {
                "test_accuracy": test_accuracy,
                "test_f1_macro": test_f1_macro,
                "saved_model_path": model_save_path,
                "classification_report": report,
                "confusion_matrix": cm.tolist()
            }
        except Exception as e:
            print(f"ERROR training or evaluating {full_model_name}: {e}")
            split_results[full_model_name] = {"error": str(e)}
            
    return split_results

# --- MAIN SCRIPT EXECUTION ---
def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    
    # Setup output directories
    output_dir_base = os.path.join(BEST_DL_RUN_RESULTS_DIR, BASELINE_RESULTS_SUBDIR)
    saved_models_dir_path = os.path.join(BEST_DL_RUN_RESULTS_DIR, SAVED_MODELS_DIR)
    os.makedirs(output_dir_base, exist_ok=True)
    os.makedirs(saved_models_dir_path, exist_ok=True)
    print(f"Baseline models will be saved in: {saved_models_dir_path}")

    # 1. Load the FMA-pretrained Feature Extractor
    print(f"Using device for feature extraction: {DEVICE}")
    fma_model_path = os.path.join(BEST_DL_RUN_RESULTS_DIR, FMA_MODEL_FILENAME)
    if not os.path.exists(fma_model_path):
        print(f"ERROR: FMA model not found: {fma_model_path}. Run main.py first."); sys.exit(1)
    
    checkpoint = torch.load(fma_model_path, map_location=DEVICE)
    feature_extractor = FeatExtractor().to(DEVICE)
    try:
        feature_extractor.load_state_dict(checkpoint["feature_extractor"])
    except Exception as e:
        print(f"Error loading feature extractor state_dict: {e}"); sys.exit(1)
    print(f"Successfully loaded FMA-pretrained feature extractor from: {fma_model_path}")

    # 2. Load data paths from the Sturm splits
    print("\nLoading data from STURM SPLITS for fair baseline creation...")
    gtzan_sturm_train_paths = build_precomp_paths_list(PRECOMP_GTZAN_TRAIN_DIR, GTZAN_CLASSES)
    gtzan_sturm_test_paths  = build_precomp_paths_list(PRECOMP_GTZAN_TEST_DIR,  GTZAN_CLASSES)

    if not gtzan_sturm_train_paths or not gtzan_sturm_test_paths:
        print("ERROR: Missing precomputed files in the Sturm split directories."); 
        print(f"Checked: {PRECOMP_GTZAN_TRAIN_DIR} and {PRECOMP_GTZAN_TEST_DIR}");
        sys.exit(1)
    
    gtzan_sturm_train_ds = PrecomputedPTDataset(gtzan_sturm_train_paths, "GTZAN Sturm Train")
    gtzan_sturm_test_ds  = PrecomputedPTDataset(gtzan_sturm_test_paths,  "GTZAN Sturm Test")
    print(f"Found {len(gtzan_sturm_train_ds)} training files and {len(gtzan_sturm_test_ds)} test files.")

    # 3. Extract features using the loaded feature extractor
    X_train, y_train = extract_features(gtzan_sturm_train_ds, feature_extractor, DEVICE, BATCH_SIZE_FEATURES)
    X_test,  y_test  = extract_features(gtzan_sturm_test_ds,  feature_extractor, DEVICE, BATCH_SIZE_FEATURES)
    
    # 4. Train, evaluate, and save the baseline models
    sturm_split_baseline_results = train_and_evaluate_baselines(
        X_train, y_train, X_test, y_test, 
        GTZAN_CLASSES, saved_models_dir_path
    )
    
    # 5. Save the performance summary to a JSON file
    results_save_path = os.path.join(output_dir_base, "sturm_split_baseline_results.json")
    try:
        with open(results_save_path, 'w') as f:
            json.dump(sturm_split_baseline_results, f, indent=2)
        print(f"\nBaseline model results on the Sturm split saved to: {results_save_path}")
    except Exception as e:
        print(f"Error saving baseline results JSON: {e}")

if __name__ == "__main__":
    main()