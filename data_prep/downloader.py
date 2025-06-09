from config import (
    KAGGLE_DATASET,
    GTZAN_SPLITS_DIR,
    GTZAN_TRAIN_SPLIT_URL,
    GTZAN_VALIDATION_SPLIT_URL,
    GTZAN_TEST_SPLIT_URL,
    GTZAN_TRAIN_SPLIT_PATH,
    GTZAN_VALIDATION_SPLIT_PATH,
    GTZAN_TEST_SPLIT_PATH,
    PRECOMP_DIR,
)
import os

data_dir = "data"

def ensure_gtzan_splits():
    """
    Ensures that the GTZAN split files are present in the expected directory.
    If not, it downloads them from the specified URLs.
    """
    os.makedirs(GTZAN_SPLITS_DIR, exist_ok=True)

    if not os.path.exists(GTZAN_TRAIN_SPLIT_PATH):
        print(f"Downloading GTZAN train split from {GTZAN_TRAIN_SPLIT_URL}...")
        os.system(f"wget -q {GTZAN_TRAIN_SPLIT_URL} -O {GTZAN_TRAIN_SPLIT_PATH}")

    if not os.path.exists(GTZAN_VALIDATION_SPLIT_PATH):
        print(f"Downloading GTZAN validation split from {GTZAN_VALIDATION_SPLIT_URL}...")
        os.system(f"wget -q {GTZAN_VALIDATION_SPLIT_URL} -O {GTZAN_VALIDATION_SPLIT_PATH}")

    if not os.path.exists(GTZAN_TEST_SPLIT_PATH):
        print(f"Downloading GTZAN test split from {GTZAN_TEST_SPLIT_URL}...")
        os.system(f"wget -q {GTZAN_TEST_SPLIT_URL} -O {GTZAN_TEST_SPLIT_PATH}")
    
    print("GTZAN split files are ready.")

def verify_kaggle_dataset():
    """
    Checks if the Kaggle precomputed dataset is available.
    If not, it prompts the user to download it from Kaggle.
    """
    kaggle_dir = os.path.join(data_dir, "precomputed_std_splits")
    if not os.path.exists(kaggle_dir):
        print(f"Precomputed dataset 'https://www.kaggle.com/datasets/bravola/precomputed-mel-spectrograms-fma-small-and-gtzan' not found.")
        print("Please download it from Kaggle and place it in 'data/' directory, so 'data/precomputed_std_splits/ should exist'.")
        return False
    print(f"Kaggle dataset 'https://www.kaggle.com/datasets/bravola/precomputed-mel-spectrograms-fma-small-and-gtzan' is present.")
    return True