import os
import torch

# DATASET PATHS
# These ROOT dirs are used if precomputed spectrograms aren't found.
# GTZAN: https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification
# FMA: https://www.kaggle.com/datasets/imsparsh/fma-free-music-archive-small-medium?select=fma_small
GTZAN_ROOT = "data/gtzan/genres"
FMA_ROOT = "data/fma_small"
FMA_METADATA = "data/fma_metadata/tracks.csv"

# It's easier to just use this Kaggle dataset of precomputed spectrograms
KAGGLE_DATASET = "bravola/precomputed-mel-spectrograms-fma-small-and-gtzan"

# Bob Sturms GTZAN Splits: https://github.com/boblsturm/GTZAN
GTZAN_SPLITS_DIR = "data/gtzan_splits"
GTZAN_TEST_SPLIT_URL = "https://github.com/boblsturm/GTZAN/blob/main/test_filtered.txt"
GTZAN_VALIDATION_SPLIT_URL = "https://github.com/boblsturm/GTZAN/blob/main/valid_filtered.txt"
GTZAN_TRAIN_SPLIT_URL = "https://github.com/boblsturm/GTZAN/blob/main/train_filtered.txt"

# Full paths to local GTZAN split files (used by build_splits.py)
GTZAN_TRAIN_SPLIT_PATH = os.path.join(GTZAN_SPLITS_DIR, "train_filtered.txt")
GTZAN_VALIDATION_SPLIT_PATH = os.path.join(GTZAN_SPLITS_DIR, "valid_filtered.txt")
GTZAN_TEST_SPLIT_PATH = os.path.join(GTZAN_SPLITS_DIR, "test_filtered.txt")


# PRECOMPUTED SPECTROGRAM DIRECTORIES
PRECOMP_DIR = "data/precomputed_std_splits"
PRECOMP_GTZAN_TRAIN_DIR = os.path.join(PRECOMP_DIR, "gtzan_train")
PRECOMP_GTZAN_VAL_DIR   = os.path.join(PRECOMP_DIR, "gtzan_val")
PRECOMP_GTZAN_TEST_DIR  = os.path.join(PRECOMP_DIR, "gtzan_test")
PRECOMP_FMA_TRAIN_DIR = os.path.join(PRECOMP_DIR, "fma_train")
PRECOMP_FMA_VAL_DIR   = os.path.join(PRECOMP_DIR, "fma_val")
PRECOMP_FMA_TEST_DIR  = os.path.join(PRECOMP_DIR, "fma_test") 

# EXECUTION PARAMETERS
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_WORKERS    = 0
CACHE_LOAD_WORKERS = 0
PIN_MEMORY     = True

# SPECAUGMENT PARAMETERS
APPLY_SPECAUGMENT_FMA   = True 
APPLY_SPECAUGMENT_GTZAN = True
SPECAUGMENT_FREQ_MASK_PARAM = 27
SPECAUGMENT_TIME_MASK_PARAM = 70
GTZAN_SPECAUG_FREQ_MASK_PARAM = 15 
GTZAN_SPECAUG_TIME_MASK_PARAM = 40 

# FMA PRE-TRAINING PARAMETERS
FMA_PRETRAIN_BATCH_SIZE   = 64
FMA_EPOCHS     = 25
FMA_LR = 5e-4
FMA_WEIGHT_DECAY = 8e-3
FMA_SCHEDULER_PATIENCE = 3
FMA_EARLY_STOPPING_PATIENCE = 10

# GTZAN FINE-TUNING PARAMETERS
FREEZE_EPOCHS  = 5
GTZAN_FINETUNE_BATCH_SIZE = 32
GTZAN_EPOCHS     = 60
GTZAN_LR_FR = 1e-3
GTZAN_WEIGHT_DECAY_FR = 1e-3
GTZAN_LR_FT = 3e-5
GTZAN_WEIGHT_DECAY_FT = 5e-3

# SPECTROGRAM PARAMETERS
SR          = 22050
N_FFT       = 2048
HOP_LENGTH  = 512
N_MELS      = 128
DURATION    = 30.0