import sys
import os
import pandas as pd

# Import individual variables from the config file
from config import (
    GTZAN_SPLITS_DIR,
    GTZAN_TRAIN_SPLIT_PATH,
    GTZAN_VALIDATION_SPLIT_PATH,
    GTZAN_TEST_SPLIT_PATH
)


def build_gtzan_samples_from_sturm_splits(gtzan_audio_root, gtzan_classes_list):
    """
    Parses Bob Sturm's official split files for GTZAN and maps them to audio file paths.
    """
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
            print(f"ERROR: GTZAN Split file not found: {filepath}. Please download/place it in '{GTZAN_SPLITS_DIR}'.", file=sys.stderr)
            sys.exit(1)
        return parsed_entries

    sturm_train_entries = load_split_file_entries(GTZAN_TRAIN_SPLIT_PATH)
    sturm_val_entries   = load_split_file_entries(GTZAN_VALIDATION_SPLIT_PATH)
    sturm_test_entries  = load_split_file_entries(GTZAN_TEST_SPLIT_PATH)

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
                if missing_files_count < 5:
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
    """
    Parses the FMA metadata CSV to get the official small subset splits.
    """
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