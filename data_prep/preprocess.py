import os
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
import contextlib
import numpy as np
import torch
import torch.nn.functional as F
import torchaudio.transforms as T
import librosa
from config import SR, DURATION, N_FFT, HOP_LENGTH, N_MELS, CACHE_LOAD_WORKERS


def run_parallel_preprocessing(samples_with_labels, classes_list, save_root_dir, dataset_name_with_split, num_processes=None):
    print(f"Checking files for {dataset_name_with_split} precomputation...")
    os.makedirs(save_root_dir, exist_ok=True) 
    for genre_name in classes_list:
        os.makedirs(os.path.join(save_root_dir, genre_name), exist_ok=True)

    tasks_to_process = []
    if not samples_with_labels: 
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
        effective_num_processes = CACHE_LOAD_WORKERS 
    effective_num_processes = min(effective_num_processes, cpu_count())
    if effective_num_processes <= 0: effective_num_processes = 1

    print(f"Starting parallel preprocessing for {len(tasks_to_process)} {dataset_name_with_split} files using {effective_num_processes} workers...")
    with Pool(processes=effective_num_processes) as pool:
        results = list(tqdm(pool.imap_unordered(worker_preprocess_audio, tasks_to_process),
                            total=len(tasks_to_process),
                            desc=f"Precomputing {dataset_name_with_split}", ncols=100))
    successful_count = sum(1 for r in results if r is not None)
    print(f"Finished precomputing {dataset_name_with_split}. {successful_count} successful, {len(tasks_to_process) - successful_count} failed/skipped.")

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