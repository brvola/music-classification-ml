import torch
from torch.utils.data import Dataset
import torchaudio.transforms as T
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
import os
import sys
import hashlib 

from config import (
    CACHE_LOAD_WORKERS,
)

def worker_load_pt_file(path):
    """Helper function for multiprocessing to load a single .pt file."""
    try:
        return torch.load(path, map_location='cpu')
    except Exception as e:
        print(f"Warning: worker_load_pt_file failed for {path}: {e}", file=sys.stderr)
        return None

class InMemorySpectrogramDataset(Dataset):
    def __init__(self, samples_paths_labels, dataset_name="Dataset",
                 augment=False, num_load_workers=None, cache_dir="data/cache", freq_mask_param=None, time_mask_param=None):
        self.augment = augment
        self.freq_masking = freq_mask_param
        self.time_masking = time_mask_param 
        
        os.makedirs(cache_dir, exist_ok=True)
        
        path_list_str = "".join([p for p, l in samples_paths_labels])
        file_hash = hashlib.md5(path_list_str.encode('utf-8')).hexdigest()

        cache_filename = f"cache_{dataset_name.replace(' ', '_')}_{file_hash}.pt"
        cache_path = os.path.join(cache_dir, cache_filename)

        if os.path.exists(cache_path):
            print(f"Loading {dataset_name} from cache: {cache_path}")
            try:
                cached_data = torch.load(cache_path)
                self.data = cached_data['data']
                self.labels = cached_data['labels']
                print(f"Successfully loaded {len(self.data)} samples from cache.")
            except Exception as e:
                print(f"Could not load cache file {cache_path}: {e}. Rebuilding...")
                self._load_from_source(samples_paths_labels, dataset_name, num_load_workers, cache_path)
        else:
            print(f"Cache not found for {dataset_name}. Building from source...")
            self._load_from_source(samples_paths_labels, dataset_name, num_load_workers, cache_path)
            
        if not self.data:
            raise ValueError(f"No data loaded for {dataset_name}.")

        if self.augment:
            print(f"Initializing SpecAugment for {dataset_name}: FreqM={freq_mask_param}, TimeM={time_mask_param}")
            self.freq_masking = T.FrequencyMasking(freq_mask_param=freq_mask_param)
            self.time_masking = T.TimeMasking(time_mask_param=time_mask_param)

    def _load_from_source(self, samples_paths_labels, dataset_name, num_load_workers, cache_path):
        """The original loading logic, now separated into its own method."""
        self.data, self.labels = [], []
        
        effective_load_workers = num_load_workers if num_load_workers is not None else CACHE_LOAD_WORKERS
        effective_load_workers = min(effective_load_workers, cpu_count(), len(samples_paths_labels))
        if effective_load_workers <= 0: effective_load_workers = 1

        print(f"Caching {dataset_name} into memory using {effective_load_workers} workers...")
        paths_to_load = [p for p, l in samples_paths_labels]
        original_labels = [l for p, l in samples_paths_labels]

        if effective_load_workers > 1:
            with Pool(processes=effective_load_workers) as pool:
                results = list(tqdm(pool.imap(worker_load_pt_file, paths_to_load), 
                                    total=len(paths_to_load), desc=f"Caching {dataset_name}", ncols=100))
        else:
            print(f"Using sequential caching for {dataset_name} (files: {len(paths_to_load)})")
            results = [worker_load_pt_file(p) for p in tqdm(paths_to_load, desc=f"Caching {dataset_name}", ncols=100)]

        for i, loaded_tensor in enumerate(results):
            if loaded_tensor is not None:
                self.data.append(loaded_tensor)
                self.labels.append(original_labels[i])
        
        print(f"Finished loading {len(self.data)} samples for {dataset_name} from source.")

        try:
            print(f"Saving {dataset_name} to cache file: {cache_path}")
            torch.save({'data': self.data, 'labels': self.labels}, cache_path)
        except Exception as e:
            print(f"Warning: Could not save cache to {cache_path}: {e}", file=sys.stderr)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        spec, label = self.data[idx], self.labels[idx]
        if self.augment:
            spec_aug = spec.unsqueeze(0) 
            spec_aug = self.freq_masking(spec_aug)
            spec_aug = self.time_masking(spec_aug)
            spec = spec_aug.squeeze(0)
        return spec, label