# inference.py (Corrected Version 2)

import torch
import torch.nn.functional as F
import librosa
import numpy as np
import argparse
import os
from tqdm import tqdm

# Make sure model/classifier.py exists and is correct
from model.classifier import FeatExtractor, Classifier

# --- CONFIGURATION ---
# These should match the parameters used during training
SR = 22050
DURATION = 30.0
N_FFT = 2048
HOP_LENGTH = 512
N_MELS = 128
MODEL_PATH = "best_model.pt"
NUM_CLASSES = 10
INPUT_FEATURES = 512 # For ResNet-18 output

# Dropout values used for the final GTZAN fine-tuning
DROPOUT1 = 0.2
DROPOUT2 = 0.6

# The labels must be in the same order used during training
GENRE_MAP = ['blues', 'classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock']
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def preprocess_chunk(audio_chunk, sample_rate):
    """
    Converts a single audio chunk into a log-Mel spectrogram tensor ready for the model.
    """
    sgram = librosa.feature.melspectrogram(
        y=audio_chunk, sr=sample_rate, n_fft=N_FFT, hop_length=HOP_LENGTH, n_mels=N_MELS
    )
    log_sgram = librosa.power_to_db(sgram, ref=np.max)
    log_sgram_tensor = torch.tensor(log_sgram, dtype=torch.float32)
    mean, std = log_sgram_tensor.mean(), log_sgram_tensor.std()
    log_sgram_tensor = (log_sgram_tensor - mean) / (std + 1e-6)
    log_sgram_tensor = log_sgram_tensor.unsqueeze(0).repeat(3, 1, 1)
    return log_sgram_tensor.unsqueeze(0)


def main(audio_path):
    """
    Main function to run inference on a given audio file.
    """
    print(f"Using device: {DEVICE}")

    # --- 1. Load Models ---
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model file not found at '{MODEL_PATH}'")
        return

    print("Loading pre-trained model...")
    feature_extractor = FeatExtractor().to(DEVICE)

    # ==============================================================================
    # THE FIX IS HERE:
    # We now pass the dropout values to match how the model was created during training.
    # We also use positional arguments to be safe.
    classifier = Classifier(INPUT_FEATURES, NUM_CLASSES, DROPOUT1, DROPOUT2).to(DEVICE)
    # ==============================================================================

    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
    feature_extractor.load_state_dict(checkpoint['feature_extractor'])
    classifier.load_state_dict(checkpoint['classifier'])

    feature_extractor.eval()
    classifier.eval()
    print("Model loaded successfully.")

    # --- 2. Load and Chunk Audio ---
    if not os.path.exists(audio_path):
        print(f"Error: Audio file not found at '{audio_path}'")
        return

    print(f"Loading and processing audio: {audio_path}")
    try:
        audio, sr = librosa.load(audio_path, sr=SR, mono=True)
    except Exception as e:
        print(f"Error loading audio file: {e}")
        return

    samples_per_chunk = int(DURATION * sr)
    all_chunk_logits = []

    # Process audio in 30-second chunks
    for i in tqdm(range(0, len(audio), samples_per_chunk), desc="Analyzing Chunks"):
        chunk = audio[i:i + samples_per_chunk]
        if len(chunk) < samples_per_chunk:
            pad_width = samples_per_chunk - len(chunk)
            chunk = np.pad(chunk, (0, pad_width), 'constant')

        # --- 3. Preprocess and Predict for each chunk ---
        chunk_tensor = preprocess_chunk(chunk, sr).to(DEVICE)
        
        with torch.no_grad():
            features = feature_extractor(chunk_tensor)
            logits = classifier(features)
            all_chunk_logits.append(logits.squeeze(0))

    if not all_chunk_logits:
        print("Could not process any audio chunks.")
        return

    # --- 4. Aggregate Results ---
    stacked_logits = torch.stack(all_chunk_logits)
    mean_logits = torch.mean(stacked_logits, dim=0)
    probabilities = F.softmax(mean_logits, dim=0)
    predicted_index = torch.argmax(probabilities).item()
    predicted_genre = GENRE_MAP[predicted_index]
    confidence = probabilities[predicted_index].item()

    print("\n--- Inference Complete ---")
    print(f"Predicted Genre: {predicted_genre.upper()} (Confidence: {confidence:.2%})")
    print("\nFull Probability Distribution:")
    for i, prob in enumerate(probabilities):
        print(f"  - {GENRE_MAP[i]:<10}: {prob.item():.2%}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Classify the genre of an MP3 file.")
    parser.add_argument("audio_path", type=str, help="Path to the audio file (e.g., MP3, WAV).")
    args = parser.parse_args()
    
    main(args.audio_path)