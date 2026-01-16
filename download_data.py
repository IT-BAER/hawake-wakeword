#!/usr/bin/env python3
"""
Download training data for HAwake WakeWord Training.

This script downloads the required data for wake word training:
1. MIT Room Impulse Responses (from HuggingFace)
2. Background audio samples (from HuggingFace FSD50K - more reliable than AudioSet)
3. Pre-computed OpenWakeWord features (validation + optional training)

Alternative to FSD50K: You can manually download AudioSet from:
https://huggingface.co/datasets/agkphysics/AudioSet

Total download: ~1-2 GB (without large training features)
"""

import os
import sys
import json
import subprocess
from pathlib import Path
from urllib.request import urlretrieve, urlopen
from urllib.error import URLError


def download_with_progress(url: str, dest: str, description: str = None):
    """Download a file with progress indicator."""
    if description:
        print(f"Downloading {description}...")
    print(f"  URL: {url}")
    print(f"  Destination: {dest}")
    
    def progress_hook(block_num, block_size, total_size):
        if total_size > 0:
            downloaded = block_num * block_size
            percent = min(100, downloaded * 100 / total_size)
            mb_downloaded = downloaded / (1024 * 1024)
            mb_total = total_size / (1024 * 1024)
            sys.stdout.write(f"\r  Progress: {percent:.1f}% ({mb_downloaded:.1f} MB / {mb_total:.1f} MB)")
            sys.stdout.flush()
    
    try:
        urlretrieve(url, dest, progress_hook)
        print()  # New line after progress
        return True
    except URLError as e:
        print(f"\n  Error: {e}")
        return False
    except Exception as e:
        print(f"\n  Error: {e}")
        return False


def download_mit_rirs():
    """Download MIT Room Impulse Responses using HuggingFace datasets."""
    print("\n" + "="*60)
    print("Step 1: Downloading MIT Room Impulse Responses")
    print("="*60)
    
    output_dir = Path("mit_rirs")
    
    # Check if already has content
    if output_dir.exists() and len(list(output_dir.glob("*.wav"))) >= 100:
        count = len(list(output_dir.glob("*.wav")))
        print(f"[✓] mit_rirs already has {count} files")
        return True
    
    output_dir.mkdir(exist_ok=True)
    
    try:
        import scipy.io.wavfile
        import numpy as np
        from tqdm import tqdm
        import datasets
    except ImportError as e:
        print(f"[!] Missing dependency: {e}")
        print("    Please run: pip install scipy numpy tqdm datasets")
        return False
    
    print("Downloading MIT RIRs from HuggingFace...")
    print("Source: davidscripka/MIT_environmental_impulse_responses")
    
    try:
        rir_dataset = datasets.load_dataset(
            "davidscripka/MIT_environmental_impulse_responses",
            split="train",
            streaming=True
        )
        
        count = 0
        for row in tqdm(rir_dataset, desc="Downloading RIRs"):
            name = row['audio']['path'].split('/')[-1]
            audio = (np.array(row['audio']['array']) * 32767).astype(np.int16)
            scipy.io.wavfile.write(output_dir / name, 16000, audio)
            count += 1
        
        print(f"\n[✓] Downloaded {count} MIT RIR files")
        return count > 0
        
    except Exception as e:
        print(f"[!] Error downloading MIT RIRs: {e}")
        return False


def download_background_audio_simple():
    """Download background audio from HuggingFace AudioSet parquet files.
    
    The parquet files contain FLAC-encoded audio bytes which need to be decoded.
    """
    print("\n" + "="*60)
    print("Step 2: Downloading Background Audio")
    print("="*60)
    
    output_dir = Path("audioset_16k")
    
    # Check if already has content
    if output_dir.exists() and len(list(output_dir.glob("*.wav"))) >= 500:
        count = len(list(output_dir.glob("*.wav")))
        print(f"[✓] audioset_16k already has {count} files")
        return True
    
    output_dir.mkdir(exist_ok=True)
    
    try:
        import scipy.io.wavfile
        import numpy as np
        from tqdm import tqdm
        import soundfile as sf
    except ImportError as e:
        print(f"[!] Missing dependency: {e}")
        print("    Please run: pip install scipy numpy tqdm soundfile")
        return False
    
    print("Downloading background audio from HuggingFace AudioSet...")
    print("This may take a few minutes...")
    
    try:
        import pyarrow.parquet as pq
        import io
        import requests
        
        # Download multiple parquet files to get enough samples
        base_url = "https://huggingface.co/datasets/agkphysics/AudioSet/resolve/refs%2Fconvert%2Fparquet/balanced/train"
        parquet_files = [f"{base_url}/{i:04d}.parquet" for i in range(5)]  # First 5 files
        
        count = 0
        max_clips = 2000
        errors = 0
        
        for parquet_url in parquet_files:
            if count >= max_clips:
                break
                
            print(f"\nFetching: {parquet_url.split('/')[-1]}")
            try:
                response = requests.get(parquet_url, timeout=120)
                response.raise_for_status()
            except Exception as e:
                print(f"  [!] Failed to download: {e}")
                continue
            
            # Read parquet into memory
            parquet_bytes = io.BytesIO(response.content)
            table = pq.read_table(parquet_bytes)
            df = table.to_pandas()
            
            print(f"  Processing {len(df)} audio samples...")
            
            for idx, row in tqdm(df.iterrows(), total=len(df), desc="  Decoding", leave=False):
                if count >= max_clips:
                    break
                try:
                    audio_data = row['audio']
                    if not isinstance(audio_data, dict) or 'bytes' not in audio_data:
                        continue
                    
                    # Decode FLAC bytes using soundfile
                    audio_bytes = io.BytesIO(audio_data['bytes'])
                    audio, sr = sf.read(audio_bytes)
                    
                    # Convert to mono if stereo
                    if len(audio.shape) > 1:
                        audio = audio.mean(axis=1)
                    
                    # Resample to 16kHz if needed
                    if sr != 16000:
                        try:
                            import librosa
                            audio = librosa.resample(audio.astype(np.float32), orig_sr=sr, target_sr=16000)
                        except ImportError:
                            # Skip if librosa not available and resampling needed
                            continue
                    
                    # Normalize and convert to int16
                    audio = audio / (np.abs(audio).max() + 1e-8)  # Normalize to -1, 1
                    audio = (audio * 32000).astype(np.int16)
                    
                    # Save
                    name = f"audioset_{row['video_id']}.wav"
                    scipy.io.wavfile.write(output_dir / name, 16000, audio)
                    count += 1
                except Exception as e:
                    errors += 1
                    if errors > 50:
                        print(f"\n  [!] Too many errors, moving to next file")
                        break
                    continue
        
        saved = len(list(output_dir.glob("*.wav")))
        print(f"\n[✓] Saved {saved} audio files to audioset_16k/")
        return saved > 0
        
    except Exception as e:
        print(f"[!] Error with parquet download: {e}")
        import traceback
        traceback.print_exc()
        print("    Trying fallback method...")
        return download_background_audio_fallback()


def download_background_audio_fallback():
    """Fallback: Generate simple noise samples for testing."""
    print("\n[!] Creating synthetic background audio for testing...")
    print("    For best results, manually download AudioSet from:")
    print("    https://huggingface.co/datasets/agkphysics/AudioSet")
    
    output_dir = Path("audioset_16k")
    output_dir.mkdir(exist_ok=True)
    
    try:
        import scipy.io.wavfile
        import numpy as np
        
        # Generate 100 noise samples for basic testing
        for i in range(100):
            # Generate pink noise (more realistic than white noise)
            samples = 16000 * 10  # 10 seconds at 16kHz
            white = np.random.randn(samples)
            
            # Simple low-pass to create pink-ish noise
            from scipy import signal
            b, a = signal.butter(3, 0.1)
            pink = signal.filtfilt(b, a, white)
            
            # Normalize and convert
            pink = pink / np.abs(pink).max() * 0.5
            audio = (pink * 32767).astype(np.int16)
            
            scipy.io.wavfile.write(output_dir / f"noise_{i:04d}.wav", 16000, audio)
        
        print(f"[✓] Created 100 synthetic background samples")
        print("    Note: For production models, use real AudioSet data")
        return True
        
    except Exception as e:
        print(f"[!] Error creating synthetic audio: {e}")
        return False


def download_fma():
    """Download Free Music Archive samples."""
    print("\n" + "="*60)
    print("Step 3: Downloading Free Music Archive (FMA) Samples")
    print("="*60)
    
    output_dir = Path("fma")
    
    # Check if already has content
    if output_dir.exists() and len(list(output_dir.glob("*.wav"))) >= 50:
        count = len(list(output_dir.glob("*.wav")))
        print(f"[✓] fma already has {count} files")
        return True
    
    output_dir.mkdir(exist_ok=True)
    
    print("[!] FMA dataset requires special handling")
    print("    For now, AudioSet alone is sufficient for training")
    print("    FMA can be manually downloaded from: https://huggingface.co/datasets/rudraml/fma")
    
    return True  # Non-fatal


def download_feature_files():
    """Download pre-computed OpenWakeWord features for training and validation."""
    print("\n" + "="*60)
    print("Step 4: Downloading OpenWakeWord Feature Files")
    print("="*60)
    
    # Validation set is smaller and required
    val_file = "validation_set_features.npy"
    val_url = "https://huggingface.co/datasets/davidscripka/openwakeword_features/resolve/main/validation_set_features.npy"
    
    if os.path.exists(val_file):
        size_mb = os.path.getsize(val_file) / (1024 * 1024)
        print(f"[✓] {val_file} already exists ({size_mb:.1f} MB)")
    else:
        if not download_with_progress(val_url, val_file, "Validation features (~50MB)"):
            print(f"[✗] Failed to download {val_file}")
            return False
        print(f"[✓] {val_file} downloaded")
    
    # Large training features file (16GB!) - make this optional
    train_file = "openwakeword_features_ACAV100M_2000_hrs_16bit.npy"
    train_url = "https://huggingface.co/datasets/davidscripka/openwakeword_features/resolve/main/openwakeword_features_ACAV100M_2000_hrs_16bit.npy"
    
    if os.path.exists(train_file):
        size_gb = os.path.getsize(train_file) / (1024 * 1024 * 1024)
        print(f"[✓] {train_file} already exists ({size_gb:.1f} GB)")
    else:
        print()
        print("[!] Large training features file not found (~16GB download)")
        print("    This file significantly improves training quality.")
        print()
        print("    To download (recommended for best quality):")
        print(f"    curl -L -o {train_file} {train_url}")
        print()
        print("    Or via browser: https://huggingface.co/datasets/davidscripka/openwakeword_features")
        print()
        print("    Training will still work without it using AudioSet background audio.")
        print()
    
    return True


def main():
    print("="*60)
    print("  HAwake WakeWord Training - Data Download")
    print("="*60)
    print()
    print("This script downloads the required data for wake word training.")
    print()
    print("Downloads:")
    print("  • MIT Room Impulse Responses (~10MB)")
    print("  • Background audio samples (~500MB-1GB)")
    print("  • OpenWakeWord feature files (validation ~50MB)")
    print()
    
    # Check if we have required dependencies
    try:
        import scipy
        import numpy
        from tqdm import tqdm
    except ImportError as e:
        print(f"Error: Missing required package: {e}")
        print("Please install dependencies first: pip install -r requirements.txt")
        sys.exit(1)
    
    # Step 1: MIT RIRs (for audio augmentation)
    download_mit_rirs()
    
    # Step 2: Background audio
    download_background_audio_simple()
    
    # Step 3: FMA (optional music background)
    download_fma()
    
    # Step 4: Feature files (for training)
    if not download_feature_files():
        print("\n[ERROR] Failed to download required feature files")
        sys.exit(1)
    
    print("\n" + "="*60)
    print("  Download Complete!")
    print("="*60)
    print()
    
    # Summary
    rir_count = len(list(Path("mit_rirs").glob("*.wav"))) if Path("mit_rirs").exists() else 0
    audioset_count = len(list(Path("audioset_16k").glob("*.wav"))) if Path("audioset_16k").exists() else 0
    fma_count = len(list(Path("fma").glob("*.wav"))) if Path("fma").exists() else 0
    has_validation = os.path.exists("validation_set_features.npy")
    has_training = os.path.exists("openwakeword_features_ACAV100M_2000_hrs_16bit.npy")
    
    print("Downloaded data summary:")
    print(f"  • mit_rirs:      {rir_count} files")
    print(f"  • audioset_16k:  {audioset_count} files")
    print(f"  • fma:           {fma_count} files")
    print(f"  • validation_set_features.npy: {'✓' if has_validation else '✗'}")
    print(f"  • training features (16GB):    {'✓' if has_training else '✗ (optional)'}")
    print()
    
    if audioset_count == 0 and fma_count == 0:
        print("[WARNING] No background audio downloaded!")
        print("Training may still work with synthetic samples.")
        print("For best results, manually download AudioSet data.")
    elif audioset_count + fma_count < 100:
        print("[WARNING] Very few background audio files.")
        print("Training may produce lower quality models.")
    else:
        print("[✓] Ready for training!")
        print("    Run: streamlit run app.py")


if __name__ == "__main__":
    main()
