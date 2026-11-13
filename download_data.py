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
import shutil
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
    """Download MIT Room Impulse Responses (same source and code as download_rirs.py)."""
    print("\n" + "="*60)
    print("Step 1: Downloading MIT Room Impulse Responses")
    print("="*60)

    output_dir = Path("mit_rirs")

    # Check if already has content
    if output_dir.exists() and len(list(output_dir.glob("*.wav"))) >= 100:
        count = len(list(output_dir.glob("*.wav")))
        print(f"[✓] mit_rirs already has {count} files")
        return True

    try:
        from download_rirs import download_and_process_rirs
        download_and_process_rirs()
    except Exception as e:
        print(f"[!] Error downloading MIT RIRs: {e}")
        return False

    count = len(list(output_dir.glob("*.wav"))) if output_dir.exists() else 0
    print(f"\n[✓] mit_rirs has {count} MIT RIR files")
    return count > 0


def download_background_audio_simple():
    """Download background audio from HuggingFace AudioSet parquet files.
    
    The parquet files contain FLAC-encoded audio bytes which need to be decoded.
    Uses parallel processing for faster downloads.
    """
    print("\n" + "="*60)
    print("Step 2: Downloading Background Audio")
    print("="*60)

    output_dir = Path("audioset_16k")

    # Only real AudioSet clips count; noise_*.wav from the synthetic fallback do not
    if output_dir.exists() and len(list(output_dir.glob("audioset_*.wav"))) >= 500:
        count = len(list(output_dir.glob("audioset_*.wav")))
        print(f"[✓] audioset_16k already has {count} files")
        return True

    output_dir.mkdir(exist_ok=True)

    try:
        import scipy.io.wavfile
        import numpy as np
        import soundfile as sf
    except ImportError as e:
        print(f"[!] Missing dependency: {e}")
        print("    Please run: pip install scipy numpy tqdm soundfile")
        return False

    print("Downloading background audio from HuggingFace AudioSet...")

    try:
        import pyarrow.parquet as pq
        import io
        import requests

        # Download multiple parquet files to get enough samples
        base_url = "https://huggingface.co/datasets/agkphysics/AudioSet/resolve/refs%2Fconvert%2Fparquet/balanced/train"
        # Defaults are conservative to avoid OOM kills in WSL
        parquet_count = int(os.getenv("HAWAKE_AUDIOSET_PARQUET_FILES", "2"))
        max_clips = int(os.getenv("HAWAKE_AUDIOSET_MAX_CLIPS", "1000"))
        parquet_files = [f"{base_url}/{i:04d}.parquet" for i in range(parquet_count)]

        count = 0

        def download_parquet(url):
            """Stream one ~690 MB parquet file to disk; holding it in RAM caused MemoryError."""
            dest = output_dir / (url.split('/')[-1] + ".part")
            try:
                with requests.get(url, timeout=120, stream=True) as response:
                    response.raise_for_status()
                    with open(dest, "wb") as f:
                        for chunk in response.iter_content(chunk_size=1 << 20):
                            f.write(chunk)
                return url, dest
            except Exception as e:
                print(f"\n  [!] {url.split('/')[-1]}: {e}")
                dest.unlink(missing_ok=True)
                return url, None

        def process_parquet_file(parquet_path, start_count, max_count, output_dir):
            """Extract clips a few rows at a time; buffered page reads keep a whole row group
            (AudioSet: ~140 MB) out of memory."""
            count = 0
            pf = pq.ParquetFile(parquet_path, pre_buffer=False, buffer_size=1 << 20)
            rows = (row for batch in pf.iter_batches(batch_size=4, columns=['video_id', 'audio'], use_threads=False)
                    for row in batch.to_pylist())
            for row in rows:
                if start_count + count >= max_count:
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
                            continue

                    # Normalize and convert to int16
                    audio = audio / (np.abs(audio).max() + 1e-8)
                    audio = (audio * 32000).astype(np.int16)

                    # Save
                    name = f"audioset_{row['video_id']}.wav"
                    scipy.io.wavfile.write(output_dir / name, 16000, audio)
                    count += 1
                except Exception:
                    continue
            return count

        # Download and process one file at a time; stop once enough clips exist
        print(f"Downloading up to {len(parquet_files)} parquet files (max_clips={max_clips})...")
        for url in parquet_files:
            if count >= max_clips:
                break
            filename = url.split('/')[-1]
            print(f"  Downloading: {filename}")
            _, path = download_parquet(url)
            if path is None:
                continue
            try:
                print(f"  Processing: {filename}")
                processed = process_parquet_file(path, count, max_clips, output_dir)
            finally:
                path.unlink(missing_ok=True)
            count += processed
            print(f"    Extracted {processed} clips (total: {count})")

        if count > 0:
            # Real clips replace synthetic noise from an earlier fallback
            for noise in output_dir.glob("noise_*.wav"):
                noise.unlink()
        else:
            print("[!] No clips extracted from AudioSet")
            return download_background_audio_fallback()
        return True

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
        if not download_with_progress(val_url, val_file, "Validation features (~180MB)"):
            print(f"[✗] Failed to download {val_file}")
            return False
        print(f"[✓] {val_file} downloaded")
    
    # Negative training features (17 GB). Without them the model fires on ordinary speech:
    # about 9700 false activations per hour on the validation set instead of about 9.
    train_file = "openwakeword_features_ACAV100M_2000_hrs_16bit.npy"
    train_url = "https://huggingface.co/datasets/davidscripka/openwakeword_features/resolve/main/openwakeword_features_ACAV100M_2000_hrs_16bit.npy"
    train_size = 17_280_000_128

    if os.path.exists(train_file):
        size_gb = os.path.getsize(train_file) / (1024 * 1024 * 1024)
        print(f"[✓] {train_file} already exists ({size_gb:.1f} GB)")
        return True

    print()
    free = shutil.disk_usage(".").free
    if os.getenv("HAWAKE_SKIP_LARGE_FEATURES") == "1":
        print(f"[!] Skipped {train_file} (HAWAKE_SKIP_LARGE_FEATURES=1)")
    elif free < train_size + 2 * 1024**3:
        print(f"[!] Not enough free disk space for {train_file}: "
              f"{free / 1024**3:.1f} GB free, 19 GB needed")
    else:
        part = train_file + ".part"
        if download_with_progress(train_url, part, "Training features (17 GB, the largest download)"):
            os.replace(part, train_file)
            print(f"[✓] {train_file} downloaded")
            return True
        if os.path.exists(part):
            os.remove(part)
        print(f"[✗] Failed to download {train_file}")
    print("    Without this file the trained model reacts to normal speech.")
    print("    Download it later with: python download_data.py")
    print(f"    or manually: curl -L -o {train_file} {train_url}")
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
    print("  • OpenWakeWord feature files (validation ~180MB, training negatives ~17GB)")
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
    audioset_count = len(list(Path("audioset_16k").glob("audioset_*.wav"))) if Path("audioset_16k").exists() else 0
    noise_count = len(list(Path("audioset_16k").glob("noise_*.wav"))) if Path("audioset_16k").exists() else 0
    fma_count = len(list(Path("fma").glob("*.wav"))) if Path("fma").exists() else 0
    has_validation = os.path.exists("validation_set_features.npy")
    has_training = os.path.exists("openwakeword_features_ACAV100M_2000_hrs_16bit.npy")
    
    print("Downloaded data summary:")
    print(f"  • mit_rirs:      {rir_count} files")
    print(f"  • audioset_16k:  {audioset_count} files")
    print(f"  • fma:           {fma_count} files")
    print(f"  • validation_set_features.npy: {'✓' if has_validation else '✗'}")
    print(f"  • training features (17GB):    {'✓' if has_training else '✗'}")
    print()
    
    if noise_count:
        print(f"  • synthetic noise: {noise_count} files")
        print()
        print("[WARNING] AudioSet download failed: training would use synthetic noise only,")
        print("          which gives a model with many false activations.")
        print("          Re-run: python download_data.py")
    elif audioset_count == 0 and fma_count == 0:
        print("[WARNING] No background audio downloaded!")
        print("Training may still work with synthetic samples.")
        print("For best results, manually download AudioSet data.")
    elif not has_training:
        print("[WARNING] Training features missing: the model will react to normal speech.")
        print("          Re-run: python download_data.py")
    elif audioset_count + fma_count < 100:
        print("[WARNING] Very few background audio files.")
        print("Training may produce lower quality models.")
    else:
        print("[✓] Ready for training!")
        print("    Run: streamlit run app.py")


if __name__ == "__main__":
    main()
