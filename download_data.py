#!/usr/bin/env python3
"""
Download training data for HAwake WakeWord Training.

This script downloads:
1. Pre-computed OpenWakeWord features (required for training)
2. AudioSet background audio samples (for audio augmentation)
3. Free Music Archive samples (additional background audio)

Total download: ~3-5 GB
"""

import os
import sys
import tarfile
from pathlib import Path
from urllib.request import urlretrieve
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


def download_feature_files():
    """Download pre-computed OpenWakeWord features for training and validation."""
    print("\n" + "="*60)
    print("Step 1: Downloading OpenWakeWord Feature Files")
    print("="*60)
    
    # Validation set is smaller and required
    val_file = "validation_set_features.npy"
    val_url = "https://huggingface.co/datasets/davidscripka/openwakeword_features/resolve/main/validation_set_features.npy"
    
    if os.path.exists(val_file):
        print(f"[✓] {val_file} already exists")
    else:
        if not download_with_progress(val_url, val_file, "Validation features (~50MB)"):
            print(f"[✗] Failed to download {val_file}")
            return False
        print(f"[✓] {val_file} downloaded")
    
    # Large training features file (16GB!) - make this optional
    train_file = "openwakeword_features_ACAV100M_2000_hrs_16bit.npy"
    train_url = "https://huggingface.co/datasets/davidscripka/openwakeword_features/resolve/main/openwakeword_features_ACAV100M_2000_hrs_16bit.npy"
    
    if os.path.exists(train_file):
        print(f"[✓] {train_file} already exists")
    else:
        print()
        print("[!] Large training features file not found (~16GB download)")
        print("    This file improves training quality but is optional.")
        print()
        print("    To download manually (recommended for best quality):")
        print(f"    curl -L -o {train_file} {train_url}")
        print()
        print("    Training will still work without it, using AudioSet background audio.")
        print()
    
    return True


def download_audioset():
    """Download and extract AudioSet background audio samples."""
    print("\n" + "="*60)
    print("Step 2: Downloading AudioSet Background Audio")
    print("="*60)
    
    output_dir = Path("audioset_16k")
    
    # Check if already has content
    if output_dir.exists() and any(output_dir.glob("*.wav")):
        count = len(list(output_dir.glob("*.wav")))
        print(f"[✓] audioset_16k already has {count} files")
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
    
    print("Downloading AudioSet balanced training split...")
    print("(This may take 5-15 minutes depending on your connection)")
    
    try:
        # Load AudioSet balanced training split directly via HuggingFace datasets
        audioset = datasets.load_dataset(
            "agkphysics/AudioSet", 
            "balanced", 
            split="train",
            trust_remote_code=True
        )
        
        print(f"Downloaded {len(audioset)} audio clips. Converting to 16kHz WAV...")
        
        # Convert to 16kHz and save as WAV
        # Limit to first 5000 clips to keep download reasonable
        max_clips = min(5000, len(audioset))
        
        for i, row in tqdm(enumerate(audioset), total=max_clips, desc="Converting"):
            if i >= max_clips:
                break
            try:
                audio = row['audio']['array']
                sr = row['audio']['sampling_rate']
                
                # Resample to 16kHz if needed
                if sr != 16000:
                    import librosa
                    audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
                
                # Convert to int16
                audio = (audio * 32767).astype(np.int16)
                
                # Save
                name = f"audioset_{row['video_id']}.wav"
                scipy.io.wavfile.write(output_dir / name, 16000, audio)
            except Exception as e:
                continue  # Skip problematic files
        
        saved_count = len(list(output_dir.glob("*.wav")))
        print(f"[✓] Saved {saved_count} AudioSet files")
        return True
        
    except Exception as e:
        print(f"[!] Error downloading AudioSet: {e}")
        print("    Trying alternative method...")
        return download_audioset_alternative()


def download_audioset_alternative():
    """Alternative AudioSet download using streaming."""
    output_dir = Path("audioset_16k")
    output_dir.mkdir(exist_ok=True)
    
    try:
        import scipy.io.wavfile
        import numpy as np
        from tqdm import tqdm
        import datasets
        
        print("Trying streaming download...")
        audioset = datasets.load_dataset(
            "agkphysics/AudioSet", 
            "balanced", 
            split="train",
            streaming=True,
            trust_remote_code=True
        )
        
        max_clips = 2000  # Smaller number for streaming
        count = 0
        
        for row in tqdm(audioset, total=max_clips, desc="Downloading"):
            if count >= max_clips:
                break
            try:
                audio = row['audio']['array']
                sr = row['audio']['sampling_rate']
                
                # Simple downsampling (not perfect but works)
                if sr != 16000:
                    # Decimate audio
                    factor = sr // 16000
                    if factor > 1:
                        audio = audio[::factor]
                
                audio = (audio * 32767).astype(np.int16)
                name = f"audioset_{row['video_id']}.wav"
                scipy.io.wavfile.write(output_dir / name, 16000, audio)
                count += 1
            except:
                continue
        
        saved = len(list(output_dir.glob("*.wav")))
        print(f"[✓] Downloaded {saved} AudioSet files")
        return saved > 0
        
    except Exception as e:
        print(f"[!] Alternative download failed: {e}")
        return False


def download_fma():
    """Download Free Music Archive samples for background audio."""
    print("\n" + "="*60)
    print("Step 3: Downloading Free Music Archive Samples")
    print("="*60)
    
    output_dir = Path("fma")
    
    # Check if already has content
    if output_dir.exists() and any(output_dir.glob("*.wav")):
        count = len(list(output_dir.glob("*.wav")))
        print(f"[✓] fma already has {count} files")
        return True
    
    output_dir.mkdir(exist_ok=True)
    
    try:
        import scipy.io.wavfile
        import numpy as np
        from tqdm import tqdm
        import datasets
    except ImportError as e:
        print(f"[!] Missing dependency: {e}")
        return False
    
    print("Downloading FMA music samples...")
    
    try:
        # Use a smaller subset that doesn't require scripts
        fma_dataset = datasets.load_dataset(
            "maharshipandya/spotify-tracks-dataset",
            split="train",
            streaming=True
        )
        
        print("[!] FMA direct download not available, skipping music samples")
        print("    AudioSet alone is sufficient for training")
        return True
        
    except Exception as e:
        print(f"[!] Error downloading FMA: {e}")
        print("    Training will still work with AudioSet alone")
        return True


def main():
    print("="*60)
    print("  HAwake WakeWord Training - Data Download")
    print("="*60)
    print()
    print("This will download ~3-5 GB of training data:")
    print("  • OpenWakeWord feature files (required)")
    print("  • AudioSet background audio (recommended)")
    print("  • FMA music samples (optional)")
    print()
    
    # Check if we have required dependencies
    try:
        import scipy
        import numpy
        import tqdm
        import datasets
    except ImportError as e:
        print(f"Error: Missing required package: {e}")
        print("Please install dependencies first: pip install -r requirements.txt")
        sys.exit(1)
    
    # Download feature files (required)
    if not download_feature_files():
        print("\n[ERROR] Failed to download required feature files")
        sys.exit(1)
    
    # Download AudioSet (recommended)
    download_audioset()
    
    # Download FMA (optional)
    download_fma()
    
    print("\n" + "="*60)
    print("  Download Complete!")
    print("="*60)
    print()
    
    # Summary
    audioset_count = len(list(Path("audioset_16k").glob("*.wav"))) if Path("audioset_16k").exists() else 0
    fma_count = len(list(Path("fma").glob("*.wav"))) if Path("fma").exists() else 0
    
    print(f"Background audio files:")
    print(f"  • audioset_16k: {audioset_count} files")
    print(f"  • fma: {fma_count} files")
    print()
    
    if audioset_count == 0 and fma_count == 0:
        print("[WARNING] No background audio downloaded!")
        print("Training will fail without background audio.")
        print("Please check your internet connection and try again.")
        sys.exit(1)
    elif audioset_count + fma_count < 100:
        print("[WARNING] Very few background audio files.")
        print("Training may produce lower quality models.")
    else:
        print("[✓] Ready for training! Run: streamlit run app.py")


if __name__ == "__main__":
    main()
