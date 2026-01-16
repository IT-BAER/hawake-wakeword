import os
import shutil
from pathlib import Path
from huggingface_hub import snapshot_download
import scipy.io.wavfile
import numpy as np
from tqdm import tqdm

def download_and_process_rirs():
    output_dir = Path("mit_rirs")
    if output_dir.exists() and any(output_dir.iterdir()):
        print("mit_rirs directory exists and is not empty. Skipping download.")
        return

    print("Downloading MIT Environmental Impulse Responses...")
    # download to cache
    repo_path = snapshot_download(repo_id="davidscripka/MIT_environmental_impulse_responses", repo_type="dataset")
    
    print(f"Processing audio files from {repo_path}...")
    output_dir.mkdir(exist_ok=True)
    
    repo_path = Path(repo_path)
    # The dataset has a 16khz folder?
    source_dir = repo_path / "16khz"
    if not source_dir.exists():
        # Fallback or check structure
        source_dir = repo_path
    
    # Files might be scattered or in a subfolder.
    # The colab script did: Path("./MIT_environmental_impulse_responses/16khz").glob("*.wav")
    
    wav_files = list(source_dir.glob("**/*.wav"))
    if not wav_files:
        print("No wav files found in downloaded dataset!")
        return

    print(f"Found {len(wav_files)} wav files. converting/copying...")
    
    for wav_file in tqdm(wav_files):
        # Read and ensure 16khz int16
        try:
            sr, audio = scipy.io.wavfile.read(wav_file)
            if sr != 16000:
                 # Resample logic if needed, but the folder says 16khz
                 # For now assuming it is correct or close enough
                 pass
            
            # Normalize/Convert to int16 if float
            if audio.dtype == np.float32:
                 audio = (audio * 32767).astype(np.int16)
            
            out_path = output_dir / wav_file.name
            scipy.io.wavfile.write(out_path, 16000, audio)
        except Exception as e:
            print(f"Error processing {wav_file}: {e}")
            
    print("Done.")

if __name__ == "__main__":
    download_and_process_rirs()
