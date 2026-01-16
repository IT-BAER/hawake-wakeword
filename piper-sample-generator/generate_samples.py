import argparse
import itertools as it
import logging
import os
import subprocess
import sys
import wave
import contextlib
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import webrtcvad

_DIR = Path(__file__).parent
_LOGGER = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

def remove_silence(
    x: bytes,
    frame_duration: float = 0.030,
    sample_rate: int = 16000,
    min_start: int = 2000,
) -> bytes:
    """Uses webrtc voice activity detection to remove silence from the clips"""
    vad = webrtcvad.Vad(0)
    
    # x is raw 16-bit PCM bytes
    # We need to process it in frames
    
    # Convert bytes to numpy int16 array for easier slicing/handling if needed, 
    # but webrtcvad expects bytes. 
    # Let's stick to the original logic which operated on numpy array.
    
    audio_data = np.frombuffer(x, dtype=np.int16)
    
    # Original logic:
    # x_new = x[0:min_start].tolist()
    # ...
    
    # We will just reimplement roughly what it did but dealing with file I/O outside.
    
    x_new = audio_data[0:min_start].tolist()
    step_size = int(sample_rate * frame_duration)
    
    for i in range(min_start, audio_data.shape[0] - step_size, step_size):
        chunk = audio_data[i : i + step_size]
        vad_res = vad.is_speech(chunk.tobytes(), sample_rate)
        if vad_res:
            x_new.extend(chunk.tolist())
            
    return np.array(x_new, dtype=np.int16).tobytes()

def generate_samples(
    text: Union[List[str], str],
    output_dir: Union[str, Path],
    max_samples: Optional[int] = None,
    file_names: Optional[List[str]] = None,
    model: Union[str, Path] = _DIR / "models" / "en_US-libritts_r-medium.pt",
    batch_size: int = 1,
    slerp_weights: Tuple[float, ...] = (0.5,),
    length_scales: Tuple[float, ...] = (0.75, 1, 1.25),
    noise_scales: Tuple[float, ...] = (0.667,),
    noise_scale_ws: Tuple[float, ...] = (0.8,),
    max_speakers: Optional[float] = None,
    verbose: bool = False,
    auto_reduce_batch_size: bool = False,
    min_phoneme_count: Optional[int] = None,
    **kwargs,
) -> None:
    
    # Resolve model path
    model_path = Path(model)
    if model_path.suffix == ".pt":
        # Try to find the onnx version
        onnx_path = model_path.with_suffix(".onnx")
        if onnx_path.exists():
            model_path = onnx_path
        else:
            # Fallback to checking if there is an onnx in the same dir with slightly different naming?
            # Or just assume the user provided a path that we should try to use as onnx if pt missing.
            # But here we assume we are replacing the pt usage.
            _LOGGER.warning(f"Requested .pt model {model_path} but we are using Piper CLI (ONNX). Checking for .onnx...")
            if not onnx_path.exists():
                 _LOGGER.error(f"Could not find ONNX model at {onnx_path}")
                 return
            model_path = onnx_path

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Prepare iterators
    if isinstance(text, str) and os.path.exists(text):
        texts = it.cycle(
            [
                i.strip()
                for i in open(text, "r", encoding="utf-8").readlines()
                if len(i.strip()) > 0
            ]
        )
        if max_samples is None:
             # This is tricky if it's a file and max_samples is None. 
             # We might default to number of lines.
             max_samples = sum(1 for line in open(text, "r", encoding="utf-8") if line.strip())
    elif isinstance(text, list):
        texts = it.cycle(text)
        if max_samples is None:
            max_samples = len(text)
    else:
        texts = it.cycle([text])
        if max_samples is None:
            max_samples = 1

    if file_names:
        file_names_iter = it.cycle(file_names)
    else:
        file_names_iter = None

    settings_iter = it.cycle(
        it.product(
            length_scales,
            noise_scales,
            noise_scale_ws,
        )
    )
    
    _LOGGER.info(f"Generating {max_samples} samples using Piper CLI...")
    
    # Use CPU for parallel generation to avoid CUDA context overhead per process
    # Piper is very fast on CPU for single sentences.
    import multiprocessing
    max_workers = multiprocessing.cpu_count()
    _LOGGER.info(f"Using {max_workers} CPU threads for generation.")

    from concurrent.futures import ThreadPoolExecutor, as_completed

    tasks = []
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for i in range(max_samples):
            current_text = next(texts)
            length_scale, noise_scale, noise_scale_w = next(settings_iter)
            
            if file_names_iter:
                fname = next(file_names_iter)
            else:
                fname = f"{i}.wav"
                
            out_path = output_dir / fname
            
            # Build command
            cmd = [
                sys.executable, "-m", "piper",
                "-m", str(model_path),
                "--output_file", str(out_path),
                "--length_scale", str(length_scale),
                "--noise_scale", str(noise_scale),
                "--noise_w_scale", str(noise_scale_w),
            ]
            
            # Submit task
            tasks.append(executor.submit(
                run_piper_generation, 
                cmd, 
                current_text, 
                out_path, 
                i
            ))
        
        # Wait for completion and log progress
        completed_count = 0
        for future in as_completed(tasks):
            completed_count += 1
            if completed_count % 100 == 0:
                _LOGGER.info(f"Generated {completed_count}/{max_samples} samples...")
            try:
                future.result()
            except Exception as e:
                _LOGGER.error(f"Task failed: {e}")

    _LOGGER.info("Generation complete.")

def run_piper_generation(cmd, text, out_path, index):
    """Helper to run a single piper process."""
    try:
        subprocess.run(
            cmd,
            input=text,
            text=True,
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
        
        # Post-process: Resample and Trim Silence
        if out_path.exists():
            process_audio(out_path)
    except subprocess.CalledProcessError as e:
        # _LOGGER.error(f"Piper failed for sample {index}: {e}")
        raise e


def process_audio(wav_path: Path):
    """Resamples to 16k and removes silence."""
    try:
        import scipy.io.wavfile
        import scipy.signal
    except ImportError:
        _LOGGER.warning("Scipy not found, skipping resampling/trimming. Install scipy!")
        return

    try:
        sr, audio = scipy.io.wavfile.read(str(wav_path))
        
        # Resample to 16000 if needed
        target_sr = 16000
        if sr != target_sr:
            # Calculate number of samples
            num_samples = int(len(audio) * target_sr / sr)
            audio = scipy.signal.resample(audio, num_samples).astype(np.int16)
            sr = target_sr
        
        # Ensure int16
        if audio.dtype != np.int16:
             # Normalize if float
             if audio.dtype == np.float32 or audio.dtype == np.float64:
                 audio = (audio / np.max(np.abs(audio)) * 32767).astype(np.int16)
        
        # Trim silence
        # remove_silence expects bytes
        trimmed_bytes = remove_silence(audio.tobytes(), sample_rate=sr)
        
        # Write back
        # We can use scipy or wave. Scipy write expects numpy array
        trimmed_audio = np.frombuffer(trimmed_bytes, dtype=np.int16)
        
        scipy.io.wavfile.write(str(wav_path), sr, trimmed_audio)
        
    except Exception as e:
        _LOGGER.error(f"Error processing {wav_path}: {e}")

if __name__ == "__main__":
    # Simplified main for testing
    parser = argparse.ArgumentParser()
    parser.add_argument("text")
    parser.add_argument("--max-samples", type=int, default=1)
    parser.add_argument("--output-dir", default="output")
    parser.add_argument("--model", default=str(_DIR / "models" / "en_US-libritts_r-medium.pt"))
    args, unknown = parser.parse_known_args()
    
    generate_samples(
        text=args.text,
        output_dir=args.output_dir,
        max_samples=args.max_samples,
        model=args.model
    )
