"""Check duration of generated clips and compute expected embedding frames."""
import os
import numpy as np
import scipy.io.wavfile
from pathlib import Path
import math

WORK_DIR = r"D:\VSC\hawake-wakeword\hee_schpustee"

# Check 50 positive clips
clips = list(Path(WORK_DIR + "/positive_train").glob("*.wav"))[:50]
durations_samples = []
for c in clips:
    sr, data = scipy.io.wavfile.read(c)
    durations_samples.append(len(data))

durations_samples = np.array(durations_samples)
print(f"Positive clip durations (samples @ 16kHz):")
print(f"  Min: {durations_samples.min()} = {durations_samples.min()/16000:.2f}s")
print(f"  Max: {durations_samples.max()} = {durations_samples.max()/16000:.2f}s")
print(f"  Median: {np.median(durations_samples):.0f} = {np.median(durations_samples)/16000:.2f}s")

median_dur = np.median(durations_samples)
total_length = int(round(median_dur / 1000) * 1000) + 12000
if total_length < 32000:
    total_length = 32000
elif abs(total_length - 32000) <= 4000:
    total_length = 32000
print(f"\n  Computed total_length: {total_length} samples = {total_length/16000:.2f}s")

# Compute expected embedding frames for this total_length
n_mel_frames = int(math.ceil(total_length / 160 - 3))
n_embedding_frames = (n_mel_frames - 76) // 8 + 1
print(f"  Expected embedding frames: {n_embedding_frames}")
print(f"\n  Actual NPY shape (from test file): (10000, 62, 96)")
print(f"\n  Android uses getFeatures(16, -1) -> 16 embedding frames")

# Compute total_length needed for 16 frames
# 16 = (n_mel - 76) // 8 + 1 → n_mel = 196 minimum
# n_mel = ceil(tl/160 - 3) = 197 → tl/160 = 200 → tl = 32000
print(f"\n  total_length needed for 16 frames: 32000 (2.0s)")
