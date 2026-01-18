#!/usr/bin/env python3
"""
Patch torch_audiomentations for torchaudio 2.1+ compatibility.

In newer torchaudio versions (2.1+), these were deprecated and removed:
- torchaudio.set_audio_backend()
- torchaudio.USE_SOUNDFILE_LEGACY_INTERFACE
- torchaudio.info()

This script patches the torch_audiomentations package to work with modern torchaudio.
"""

import sys
from pathlib import Path


def get_patched_io_content():
    """Return the complete patched io.py content."""
    return '''"""
Audio I/O utilities for torch_audiomentations.
Patched for torchaudio 2.1+ compatibility by hawake-wakeword installer.
"""

import warnings
from pathlib import Path
from typing import Text, Union

import librosa
import torch
import torchaudio
from torch import Tensor

import math

# Try to import soundfile for audio metadata (replaces torchaudio.info)
try:
    import soundfile as sf
    HAS_SOUNDFILE = True
except ImportError:
    HAS_SOUNDFILE = False
    warnings.warn("soundfile not available - some features may not work")

AudioFile = Union[Path, Text, dict]
"""
Audio files can be provided to the Audio class using different types:
    - a "str" instance: "/path/to/audio.wav"
    - a "Path" instance: Path("/path/to/audio.wav")
    - a dict with a mandatory "audio" key (mandatory) and an optional "channel" key:
        {"audio": "/path/to/audio.wav", "channel": 0}
    - a dict with mandatory "samples" and "sample_rate" keys and an optional "channel" key:
        {"samples": (channel, time) torch.Tensor, "sample_rate": 44100}

The optional "channel" key can be used to indicate a specific channel.
"""

# Legacy interface settings - no longer used in torchaudio 2.1+
# These are kept as no-ops for backwards compatibility
# torchaudio.USE_SOUNDFILE_LEGACY_INTERFACE = False
# torchaudio.set_audio_backend("soundfile")


class Audio:
    """Audio IO with on-the-fly resampling

    Parameters
    ----------
    sample_rate: int
        Target sample rate.
    mono : int, optional
        Convert multi-channel to mono. Defaults to True.

    Usage
    -----
    >>> audio = Audio(sample_rate=16000)
    >>> samples = audio("/path/to/audio.wav")

    # on-the-fly resampling
    >>> original_sample_rate = 44100
    >>> two_seconds_stereo = torch.rand(2, 2 * original_sample_rate)
    >>> samples = audio({"samples": two_seconds_stereo, "sample_rate": original_sample_rate})
    >>> assert samples.shape[1] == 2 * 16000
    """

    @staticmethod
    def is_valid(file: AudioFile) -> bool:

        if isinstance(file, dict):

            if "samples" in file:

                samples = file["samples"]
                if len(samples.shape) != 2 or samples.shape[0] > samples.shape[1]:
                    raise ValueError(
                        "'samples' must be provided as a (channel, time) torch.Tensor."
                    )

                sample_rate = file.get("sample_rate", None)
                if sample_rate is None:
                    raise ValueError(
                        "'samples' must be provided with their 'sample_rate'."
                    )
                return True

            elif "audio" in file:
                return True

            else:
                # TODO improve error message
                raise ValueError("either 'audio' or 'samples' key must be provided.")

        return True

    @staticmethod
    def rms_normalize(samples: Tensor) -> Tensor:
        """Power-normalize samples

        Parameters
        ----------
        samples : (..., time) Tensor
            Single (or multichannel) samples or batch of samples

        Returns
        -------
        samples: (..., time) Tensor
            Power-normalized samples
        """
        rms = samples.square().mean(dim=-1, keepdim=True).sqrt()
        return samples / (rms + 1e-8)

    @staticmethod
    def get_audio_metadata(file_path) -> tuple:
        """Return (num_samples, sample_rate).
        
        Uses soundfile for compatibility with torchaudio 2.1+.
        Falls back to torchaudio.info if available (older versions).
        """
        # Try soundfile first (works with all torchaudio versions)
        if HAS_SOUNDFILE:
            try:
                with sf.SoundFile(str(file_path)) as f:
                    return f.frames, f.samplerate
            except Exception:
                pass  # Fall through to torchaudio fallback
        
        # Fallback to torchaudio.info (older torchaudio versions)
        if hasattr(torchaudio, 'info'):
            info = torchaudio.info(str(file_path))
            # Deal with backwards-incompatible signature change.
            # See https://github.com/pytorch/audio/issues/903 for more information.
            if type(info) is tuple:
                si, ei = info
                num_samples = si.length
                sample_rate = si.rate
            else:
                num_samples = info.num_frames
                sample_rate = info.sample_rate
            return num_samples, sample_rate
        
        raise RuntimeError(
            f"Cannot get audio metadata for {file_path}. "
            "Install soundfile: pip install soundfile"
        )

    def get_num_samples(self, file: AudioFile) -> int:
        """Number of samples (in target sample rate)

        :param file: audio file

        """

        self.is_valid(file)

        if isinstance(file, dict):

            # file = {"samples": torch.Tensor, "sample_rate": int, [ "channel": int ]}
            if "samples" in file:
                num_samples = file["samples"].shape[1]
                sample_rate = file["sample_rate"]

            # file = {"audio": str or Path, [ "channel": int ]}
            else:
                num_samples, sample_rate = self.get_audio_metadata(file["audio"])

        #  file = str or Path
        else:
            num_samples, sample_rate = self.get_audio_metadata(file)

        return math.ceil(num_samples * self.sample_rate / sample_rate)

    def __init__(self, sample_rate: int, mono: bool = True):
        super().__init__()
        self.sample_rate = sample_rate
        self.mono = mono

    def downmix_and_resample(self, samples: Tensor, sample_rate: int) -> Tensor:
        """Downmix and resample

        Parameters
        ----------
        samples : (channel, time) Tensor
            Samples.
        sample_rate : int
            Original sample rate.

        Returns
        -------
        samples : (channel, time) Tensor
            Remixed and resampled samples
        """

        # downmix to mono
        if self.mono and samples.shape[0] > 1:
            samples = samples.mean(dim=0, keepdim=True)

        # resample
        if self.sample_rate != sample_rate:
            samples = samples.numpy()
            if self.mono:
                # librosa expects mono audio to be of shape (n,), but we have (1, n).
                samples = librosa.core.resample(
                    samples[0], orig_sr=sample_rate, target_sr=self.sample_rate
                )[None]
            else:
                samples = librosa.core.resample(
                    samples.T, orig_sr=sample_rate, target_sr=self.sample_rate
                ).T

            samples = torch.tensor(samples)

        return samples

    def __call__(
        self, file: AudioFile, sample_offset: int = 0, num_samples: int = None
    ) -> Tensor:
        """

        Parameters
        ----------
        file : AudioFile
            Audio file.
        sample_offset : int, optional
            Start loading at this `sample_offset` sample. Defaults ot 0.
        num_samples : int, optional
            Load that many samples. Defaults to load up to the end of the file.

        Returns
        -------
        samples : (time, channel) torch.Tensor
            Samples

        """

        self.is_valid(file)

        if isinstance(file, dict):

            # file = {"samples": torch.Tensor, "sample_rate": int, [ "channel": int ]}
            if "samples" in file:

                samples = file["samples"]
                sample_rate = file["sample_rate"]
                num_channels = samples.shape[0]

                # channel = file.get("channel", None)
                # if channel is not None:
                #     samples = samples[channel - 1 : channel, :]
                # FIXME ^ if we implement this, be careful about the num_channels variable

                if sample_offset > 0:
                    samples = samples[:, sample_offset:]

                if num_samples is not None:
                    samples = samples[:, :num_samples]

                return self.downmix_and_resample(samples, sample_rate)

            # file = {"audio": str or Path, [ "channel": int ]}
            else:
                file = file["audio"]

        original_samples, sample_rate = torchaudio.load(
            file, frame_offset=sample_offset, num_frames=num_samples or -1
        )

        return self.downmix_and_resample(original_samples, sample_rate)


def read_audio(
    audio_file,
    sample_rate,
    start=None,
    stop=None
):
    """
    Read the audio from the file with the specified sample rate.
    Optionally read only a specified region.

    :param audio_file: Path to the audio file to read
    :param sample_rate: The target sample rate
    :param start: The start time (seconds)
    :param stop: The stop time (seconds)
    """
    audio = Audio(sample_rate=sample_rate, mono=True)

    original_num_samples, original_sample_rate = audio.get_audio_metadata(audio_file)
    original_duration = original_num_samples / original_sample_rate

    if start is None:
        start = 0.
    if stop is None:
        stop = original_duration

    assert stop <= original_duration
    sample_offset = int(start * original_sample_rate)
    num_samples = int((stop - start) * original_sample_rate)

    return audio(audio_file, sample_offset=sample_offset, num_samples=num_samples)
'''


def find_torch_audiomentations_io():
    """Find the torch_audiomentations io.py file in site-packages."""
    import site
    
    # Get all site-packages directories
    site_packages = site.getsitepackages()
    if hasattr(site, 'getusersitepackages'):
        user_site = site.getusersitepackages()
        if user_site:
            site_packages.append(user_site)
    
    # Also check in the current venv
    venv_path = Path(sys.prefix) / "Lib" / "site-packages"
    if venv_path.exists():
        site_packages.insert(0, str(venv_path))
    
    # Search for torch_audiomentations
    for sp in site_packages:
        io_path = Path(sp) / "torch_audiomentations" / "utils" / "io.py"
        if io_path.exists():
            return io_path
    
    return None


def patch_torch_audiomentations():
    """Patch torch_audiomentations io.py for torchaudio 2.1+ compatibility."""
    io_path = find_torch_audiomentations_io()
    
    if io_path is None:
        print("[!] torch_audiomentations not found - skipping patch")
        return False
    
    # Check if already patched
    content = io_path.read_text(encoding='utf-8')
    if "Patched for torchaudio 2.1+" in content:
        print(f"[✓] torch_audiomentations already patched: {io_path}")
        return True
    
    # Check if patch is needed - any of these indicate incompatibility with torchaudio 2.1+
    needs_patch = (
        "torchaudio.set_audio_backend" in content or
        "torchaudio.USE_SOUNDFILE_LEGACY_INTERFACE" in content or
        "torchaudio.info(" in content  # torchaudio.info() was removed in 2.1+
    )
    
    if not needs_patch:
        print(f"[✓] torch_audiomentations doesn't need patching: {io_path}")
        return True
    
    # Apply patch
    print(f"[*] Patching torch_audiomentations: {io_path}")
    
    # Backup original
    backup_path = io_path.with_suffix('.py.backup')
    if not backup_path.exists():
        backup_path.write_text(content, encoding='utf-8')
        print(f"    Backup saved to: {backup_path}")
    
    # Write patched content
    io_path.write_text(get_patched_io_content(), encoding='utf-8')
    print(f"[✓] Patched torch_audiomentations for torchaudio 2.1+ compatibility")
    
    return True


if __name__ == "__main__":
    success = patch_torch_audiomentations()
    sys.exit(0 if success else 1)
