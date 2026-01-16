"""
Patch for torchaudio compatibility with PyTorch nightly and torchaudio 2.1+

In newer torchaudio versions (2.1+), the following were deprecated and removed:
- torchaudio.set_audio_backend() 
- torchaudio.USE_SOUNDFILE_LEGACY_INTERFACE
- torchaudio.info()

This patch adds back stub implementations to allow torch-audiomentations 0.11.0
to work with modern torchaudio versions.

Usage:
    import patch_torchaudio  # Must be imported BEFORE torch_audiomentations
    import torch_audiomentations
"""

import torchaudio
import warnings


def _patch_set_audio_backend():
    """Add back set_audio_backend as a no-op for compatibility."""
    if not hasattr(torchaudio, 'set_audio_backend'):
        def set_audio_backend(backend):
            """Stub for deprecated set_audio_backend - does nothing in modern torchaudio."""
            pass
        torchaudio.set_audio_backend = set_audio_backend


def _patch_use_soundfile_legacy():
    """Add back USE_SOUNDFILE_LEGACY_INTERFACE attribute."""
    if not hasattr(torchaudio, 'USE_SOUNDFILE_LEGACY_INTERFACE'):
        torchaudio.USE_SOUNDFILE_LEGACY_INTERFACE = False


def _patch_torchaudio_info():
    """Add back torchaudio.info() using soundfile for metadata."""
    if not hasattr(torchaudio, 'info'):
        try:
            import soundfile as sf
            from collections import namedtuple
            
            # Create a simple AudioMetaData-like object
            AudioMetaData = namedtuple('AudioMetaData', [
                'sample_rate', 'num_frames', 'num_channels', 'bits_per_sample', 'encoding'
            ])
            
            def info(filepath):
                """Get audio file metadata using soundfile."""
                with sf.SoundFile(str(filepath)) as f:
                    return AudioMetaData(
                        sample_rate=f.samplerate,
                        num_frames=f.frames,
                        num_channels=f.channels,
                        bits_per_sample=16 if f.subtype == 'PCM_16' else 32,
                        encoding=f.subtype or 'UNKNOWN'
                    )
            
            torchaudio.info = info
        except ImportError:
            warnings.warn(
                "soundfile not available for torchaudio.info() patch. "
                "Install with: pip install soundfile"
            )


def apply_patches():
    """Apply all torchaudio compatibility patches."""
    _patch_set_audio_backend()
    _patch_use_soundfile_legacy()
    _patch_torchaudio_info()


# Auto-apply patches on import
apply_patches()
