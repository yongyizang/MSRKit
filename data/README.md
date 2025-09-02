# Data Module

This directory contains all the necessary components for data loading, processing, and augmentation.

## Files

### `dataset.py`

This file defines the `RawStems` dataset class, which is the core of the data pipeline. It dynamically creates training examples by mixing a target stem with other stems based on a specified Signal-to-Noise Ratio (SNR).

#### `RawStems`

A PyTorch `Dataset` that loads and processes raw audio stems for music source restoration tasks.

**`__init__` Arguments:**

  - `target_stem` (`str`): The name of the target stem folder (e.g., `"Voc"` or `"Gtr_EG"`).
  - `root_directory` (`Union[str, Path]`): The root directory containing subfolders for each song.
  - `file_list` (`Optional[Union[str, Path]]`): Path to a `.txt` file where each line is a path to a song folder, relative to `root_directory`.
  - `sr` (`int`): The target sample rate to load audio at. Default: `44100`.
  - `clip_duration` (`float`): The duration of the audio clips to be extracted, in seconds. Default: `3.0`.
  - `snr_range` (`Tuple[float, float]`): A tuple representing the min and max SNR (in dB) for mixing the target stem with the noise (other stems). Default: `(0.0, 10.0)`.
  - `apply_augmentation` (`bool`): Whether to apply on-the-fly augmentations to the audio. Default: `True`.

### `augment.py`

This file implements the audio augmentation pipelines using the `pedalboard` library.

#### `StemAugmentation`

Applies a chain of augmentations suitable for the *target* audio source before it's mixed. This simulates variations in recording quality and effects.

  - **Effects include**: Random EQ, Resampling, Compression, Distortion, and Reverb.

#### `MixtureAugmentation`

Applies a chain of augmentations to the final *mixture* audio. This simulates artifacts that could occur on a fully mixed track.

  - **Effects include**: Limiting, Resampling, and MP3 a.k.a Codec compression.