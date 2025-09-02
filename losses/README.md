# Losses Module

This directory contains the implementations of various loss functions used for training the generator and discriminators.

## Files

### `gan_loss.py`

This file implements adversarial losses for both the generator and discriminator, as well as a feature matching loss.

We provide both LSGAN and Hinge GAN implementations. LSGAN and Hinge GAN differ primarily in how they penalize mistakes. 

- LSGAN uses a "least squares" approach that constantly pushes fake samples toward looking real, with the penalty growing quadratically the further off they are - this means even terrible fakes get strong learning signals, preventing vanishing gradients, but the discriminator never stops pushing even on samples that are already good enough, which can cause instability. 
- Hinge GAN instead creates a "satisfaction zone" where once the discriminator is confident enough about a sample (real or fake), it stops trying to improve its classification - this focuses all the learning on ambiguous samples near the decision boundary. The result: LSGAN provides consistent gradients throughout training but can overshoot and destabilize, while Hinge GAN typically produces sharper images by not wasting effort on already-separated samples, though it risks killing gradients entirely if the discriminator gets too confident too fast.

#### `GeneratorLoss`

Calculates the adversarial loss for the generator, encouraging it to produce outputs that the discriminator classifies as real.

**`__init__` Arguments:**

  - `gan_type` (`str`): The type of GAN loss to use. Supports `'hinge'` and `'lsgan'` (Least Squares GAN). Default: `'hinge'`.

#### `DiscriminatorLoss`

Calculates the adversarial loss for the discriminator, training it to distinguish between real and fake (generated) inputs.

**`__init__` Arguments:**

  - `gan_type` (`str`): The type of GAN loss to use. Supports `'hinge'` and `'lsgan'`. Default: `'hinge'`.

#### `FeatureMatchingLoss`

Calculates the L1 distance between the feature maps of the real and fake inputs from the intermediate layers of the discriminator. This helps stabilize training by matching the statistical properties of the features.

-----

### `reconstruction_loss.py`

This file implements reconstruction losses that measure the direct difference between the generated audio and the ground truth target audio in various domains.

#### `MultiMelSpecReconstructionLoss`

Calculates the L1 loss between the log-mel spectrograms of the predicted and target audio. It computes this loss using multiple different STFT configurations (FFT size, hop length, mel bands) and averages the results for a more robust, multi-resolution spectral loss.

**`__init__` Arguments:**

  - `sample_rate` (`int`): The sample rate of the audio.
  - `n_fft` (`List[int]`): A list of FFT sizes for the different STFT resolutions.
  - `hop_length` (`List[int]`): A list of hop lengths corresponding to the FFT sizes.
  - `n_mels` (`List[int]`): A list of the number of mel bands corresponding to the FFT sizes.

#### `ComplexSpecReconstructionLoss`

Calculates the L1 loss on the magnitude of the complex spectrograms.

#### `MultiComplexSpecReconstructionLoss`

A multi-resolution version of `ComplexSpecReconstructionLoss`.

#### `WaveformReconstructionLoss`

Calculates a simple L1 loss directly on the raw audio waveforms.
