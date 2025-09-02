# Evaluation Module

This directory contains classes for evaluating model performance during validation. All metrics inherit from a base `Metric` class for a consistent interface.

## Files

### `metrics.py`

#### `SI_SNR` (Scale-Invariant Signal-to-Noise Ratio)

A common metric for audio source separation that measures the quality of the restored signal relative to the original target. It is invariant to the overall scaling of the estimated signal.

  - `update(pred, target)`: Updates the running statistics with a new batch of predicted and target audio tensors.
  - `compute()`: Calculates the mean and standard deviation of the SI-SNR scores accumulated since the last reset.
  - `reset()`: Clears the accumulated statistics.

#### `FAD_CLAP` (Fréchet Audio Distance using CLAP)

Measures the Fréchet distance between the distributions of embeddings from the generated audio and the ground truth audio. It uses a pre-trained CLAP (Contrastive Language-Audio Pretraining) model to generate these embeddings, providing a perceptually relevant measure of audio quality and similarity.

**Note:** This metric requires the `laion-clap` library. If not installed, it will fall back to using random embeddings, which is not meaningful for evaluation.

  - `update(pred, target)`: Extracts CLAP embeddings from the predicted and target audio tensors and stores them.
  - `compute()`: Calculates the FAD score between the collected sets of embeddings.
  - `reset()`: Clears the stored embeddings.

**`__init__` Arguments:**

  - `embedding_dim` (`int`): The dimensionality of the embeddings. Should match the CLAP model. Default: `512`.
  - `model_name` (`str`): The name of the CLAP model architecture to use. Default: `'HTSAT-base'`.
  - `ckpt_path` (`Optional[str]`): Optional path to a specific CLAP model checkpoint. If `None`, it uses the default pre-trained weights.
