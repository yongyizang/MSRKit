# Models Module

This directory contains the high-level generator architectures. These models define the main structure for transforming a mixed audio waveform into a restored target stem. They process the audio in the spectral domain and utilize various building blocks from the `modules/` directory.

All models (currently) first transform the input waveform into a spectrogram, process it in the time-frequency domain, and then convert it back to a waveform using an inverse STFT. They uniformly assumes a mono audio tensor being processed of shape [batch, samples].

## Files

### `MelRoFormer.py`

#### `MelRoFormer`

A dual-path Transformer-based model that applies attention alternately along the frequency and time axes of the spectrogram. It uses `RoFormerBlock`s, which incorporate Rotary Position Embeddings (RoPE) for effective sequence modeling. This model references https://arxiv.org/abs/2409.04702.

**`__init__` Arguments:**

  - `hidden_channels` (`int`): The number of channels (embedding dimension) used throughout the model.
  - `num_layers` (`int`): The number of layers (a time block + a frequency block is one layer).
  - `num_heads` (`int`): The number of attention heads in each RoFormer block.
  - `window_size` (`int`): The STFT window size.
  - `hop_size` (`int`): The STFT hop size.
  - `sample_rate` (`int`): The sample rate of the input audio.

-----

### `MelRNN.py`

#### `MelRNN`

A dual-path model similar to `MelRoFormer`, but it uses bidirectional GRUs (`RNNBlock`) instead of Transformers for processing the time and frequency axes. This can be a lighter-weight alternative to the attention-based models. This model references (yet deviates from) https://arxiv.org/abs/2209.15174.

**`__init__` Arguments:**

  - `hidden_channels` (`int`): The number of channels (embedding dimension).
  - `num_layers` (`int`): The number of RNN layers.
  - `num_groups` (`int`): The number of groups for the `GroupedRNN` within each `RNNBlock`.
  - `window_size` (`int`): The STFT window size.
  - `hop_size` (`int`): The STFT hop size.
  - `sample_rate` (`int`): The sample rate of the input audio.

-----

### `UNet.py`

#### `MelUNet`

A U-Net architecture that operates on the 2D spectrogram. It uses a series of downsampling and upsampling blocks (`ConvNeXt2DBlock`) with skip connections to capture multi-scale features in the spectrogram.

**`__init__` Arguments:**

  - `hidden_channels` (`int`): The initial number of channels in the network. Channel count doubles with each downsampling step.
  - `num_layers` (`int`): The depth of the U-Net (number of downsampling/upsampling stages).
  - `upsampling_factor` (`int`): The factor for upsampling/downsampling in each block (typically `2`).
  - `window_size` (`int`): The STFT window size.
  - `hop_size` (`int`): The STFT hop size.
  - `sample_rate` (`int`): The sample rate of the input audio.

-----
