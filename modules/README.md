# Modules Directory

This directory contains the fundamental building blocks used to construct the larger models and discriminators. It is divided into subdirectories based on function.

## Subdirectories

  - **`discriminator/`**: Contains complete, stand-alone discriminator architectures.
  - **`generator/`**: Contains reusable neural network layers and blocks (e.g., attention, RNN, ConvNeXt blocks) used in the main generator models.
  - **`spectral_ops.py`**: Includes modules for spectral processing:
      - `Fourier`: A wrapper for `torch.stft` and `torch.istft`.
      - `Band`: A module to split a spectrogram into different frequency bands (e.g., mel scale) for processing and reassemble them.

# Discriminator Modules

This directory provides a suite of powerful, multi-component discriminators. The training script combines these into a single powerful ensemble discriminator. Each is designed to analyze audio from a different perspective (time, frequency, scale), making the generator's task more challenging and leading to higher-quality results.

## Files

### `MultiPeriodDiscriminator.py`

#### `MultiPeriodDiscriminator`

This discriminator operates on the raw audio waveform. It consists of several sub-discriminators, each viewing the input signal at a different *period*. For example, a sub-discriminator with `period=2` will reshape the audio into a 2D representation where adjacent samples are folded, allowing it to spot artifacts at that specific frequency. This is highly effective at detecting periodic artifacts.

**`__init__` Arguments:**

  - `nch` (`int`): Number of input channels (e.g., `1` for mono). Default: `1`.
  - `sample_rate` (`int`): Sample rate of the audio. Default: `48000`.
  - `periods` (`List[int]`): A list of periods for each sub-discriminator. Prime numbers are recommended. Default: `[2, 3, 5, 7, 11]`.
  - `norm` (`bool`): Whether to use spectral normalization. Default: `True`.

-----

### `MultiScaleDiscriminator.py`

#### `MultiScaleDiscriminator`

This discriminator also operates on the raw waveform. It contains multiple sub-discriminators that process the audio at different resolutions by downsampling the input. This allows it to identify artifacts at various time scales, from fine-grained details to broader structural issues.

**`__init__` Arguments:**

  - `sample_rate` (`int`): Sample rate of the audio.
  - `downsample_rates` (`List[int]`): A list of factors to downsample the audio for each sub-discriminator. Default: `[2, 4]`.
  - `nch` (`int`): Number of input channels. Default: `1`.
  - `norm` (`bool`): Whether to use spectral normalization. Default: `True`.

-----

### `MultiResolutionDiscriminator.py`

#### `MultiResolutionDiscriminator`

This discriminator operates in the spectral domain. It consists of several sub-discriminators, each analyzing the STFT of the input audio using a different window length. This allows it to detect spectral artifacts across different time-frequency resolutions.

**`__init__` Arguments:**

  - `nch` (`int`): Number of input channels. Default: `1`.
  - `sample_rate` (`int`): Sample rate of the audio. Default: `48000`.
  - `window_lengths` (`List[int]`): A list of STFT window lengths for each sub-discriminator. Default: `[2048, 1024, 512]`.
  - `hop_factor` (`float`): The ratio of hop length to window length. Default: `0.25`.
  - `bands` (`List[Tuple[float, float]]`): Frequency bands to analyze, specified as fractions of the Nyquist frequency.
  - `norm` (`bool`): Whether to use spectral normalization. Default: `True`.
  - `hidden_channels` (`int`): The number of hidden channels in the conv layers. Default: `32`.

-----

### `MultiFrequencyDiscriminator.py`

#### `MultiFrequencyDiscriminator`

This discriminator is similar to `MultiResolutionDiscriminator` but with a different internal architecture focused on capturing features across frequency bands. It also processes the real and imaginary parts of the STFT as separate channels. This discriminator references https://arxiv.org/abs/2210.13438's discriminator architecture.

**`__init__` Arguments:**

  - `nch` (`int`): Number of input channels.
  - `window_sizes` (`List[int]`): A list of STFT window sizes for each sub-discriminator.
  - `hidden_channels` (`int`): The number of base hidden channels. Default: `8`.
  - `sample_rate` (`int`): Sample rate of the audio. Default: `48000`.
  - `norm` (`bool`): Whether to use spectral normalization. Default: `True`.

-----

# Generator Modules

This directory contains reusable building blocks that form the core components of the main generator models in the `/models` directory.

## Files

### `RoFormerBlock.py`

#### `RoFormerBlock`

A standard Transformer block that uses **Ro**tary **P**osition **E**mbeddings (RoPE) instead of absolute or learned position embeddings. RoPE injects positional information by rotating the query and key vectors, which is particularly effective for sequence modeling. The block consists of a self-attention layer followed by an MLP, with residual connections and RMS normalization.

**`__init__` Arguments:**

  - `n_embd` (`int`): The embedding dimension (number of channels).
  - `n_head` (`int`): The number of attention heads.
  - `max_seq_len` (`int`): The maximum sequence length this block can handle, used to pre-compute the RoPE cache.
  - `rope_base` (`int`): The base value for the rotary position embedding calculation. Default: `10000`.

-----

### `AttentionRegisterRoFormerBlock.py`

#### `AttentionRegisterRoFormerBlock`

An extension of the `RoFormerBlock` that implements **Attention Registers**. This technique adds a small number of learnable "register" tokens to the sequence. These tokens act as a global memory or scratchpad for the attention mechanism, improving its ability to retain and access information across the entire sequence, especially when combined with a sliding window attention mechanism.

**`__init__` Arguments:**

  - *(Inherits from `RoFormerBlock`)*
  - `num_register_tokens` (`int`): The number of register tokens to prepend to the sequence. Default: `0`.
  - `window_size` (`int`): The size of the sliding attention window. If `-1`, full attention is used. Default: `-1`.

-----

### `RNNBlock.py`

#### `RNNBlock`

A block that uses a Recurrent Neural Network (RNN) layer followed by an MLP, with residual connections and RMS normalization. It uses a `GroupedRNN` internally.

**`__init__` Arguments:**

  - `n_embd` (`int`): The embedding dimension.
  - `n_layer` (`int`): The number of layers in the RNN.
  - `n_groups` (`int`): The number of parallel, smaller RNNs to use in the `GroupedRNN`. The embedding dimension is split across these groups.
  - `rnn_type` (`str`): The type of RNN cell to use, either `'gru'` or `'lstm'`. Default: `'gru'`.
  - `bidirectional` (`bool`): Whether to use a bidirectional RNN. Default: `False`.

-----

### `ConvNeXt1DBlock.py` & `ConvNeXt2DBlock.py`

#### `ConvNeXt1DBlock` / `ConvNeXt2DBlock`

Implementations of the ConvNeXt block for 1D and 2D data, respectively. This block is a modern, pure-convolutional architecture that adopts design principles from Vision Transformers. It features a depthwise convolution followed by pointwise convolutions (linear layers) in an inverted bottleneck structure. These blocks can be run in `'normal'` mode (downsampling) or `'transposed'` mode (upsampling).

**`__init__` Arguments:**

  - `kernel_size` (`int` or `tuple`): The kernel size for the depthwise convolution.
  - `stride` (`int` or `tuple`): The stride for the convolution, used for down/up-sampling.
  - `input_dim` (`int`): The number of input channels.
  - `output_dim` (`int`): The number of output channels.
  - `mode` (`str`): Operation mode, either `'normal'` for `ConvNd` or `'transposed'` for `ConvTransposeNd`. Default: `'normal'`.
