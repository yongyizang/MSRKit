# Music Source Restoration Kit 

This repository offers a collection of model implementations, training configurations, and evaluation scripts to help you quickly get started with training and evaluating music source restoration models.

We have designed the repository to be a GAN-based framework; to learn more about the GANs, you can watch [this video](https://www.youtube.com/watch?v=TpMIssRdhco).

## Directory Structure

The repository is organized to separate concerns, making it easy to extend and maintain. Click on a directory to learn more about its contents.

```
MSRKit/
├── README.md                 <- You are here
├── config.yaml               <- Main configuration file for experiments
├── train.py                  <- Main script to start training
├── unwrap.py                 <- Utility to extract generator weights from a checkpoint
│
├── data/                     <- [Data loading and augmentation](./data/README.md)
│
├── evaluation/               <- [Evaluation metrics](./evaluation/README.md)
│
├── losses/                   <- [Loss function implementations](./losses/README.md)
│
├── models/                   <- [Top-level generator model architectures](./models/README.md)
│
└── modules/                  <- [Core building blocks for models](./modules/README.md)
    ├── discriminator/        <- [Discriminator architectures](./modules/discriminator/README.md)
    └── generator/            <- [Reusable generator components](./modules/generator/README.md)
```

## 🚀 Getting Started

### 1. Setup

First, clone the repository and install the required dependencies.

```bash
git clone https://github.com/yongyizang/MSRKit.git
cd MSRKit
pip install -r requirements.txt
```

*Note: The `FAD_CLAP` metric requires `laion-clap`. Please install it via `pip install laion-clap`.*

### 2. Configure Your Experiment

Modify the `config.yaml` file to set up your dataset paths, model hyperparameters, and training settings.

Key sections to update:

  - `data.train_dataset.root_directory`: Path to your training data.
  - `data.train_dataset.file_list`: Path to a `.txt` file listing your training samples.
  - `data.val_dataset.root_directory`: Path to your validation data.
  - `data.val_dataset.file_list`: Path to a `.txt` file listing your validation samples.
  - `model`: Choose the generator model and its parameters.
  - `discriminators`: Add and configure one or more discriminators.
  - `trainer`: Set training parameters like `max_steps`, `devices` (GPU IDs), and `precision`.

### 3. Start Training

Launch the training process using the `train.py` script and your configuration file.

```bash
python train.py --config config.yaml
```

Logs, checkpoints, and audio samples will be saved in the `lightning_logs/` directory.

### 4. Unwrap Generator Weights

After training, you may want to use the generator model for inference without the rest of the Lightning module. The `unwrap.py` script extracts the generator's `state_dict` from a checkpoint file.

```bash
python unwrap.py --ckpt "path/to/your/checkpoint.ckpt" --out "path/to/save/generator.pth"
```

This creates a clean `.pth` file containing only the generator's weights. This is useful if you want to use the generator model for inference without the rest of the Lightning module, or if you want to fine-tune the generator model on a different dataset.

## Building Your First Model

To build your first model, you can reference the model architecture in the `models/` directory. You can also refer to the `modules/` directory for the building blocks used in the model architectures. At a very high level, we have implemented the following processing blocks:
- Spectral Operations: `Fourier`, `Band`
- Sequence Modeling Blocks: `RoFormerBlock` (and an example of modified attention pattern, `AttentionRegisterRoFormerBlock`), `RNNBlock`, `ConvNeXt1DBlock`
- Convolutional Blocks: `ConvNeXt2DBlock`, `ConvNeXt1DBlock`
- Discriminator Architectures: `MultiPeriodDiscriminator`, `MultiScaleDiscriminator`, `MultiResolutionDiscriminator`, `MultiFrequencyDiscriminator`

## ⚖️ License
This project is licensed under the MIT License.