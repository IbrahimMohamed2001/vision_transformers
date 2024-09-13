# Vision Transformer (ViT) for Oxford-IIIT Pet Dataset

This repository contains an implementation of a Vision Transformer (ViT) model for image classification on the Oxford-IIIT Pet dataset. The project includes data loading, model training, validation, and logging using TensorBoard.

## Table of Contents

- [Vision Transformer (ViT) for Oxford-IIIT Pet Dataset](#vision-transformer-vit-for-oxford-iiit-pet-dataset)
  - [Table of Contents](#table-of-contents)
  - [Installation](#installation)
  - [Usage](#usage)
    - [Training the Model](#training-the-model)
    - [Viewing TensorBoard Logs](#viewing-tensorboard-logs)
    - [Displaying Sample Images](#displaying-sample-images)
  - [Model Architecture](#model-architecture)
  - [Training and Validation](#training-and-validation)
  - [Results](#results)
  - [Acknowledgements](#acknowledgements)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/IbrahimMohamed2001/vision_transformers.git
cd vision_transformers
```

2. Install the required packages:
```bash
pip install -r requirements.txt
```

## Usage

### Training the Model

To train the model, run:

```bash
python train.py
```

### Viewing TensorBoard Logs

To view the training and validation logs using TensorBoard, run:

```bash
tensorboard --logdir=./logs
```

### Displaying Sample Images

To display sample images from the dataset, run:

```bash
python dataset.py
```


## Model Architecture

The Vision Transformer (ViT) model is implemented in `model.py`. The key components include:

- **Patch Embedding**: Converts image patches into embedding vectors.
- **Attention**: Multi-head self-attention mechanism.
- **FeedForward**: Fully connected feed-forward network.
- **ResidualAdd**: Adds residual connections.
- **PreNorm**: Applies layer normalization before the main function.
- **ViT**: Combines all components into a complete Vision Transformer model.

## Training and Validation

The training and validation scripts are implemented in `train.py`. Key features include:

- **Data Loading**: Uses `dataset.py` to load and preprocess the Oxford-IIIT Pet dataset.
- **Metrics**: Computes precision, recall, and F1 score using `torchmetrics`.
- **Checkpointing**: Saves model checkpoints after each epoch.
- **Logging**: Logs training and validation metrics to TensorBoard.

## Results

After training, the model's performance can be evaluated using the validation metrics logged in TensorBoard. The metrics include loss, accuracy, precision, recall, and F1 score.

## Acknowledgements

- The Vision Transformer (ViT) model is inspired by the paper [An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](https://arxiv.org/abs/2010.11929).
- The dataset used is the [Oxford-IIIT Pet Dataset](https://www.robots.ox.ac.uk/~vgg/data/pets/).