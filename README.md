# Image-Classification-with-CNN-LSTM-Hybrid-vs-Standalone-CNN
This repository contains a comparative study between a Hybrid CNN-LSTM architecture and a Standalone Convolutional Neural Network (CNN) for image classification on the CIFAR-10 dataset. Both models are implemented using TensorFlow and Keras, employing Batch Normalization to stabilize and accelerate training.

## Project Overview

The goal of this project is to analyze the performance, training characteristics, and classification accuracy of incorporating temporal sequence modeling (LSTM) after spatial feature extraction (CNN) compared to a standard spatial feature extraction and fully connected pipeline (Standalone CNN).

## Dataset

- **Dataset**: CIFAR-10
- **Classes**: 10 distinct classes (airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck)
- **Image Size**: 32x32 pixels, RGB (3 channels)
- **Split**: 50,000 training images, 10,000 testing images
- **Preprocessing**: Feature scaling (pixel values normalized to the range [0, 1])

## Model Architectures

### 1. CNN-LSTM Hybrid Model

This model extracts spatial features through convolution and pooling layers, reshapes the spatial feature map into a sequential format, and processes the sequence using an LSTM layer before classification.

- **Input**: (32, 32, 3)
- **Conv2D**: 32 filters, 3x3 kernel, ReLU
- **Batch Normalization**
- **MaxPooling2D**: 2x2
- **Conv2D**: 64 filters, 3x3 kernel, ReLU
- **Batch Normalization**
- **MaxPooling2D**: 2x2 (Output shape: 6x6x64)
- **Reshape**: Reshapes the 6x6 spatial grid to a sequence of length 36 (36 time-steps, 64 features)
- **LSTM**: 64 units
- **Batch Normalization**
- **Dense**: 10 units, Softmax

### 2. Standalone CNN Model

This is a traditional Convolutional Neural Network that flattens spatial feature maps directly into fully connected layers.

- **Input**: (32, 32, 3)
- **Conv2D**: 32 filters, 3x3 kernel, ReLU
- **Batch Normalization**
- **MaxPooling2D**: 2x2
- **Conv2D**: 64 filters, 3x3 kernel, ReLU
- **Batch Normalization**
- **MaxPooling2D**: 2x2 (Output shape: 6x6x64)
- **Flatten**: Flattens the tensor to 2304 features
- **Dense**: 64 units, ReLU
- **Batch Normalization**
- **Dense**: 10 units, Softmax

## Training and Configuration

Both models were trained using identical hyperparameters for consistency and fair comparison:

- **Optimizer**: Adam
- **Loss Function**: Sparse Categorical Crossentropy
- **Metrics**: Accuracy
- **Batch Size**: 64
- **Epochs**: 5

## Experimental Results (5 Epochs)

### CNN-LSTM Hybrid Model

- **Training Accuracy (Epoch 5)**: ~70.9%
- **Validation Accuracy (Epoch 5)**: ~64.6%
- **Training Loss (Epoch 5)**: ~0.822
- **Validation Loss (Epoch 5)**: ~1.029

### Standalone CNN Model

- **Training Accuracy (Epoch 5)**: ~79.0%
- **Validation Accuracy (Epoch 5)**: ~70.1%
- **Training Loss (Epoch 5)**: ~0.597
- **Validation Loss (Epoch 5)**: ~0.883

## Key Observations

1. **Convergence Speed**: The Standalone CNN achieved higher validation accuracy (70.1%) in fewer epochs compared to the CNN-LSTM Hybrid (64.6%).
2. **Complexity and Overhead**: Flattening spatial dimensions directly to fully connected layers proved more efficient for training time per epoch on this specific 2D image task.
3. **Sequential Modeling for Images**: Treating the flattened 2D spatial grid as a 1D sequence for an LSTM presents a slower training curve. While LSTMs are useful for sequential/temporal patterns, standard CNNs remain highly effective and dominant for spatial 2D data like CIFAR-10.

## Requirements

- Python 3.x
- TensorFlow 2.x
- NumPy
