# Assignment 3 — Deep Learning (MLP · CNN · RNN · Transformer)

**Framework:** PyTorch

This part contains the implementation and analysis for Assignment 3, focusing on modern deep learning architectures.

The goal of this assignment is not to train a large number of models, but to understand how different neural network architectures, optimization methods, and regularization techniques affect model performance and generalization.

The notebook covers:

- Multilayer Perceptrons (MLPs)
- Convolutional Neural Networks (CNNs)
- Recurrent Neural Networks (RNNs)
- LSTM and GRU architectures
- Transformer-based models

---

# How to Run

Install the required packages:

```bash
pip install torch torchvision medmnist matplotlib seaborn scikit-learn pandas numpy
```

Then run the notebook from beginning to end.

The experiments are designed to be reproducible:

- Random seeds are fixed (`SEED = 42`)
- Data preprocessing is performed only using the training split
- Shared training utilities are reused across experiments
- Each experiment changes specific variables while keeping other conditions fixed

GPU is recommended for CNN and Transformer experiments, but most experiments can also run on CPU.

---

# Dataset Summary

| Part | Dataset | Type | Task |
|---|---|---|---|
| Part 1 — MLP | Breast Cancer Wisconsin Dataset | Tabular data | Binary Classification + Regression experiments |
| Part 2 — CNN | BloodMNIST (MedMNIST) | Image data | Multi-class image classification |
| Part 3 — RNN/LSTM/GRU | ECG5000 Dataset | Time-series sequence data | Sequence classification |
| Part 4 — Transformer | ECG5000 Dataset | Time-series sequence data | Sequence classification |
| Part 5 — Research Review | — | — | Industry analysis of ML models |

---

# Notebook Structure

| Part | Topic | Dataset | Description |
|---|---|---|---|
| 1 | MLP — Deep Learning Foundations | Breast Cancer Wisconsin Dataset | Fully connected neural networks for classification and regression. Includes optimizer comparison, learning rate experiments, batch size analysis, architecture tuning, activation functions, initialization methods, BatchNorm, regularization, early stopping, and gradient clipping. |
| 2 | CNN — Image Modeling | BloodMNIST | CNN architecture experiments including kernel size, stride, number of filters, pooling strategies, depth, augmentation, and transfer learning with ResNet18. |
| 3 | RNN — Sequence Modeling | ECG5000 | Comparison of Vanilla RNN, LSTM, and GRU models. Includes experiments with sequence length, hidden size, number of layers, bidirectional networks, and dropout. |
| 4 | Transformer Models | ECG5000 | Transformer Encoder implementation using PyTorch attention layers and comparison with recurrent models. |
| 5 | Industry Review | — | Research review about commonly used ML models in industry and future trends. |

---

# Part 1 — MLP Experiments

## Dataset

**Breast Cancer Wisconsin Dataset**

A tabular dataset used for binary classification. Additional regression-style experiments were also performed to study neural network behavior under different settings.

## Experiments

### Optimization

- SGD
- SGD with momentum
- Adam
- Learning rate comparison
- Learning rate scheduling
- Batch size effects
- Early stopping

### Architecture

Different network designs were tested:

- Number of hidden layers
- Number of neurons per layer

Activation functions:

- ReLU
- LeakyReLU
- Tanh
- Sigmoid
- GELU

### Initialization and Stability

- Random initialization
- Xavier initialization
- He initialization
- Batch Normalization

### Regularization

- Dropout
- L1 regularization
- L2 regularization
- Activity regularization
- Gradient clipping

The experiments demonstrate how increasing capacity can improve representation ability but can also increase overfitting risk.

---

# Part 2 — CNN Experiments

## Dataset

**BloodMNIST (MedMNIST)**

An image classification dataset containing blood cell images.

## Experiments

The CNN experiments analyze:

- Convolution kernel size
- Stride
- Number of filters
- Pooling strategies:
  - Max pooling
  - Average pooling
- Pooling window size
- Network depth

Evaluation includes:

- Accuracy
- Loss curves
- Confusion matrix
- Misclassified samples

## Data Augmentation

Augmentation techniques were applied to study their effect on generalization and overfitting.

## Transfer Learning

A pretrained **ResNet18** model was used:

- Feature extraction
- Fine tuning

The results show how pretrained visual features can improve performance when adapted to a target dataset.

---

# Part 3 — RNN / LSTM / GRU Experiments

## Dataset

**ECG5000**

A time-series dataset containing ECG signal sequences.

## Implemented Models

- Vanilla RNN
- LSTM
- GRU

## Experiments

- Sequence length
- Hidden layer size
- Number of recurrent layers
- Bidirectional RNNs
- Dropout

The results demonstrate why gated models such as LSTM and GRU are generally better at learning long-term dependencies compared to standard RNNs.

---

# Part 4 — Transformer Experiments

## Dataset

**ECG5000**

A Transformer Encoder model was implemented using PyTorch.

## Experiments

The analysis focuses on:

- Self-attention
- Multi-head attention
- Positional encoding
- Comparison with recurrent models

Transformers can model long-range relationships more directly through attention mechanisms, but usually require more computational resources and larger datasets.

---

# Part 5 — Industry Model Research Review

This section studies which machine learning models are commonly used in real-world applications.

## Classical Machine Learning

Discussed models:

- Logistic Regression
- Linear Regression
- Decision Trees
- Random Forest
- Gradient Boosting models:
  - XGBoost
  - LightGBM
  - CatBoost

## Deep Learning

Discussed approaches:

- CNNs
- Transformers
- Large Language Models

The conclusion is that there is no universally best model.

Classical models remain important for structured/tabular data because they are fast, efficient, and interpretable.

Deep learning models dominate areas such as:

- Computer vision
- Natural language processing
- Speech
- Multimodal AI

---

# Main Takeaway

This assignment shows that choosing a machine learning model depends on the problem and the type of data.

Different architectures have different strengths:

- **MLPs** → general tabular learning
- **CNNs** → spatial patterns in images
- **RNN/LSTM/GRU** → sequential and time-series data
- **Transformers** → attention-based modeling of complex dependencies

Good machine learning practice is not only about using larger models, but about selecting the right architecture and understanding its limitations.
