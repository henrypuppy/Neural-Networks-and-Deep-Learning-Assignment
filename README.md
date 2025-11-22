# Neural Networks and Deep Learning Assignment

This repository contains a collection of assignments for a Neural Networks and Deep Learning course. The assignments cover fundamental concepts and implementations of various neural network architectures, from basic NumPy operations to advanced deep learning frameworks.

## Project Structure

```
Neural-Networks-and-Deep-Learning-Assignment/
│
├── Assignment1/              # NumPy Basics
├── Assignment2.1/            # Fully Connected Neural Networks (FNN)
├── Assignment2.2/            # ReLU Neural Networks for Function Fitting
├── Assignment2.3/            # Convolutional Neural Networks (CNN)
└── Assignment3/              # Recurrent Neural Networks (RNN)
```

## Assignments Overview

### Assignment 1: NumPy Tutorial

**Objective**: Learn the fundamentals of NumPy, a fundamental library for numerical computing in Python.

**Contents**:
- Basic NumPy operations and array manipulations
- Implementation of numerical computations from scratch

**Dataset**: None

**Requirements**: Complete the exercises specified in the Jupyter notebook following the instructions in the ipython file.

---

### Assignment 2.1: Fully Connected Neural Networks

**Objective**: Implement fully connected (feedforward) neural networks using NumPy and deep learning frameworks (TensorFlow/PyTorch).

**Contents**:
- Manual gradient computation using NumPy
- Automatic differentiation with TensorFlow and PyTorch
- MNIST handwritten digit classification

**Dataset**: 
- **MNIST**: 60,000 training images and 10,000 test images
  - A standard benchmark dataset for pattern recognition
  - Suitable for training complex models like deep CNNs
  - Small enough to run on laptop CPUs

**Files**:
- `tutorial_minst_fnn-numpy-exercise.ipynb`: NumPy implementation with manual backpropagation
- `tutorial_minst_fnn-tf2.0-exercise.ipynb`: TensorFlow 2.0 implementation
- `tf2.0-exercise.ipynb`: Additional TensorFlow exercises

**Requirements**: Complete all the marked sections in the `*.ipynb` files.

---

### Assignment 2.2: Function Fitting with ReLU Networks

**Objective**: Demonstrate the universal approximation capability of ReLU-based neural networks by fitting a custom-defined function.

**Key Concept**: Two-layer ReLU networks can approximate any continuous function (universal approximation theorem).

**Requirements**:
- Define a custom function
- Generate training and test sets by sampling from the function
- Train a ReLU-based neural network to fit the function
- Evaluate fitting performance on the test set
- Submit both code and report

**Framework Options**:
- TensorFlow, PyTorch, or Keras (standard implementation)
- NumPy only (additional 5 bonus points)

**Report Components**:
- Function definition
- Data collection methodology
- Model description
- Fitting results and visualizations

**References**:
- Cybenko (1989), Hornik et al. (1989), Leshno et al. (1993) - Universal approximation theorems
- Nair & Hinton (2010), Glorot et al. (2011) - ReLU activation functions

---

### Assignment 2.3: Convolutional Neural Networks

**Objective**: Implement convolutional neural networks for image classification on the MNIST dataset.

**Contents**:
- CNN architecture implementation
- Convolution and pooling operations
- Training and evaluation on MNIST

**Dataset**: 
- **MNIST**: Same as Assignment 2.1 (60,000 training, 10,000 test images)

**Files**:
- `CNN_tensorflow.ipynb`: TensorFlow implementation
  - Implement `conv2d()` and `max_pool_2x2()` functions
  - Complete 8 blanks for two-layer convolutional architecture
- `CNN_pytorch.ipynb`: PyTorch implementation
  - Configure `nn.Conv2d()` parameters for `self.conv1` and `self.conv2`
  - Complete the `x.view()` operation
- Tutorial notebooks for CIFAR-10 and various CNN architectures

**Requirements**:
- Complete the specified code sections
- Achieve test accuracy above 96%

---

### Assignment 3: Recurrent Neural Networks

**Objective**: Implement recurrent neural networks for Chinese poetry generation (Tang poetry).

**Task**: Generate classical Chinese poems using RNN/LSTM models trained on a collection of Tang poems.

**Contents**:
- RNN/LSTM implementation
- Character-level language modeling
- Text generation for poetry creation

**Dataset**: 
- **Tang Poetry** (`tangshi.txt`, `poems.txt`): Collection of classical Chinese poems

**Files**:
- `poem_generation_with_RNN-exercise.ipynb`: TensorFlow implementation
  - Complete 3 initial blanks
  - Implement poetry generation code
- `Learn2Carry-exercise.ipynb`: Additional RNN exercise
- `tangshi_for_pytorch/`: PyTorch implementation
  - `rnn.py`: Complete two code sections
  - `main.py`: Main training and generation script

**Generation Requirements**:
- Generate poems starting with specific keywords: "日", "红", "山", "夜", "湖", "海", "月"

**References**:
- Zhang & Lapata (2014). Chinese poetry generation with recurrent neural networks. EMNLP 2014.

---

## Technologies and Frameworks

- **Python 3.x**
- **NumPy**: Numerical computing and manual neural network implementation
- **TensorFlow 2.0**: Deep learning framework with automatic differentiation
- **PyTorch**: Deep learning framework with dynamic computation graphs
- **Keras**: High-level neural networks API
- **Jupyter Notebook**: Interactive development environment

## Datasets

1. **MNIST**: Handwritten digit classification (0-9)
   - 28×28 grayscale images
   - 10 classes
   - Available via TensorFlow/PyTorch datasets API

2. **Tang Poetry**: Classical Chinese poem collection
   - Used for sequence generation tasks
   - Included in Assignment 3 directory

## Getting Started

### Prerequisites

```bash
pip install numpy tensorflow torch jupyter matplotlib
```

### Running the Assignments

1. Clone this repository:
```bash
git clone <repository-url>
cd Neural-Networks-and-Deep-Learning-Assignment
```

2. Navigate to the desired assignment directory:
```bash
cd Assignment<X>.<Y>
```

3. Launch Jupyter Notebook:
```bash
jupyter notebook
```

4. Open and run the `.ipynb` files in order, completing the marked sections.

## Learning Outcomes

Through these assignments, students will:

- ✅ Understand the mathematical foundations of neural networks
- ✅ Implement neural networks from scratch using NumPy
- ✅ Master automatic differentiation in modern deep learning frameworks
- ✅ Build and train feedforward neural networks (FNN)
- ✅ Design convolutional neural networks (CNN) for image classification
- ✅ Implement recurrent neural networks (RNN) for sequence modeling
- ✅ Apply neural networks to real-world tasks (classification, regression, generation)
- ✅ Evaluate model performance and interpret results

## License

This repository is for educational purposes as part of a Neural Networks and Deep Learning course.

## Author

Student ID: 2252334
Name: 孙毓涵

---

**Note**: Each assignment directory contains its own README with specific instructions and requirements. Please refer to the individual assignment READMEs for detailed information.

