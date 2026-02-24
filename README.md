# Digit Recognition using Neural Network (From Scratch)

A Python implementation of a handwritten digit recognition system built entirely from scratch. This project focuses on understanding the mathematical foundations of neural networks without relying on high-level deep learning frameworks.

The goal is educational clarity, architectural transparency, and strong fundamentals in machine learning implementation.

---

## Project Overview

This repository implements a fully connected neural network to classify handwritten digits from the MNIST dataset. All core components such as forward propagation, backpropagation, weight updates, and data handling are manually implemented.

This project demonstrates:

* Manual neural network implementation
* Matrix-based forward and backward propagation
* Gradient descent optimization
* Dataset handling and preprocessing
* Modular and readable Python structure

---

## Project Structure

```
digitrecognition/
│
├── data_loader.py              # Loads and preprocesses dataset
├── download_mnist.py           # Downloads MNIST dataset
├── find_data.py                # Utility to locate dataset files
├── my_math.py                  # Custom mathematical helper functions
├── neural_network_scratch.py   # Core neural network implementation
├── show_structure.py           # Displays model architecture details
├── train.py                    # Training script
├── test_single.py              # Test prediction for a single sample
├── README.md
└── .gitignore
```

---

## Technologies Used

* **Language:** Python
* **Numerical Computation:** NumPy
* **Dataset:** MNIST

No high-level ML frameworks such as TensorFlow or PyTorch are used. All learning logic is implemented manually.

---

## How It Works

1. MNIST dataset is downloaded and prepared.
2. Data is normalized and structured for training.
3. A feedforward neural network performs prediction.
4. Backpropagation computes gradients.
5. Weights are updated using gradient descent.

---

## Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/kaushalrog/digitrecognition.git
cd digitrecognition
```

### 2. Install Dependencies

```bash
pip install numpy
```

### 3. Download Dataset

```bash
python download_mnist.py
```

### 4. Train the Model

```bash
python train.py
```

### 5. Test a Single Sample

```bash
python test_single.py
```

---

## Learning Objectives

This project is ideal for understanding:

* Neural network internals
* Backpropagation mathematics
* Gradient descent optimization
* Matrix operations in ML
* Building ML systems without frameworks

---

## Future Improvements

* Add configurable hyperparameters
* Implement mini-batch gradient descent
* Add performance metrics visualization
* Export trained model weights
* Build a simple web interface for predictions

---

## License

This project is licensed under the MIT License.

---

## Author

Kaushal S
GitHub: h
