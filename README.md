# Handwritten Digit Recognition from Scratch

A neural network built from **scratch using pure Python** to recognize MNIST handwritten digits with **95-97% accuracy** — no NumPy, TensorFlow, or PyTorch!

## 🎯 Features

- ✅ Pure Python implementation (no ML libraries)
- ✅ Custom matrix operations from scratch
- ✅ Forward & backward propagation
- ✅ Mini-batch gradient descent
- ✅ 95-97% test accuracy on MNIST

## 🏗️ Architecture
```
Input (784) → Hidden (128, ReLU) → Output (10, Softmax)
```

## 🚀 Quick Start
```bash
# Clone repository
git clone https://github.com/kaushalrog/digitrecognition.git
cd digitrecognition

# Install matplotlib (only dependency)
pip install matplotlib

# Download MNIST data
python download_mnist.py

# Train model
python train.py
```

## 📊 Results

- **Training Accuracy**: ~96-98%
- **Test Accuracy**: ~95-97%
- **Training Time**: ~5-10 minutes (20 epochs)

## 📁 Files

- `my_math.py` - Custom matrix operations
- `data_loader.py` - MNIST loader
- `neural_network_scratch.py` - Neural network
- `train.py` - Training script

## 🧠 What's Inside

Built from scratch:
- Matrix class with all operations
- ReLU & Softmax activations
- Cross-entropy loss
- Backpropagation algorithm
- He weight initialization
- Mini-batch gradient descent

## 👨‍💻 Author

**Kaushal** - [GitHub](https://github.com/kaushalrog)

## 📝 License

MIT License

---

⭐ Star if you found this helpful!
