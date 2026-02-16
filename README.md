markdown# Handwritten Digit Recognition - Neural Network From Scratch



A fully connected neural network built \*\*completely from scratch\*\* using only pure Python to recognize handwritten digits from the MNIST dataset. No NumPy, TensorFlow, or PyTorch used for the neural network implementation!



\## 🎯 Project Overview



This project implements a neural network from absolute scratch to classify handwritten digits (0-9) from the MNIST dataset, achieving \*\*95-97% accuracy\*\* without using any deep learning frameworks.



\## ✨ Key Features



\- ✅ \*\*Pure Python Implementation\*\* - Neural network built from ground up

\- ✅ \*\*No NumPy\*\* - Custom matrix operations library

\- ✅ \*\*No TensorFlow/PyTorch\*\* - Complete backpropagation implementation

\- ✅ \*\*Custom Matrix Class\*\* - All linear algebra operations coded manually

\- ✅ \*\*Mini-batch Gradient Descent\*\* - Efficient training optimization

\- ✅ \*\*He Weight Initialization\*\* - Stable training convergence

\- ✅ \*\*ReLU + Softmax Activations\*\* - Modern activation functions

\- ✅ \*\*95-97% Test Accuracy\*\* - Competitive performance



\## 🏗️ ArchitectureInput Layer (784 neurons - 28x28 pixels)

↓

Hidden Layer (128 neurons - ReLU activation)

↓

Output Layer (10 neurons - Softmax activation)



\## 📁 Project StructureDigitRecognition/

├── my\_math.py                    # Custom matrix operations library

├── data\_loader.py                # MNIST dataset loader

├── neural\_network\_scratch.py     # Neural network implementation

├── train.py                      # Main training script

├── test\_single.py               # Single prediction testing

├── find\_data.py                 # Data location finder

├── README.md                    # Project documentation

└── requirements.txt             # Python dependencies



\## 🚀 Getting Started



\### Prerequisites



\- Python 3.7+

\- matplotlib (only for visualization)



\### Installation



1\. Clone the repository:

```bashgit clone https://github.com/kaushalrog/digitrecognition.git

cd digitrecognition



2\. Create virtual environment:

```bashpython -m venv venv

.\\venv\\Scripts\\Activate.ps1  # On Windows

source venv/bin/activate    # On Linux/Mac



3\. Install dependencies:

```bashpip install -r requirements.txt



4\. Download MNIST dataset:

&nbsp;  - Visit: http://yann.lecun.com/exdb/mnist/

&nbsp;  - Download all 4 files (.gz format)

&nbsp;  - Place in `data/MNIST/` folder



\### Running the Project

```bashTrain the neural network

python train.pyTest single predictions

python test\_single.pyFind MNIST data location

python find\_data.py



\## 📊 Results



\- \*\*Training Accuracy\*\*: 96-98%

\- \*\*Test Accuracy\*\*: 95-97%

\- \*\*Training Time\*\*: ~20 epochs (10-15 minutes on CPU)

\- \*\*Model Size\*\*: Lightweight (~100KB parameters)



\## 🧮 Implementation Details



\### Custom Components Built From Scratch:



1\. \*\*Matrix Class\*\*: Complete matrix operations

&nbsp;  - Matrix multiplication (dot product)

&nbsp;  - Transpose

&nbsp;  - Element-wise operations

&nbsp;  - Broadcasting support



2\. \*\*Activation Functions\*\*:

&nbsp;  - ReLU (Rectified Linear Unit)

&nbsp;  - Softmax (for output probabilities)

&nbsp;  - Derivatives for backpropagation



3\. \*\*Forward Propagation\*\*:

&nbsp;  - Linear transformations

&nbsp;  - Activation functions

&nbsp;  - Layer-wise computation



4\. \*\*Backpropagation\*\*:

&nbsp;  - Gradient computation

&nbsp;  - Chain rule implementation

&nbsp;  - Weight update mechanism



5\. \*\*Training Algorithm\*\*:

&nbsp;  - Mini-batch gradient descent

&nbsp;  - Cross-entropy loss

&nbsp;  - Data shuffling

&nbsp;  - Epoch-based training



\## 📚 Learning Outcomes



This project demonstrates deep understanding of:

\- Neural network fundamentals

\- Matrix-based computation

\- Gradient descent optimization

\- Backpropagation algorithm

\- Activation functions

\- Multi-class classification

\- Mini-batch training



\## 🛠️ Technologies Used



\- \*\*Python\*\* - Core programming language

\- \*\*Pure Python\*\* - Neural network implementation

\- \*\*matplotlib\*\* - Visualization only



\## 📈 Training OutputEpoch  1/20 | Loss: 0.4234 | Train Acc: 88.45% | Test Acc: 88.92%

Epoch  2/20 | Loss: 0.2156 | Train Acc: 93.21% | Test Acc: 93.54%

...

Epoch 20/20 | Loss: 0.0891 | Train Acc: 97.34% | Test Acc: 96.12%



\## 🎓 Mathematical Foundations



\### Forward Pass:Z1 = X·W1 + b1

A1 = ReLU(Z1)

Z2 = A1·W2 + b2

A2 = Softmax(Z2)



\### Loss Function:Loss = -1/m × Σ(Y\_true × log(Y\_pred))



\### Backpropagation:dZ2 = A2 - Y\_true

dW2 = A1ᵀ·dZ2 / m

dZ1 = (dZ2·W2ᵀ) ⊙ ReLU'(Z1)

dW1 = Xᵀ·dZ1 / m



\## 🤝 Contributing



Contributions are welcome! Feel free to:

\- Report bugs

\- Suggest features

\- Submit pull requests



\## 📝 License



This project is open source and available under the MIT License.



\## 👤 Author



\*\*Kaushal\*\*

\- GitHub: \[@kaushalrog](https://github.com/kaushalrog)



\## 🙏 Acknowledgments



\- MNIST Dataset by Yann LeCun

\- Neural Networks and Deep Learning concepts

\- Pure Python implementation challenge



\## 📧 Contact



For questions or feedback, please open an issue on GitHub.



---



⭐ If you found this project helpful, please give it a star!

