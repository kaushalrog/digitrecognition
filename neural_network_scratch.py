from my_math import *
import random

class NeuralNetworkScratch:
    """Neural Network built from absolute scratch"""
    
    def __init__(self, input_size=784, hidden_size=128, output_size=10, learning_rate=0.1):
        print(f"\nInitializing Neural Network...")
        print(f"Architecture: {input_size} -> {hidden_size} -> {output_size}")
        print(f"Learning rate: {learning_rate}")
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.learning_rate = learning_rate
        
        # He initialization for weights
        std_w1 = sqrt(2.0 / input_size)
        std_w2 = sqrt(2.0 / hidden_size)
        
        self.W1 = create_random_normal(input_size, hidden_size, 0.0, std_w1)
        self.b1 = create_zeros(1, hidden_size)
        
        self.W2 = create_random_normal(hidden_size, output_size, 0.0, std_w2)
        self.b2 = create_zeros(1, output_size)
        
        print("✓ Weights initialized successfully!")
        
        # Training history
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'test_acc': []
        }
    
    def forward(self, X):
        """Forward propagation"""
        # Hidden layer: Z1 = X · W1 + b1
        self.Z1 = X.dot(self.W1)
        
        # Add bias (broadcast)
        for i in range(self.Z1.rows):
            for j in range(self.Z1.cols):
                self.Z1.data[i][j] += self.b1.data[0][j]
        
        # Apply ReLU activation
        self.A1 = self.Z1.apply_function(relu)
        
        # Output layer: Z2 = A1 · W2 + b2
        self.Z2 = self.A1.dot(self.W2)
        
        # Add bias
        for i in range(self.Z2.rows):
            for j in range(self.Z2.cols):
                self.Z2.data[i][j] += self.b2.data[0][j]
        
        # Apply Softmax
        self.A2 = self.softmax(self.Z2)
        
        return self.A2
    
    def softmax(self, Z):
        """Softmax activation"""
        result = Matrix(Z.rows, Z.cols)
        
        for i in range(Z.rows):
            # Find max for numerical stability
            max_val = Z.data[i][0]
            for j in range(1, Z.cols):
                if Z.data[i][j] > max_val:
                    max_val = Z.data[i][j]
            
            # Compute exp and sum
            exp_sum = 0.0
            exp_vals = []
            for j in range(Z.cols):
                exp_val = exp(Z.data[i][j] - max_val)
                exp_vals.append(exp_val)
                exp_sum += exp_val
            
            # Normalize
            for j in range(Z.cols):
                result.data[i][j] = exp_vals[j] / exp_sum
        
        return result
    
    def compute_loss(self, Y_pred, Y_true):
        """Cross-entropy loss"""
        m = Y_pred.rows
        total_loss = 0.0
        
        for i in range(m):
            for j in range(Y_pred.cols):
                if Y_true.data[i][j] == 1.0:
                    total_loss += -log(Y_pred.data[i][j])
        
        return total_loss / m
    
    def backward(self, X, Y_true):
        """Backpropagation"""
        m = X.rows
        
        # Output layer gradient: dZ2 = A2 - Y_true
        dZ2 = self.A2.subtract(Y_true)
        
        # dW2 = A1^T · dZ2 / m
        dW2 = self.A1.transpose().dot(dZ2)
        dW2 = dW2.multiply(1.0 / m)
        
        # db2 = sum(dZ2) / m
        db2 = dZ2.sum_axis_0()
        db2 = db2.multiply(1.0 / m)
        
        # Hidden layer gradient: dA1 = dZ2 · W2^T
        dA1 = dZ2.dot(self.W2.transpose())
        
        # dZ1 = dA1 * ReLU'(Z1)
        relu_derivative_Z1 = self.Z1.apply_function(relu_derivative)
        dZ1 = dA1.multiply(relu_derivative_Z1)
        
        # dW1 = X^T · dZ1 / m
        dW1 = X.transpose().dot(dZ1)
        dW1 = dW1.multiply(1.0 / m)
        
        # db1 = sum(dZ1) / m
        db1 = dZ1.sum_axis_0()
        db1 = db1.multiply(1.0 / m)
        
        # Update parameters
        self.W1 = self.W1.subtract(dW1.multiply(self.learning_rate))
        self.b1 = self.b1.subtract(db1.multiply(self.learning_rate))
        self.W2 = self.W2.subtract(dW2.multiply(self.learning_rate))
        self.b2 = self.b2.subtract(db2.multiply(self.learning_rate))
    
    def predict(self, X):
        """Make predictions"""
        Y_pred = self.forward(X)
        return Y_pred.argmax_axis_1()
    
    def accuracy(self, X, Y_true_labels):
        """Calculate accuracy"""
        predictions = self.predict(X)
        correct = 0
        for i in range(len(predictions)):
            if predictions[i] == Y_true_labels[i]:
                correct += 1
        return (correct / len(predictions)) * 100
    
    def train(self, X_train, Y_train_labels, Y_train_onehot, X_test, Y_test_labels, 
              epochs=10, batch_size=128):
        """Train the neural network"""
        n_samples = X_train.rows
        n_batches = n_samples // batch_size
        
        print("\n" + "="*70)
        print("TRAINING STARTED")
        print("="*70)
        print(f"Total samples: {n_samples}")
        print(f"Batch size: {batch_size}")
        print(f"Batches per epoch: {n_batches}")
        print(f"Epochs: {epochs}")
        print("-"*70)
        
        for epoch in range(epochs):
            # Shuffle data
            indices = list(range(n_samples))
            random.shuffle(indices)
            
            epoch_loss = 0.0
            
            # Mini-batch training
            for batch in range(n_batches):
                start_idx = batch * batch_size
                end_idx = start_idx + batch_size
                
                # Get batch indices
                batch_indices = indices[start_idx:end_idx]
                
                # Extract batch data
                X_batch = Matrix(batch_size, X_train.cols)
                Y_batch = Matrix(batch_size, Y_train_onehot.cols)
                
                for i, idx in enumerate(batch_indices):
                    for j in range(X_train.cols):
                        X_batch.data[i][j] = X_train.data[idx][j]
                    for j in range(Y_train_onehot.cols):
                        Y_batch.data[i][j] = Y_train_onehot.data[idx][j]
                
                # Forward pass
                Y_pred = self.forward(X_batch)
                
                # Compute loss
                loss = self.compute_loss(Y_pred, Y_batch)
                epoch_loss += loss
                
                # Backward pass
                self.backward(X_batch, Y_batch)
            
            # Calculate metrics
            avg_loss = epoch_loss / n_batches
            
            # Calculate accuracy (use subset for speed)
            train_subset_size = 10000
            X_train_subset = Matrix(train_subset_size, X_train.cols)
            train_labels_subset = []
            
            for i in range(train_subset_size):
                for j in range(X_train.cols):
                    X_train_subset.data[i][j] = X_train.data[i][j]
                train_labels_subset.append(Y_train_labels[i])
            
            train_acc = self.accuracy(X_train_subset, train_labels_subset)
            test_acc = self.accuracy(X_test, Y_test_labels)
            
            # Store history
            self.history['train_loss'].append(avg_loss)
            self.history['train_acc'].append(train_acc)
            self.history['test_acc'].append(test_acc)
            
            print(f"Epoch {epoch+1:2d}/{epochs} | Loss: {avg_loss:.4f} | "
                  f"Train Acc: {train_acc:5.2f}% | Test Acc: {test_acc:5.2f}%")
        
        print("-"*70)
        print("✓ TRAINING COMPLETED!")
        print("="*70)

print("✓ Neural Network module loaded!")