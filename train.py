from data_loader import load_data
from neural_network_scratch import NeuralNetworkScratch
import matplotlib.pyplot as plt

def plot_training_history(model):
    """Plot training metrics"""
    epochs = range(1, len(model.history['train_loss']) + 1)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot loss
    ax1.plot(epochs, model.history['train_loss'], 'b-', linewidth=2, label='Training Loss')
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title('Training Loss Over Time', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot accuracy
    ax2.plot(epochs, model.history['train_acc'], 'g-', linewidth=2, label='Train Accuracy')
    ax2.plot(epochs, model.history['test_acc'], 'r-', linewidth=2, label='Test Accuracy')
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Accuracy (%)', fontsize=12)
    ax2.set_title('Model Accuracy Over Time', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('training_history.png', dpi=150)
    print("\n📊 Training history saved to 'training_history.png'")
    plt.show()

def visualize_predictions(model, X_test, Y_test_labels, num_samples=10):
    """Visualize predictions"""
    import random
    
    indices = random.sample(range(X_test.rows), num_samples)
    
    fig, axes = plt.subplots(2, 5, figsize=(12, 6))
    axes = axes.ravel()
    
    for i, idx in enumerate(indices):
        # Get image
        image_data = [X_test.data[idx][j] for j in range(X_test.cols)]
        image = [[image_data[row*28 + col] for col in range(28)] for row in range(28)]
        
        # Get prediction
        X_single = Matrix(1, X_test.cols)
        for j in range(X_test.cols):
            X_single.data[0][j] = X_test.data[idx][j]
        
        prediction = model.predict(X_single)[0]
        true_label = Y_test_labels[idx]
        
        # Plot
        axes[i].imshow(image, cmap='gray')
        color = 'green' if prediction == true_label else 'red'
        axes[i].set_title(f'Pred: {prediction} | True: {true_label}', color=color, fontweight='bold')
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.savefig('predictions.png', dpi=150)
    print("🖼️  Predictions saved to 'predictions.png'")
    plt.show()

def main():
    """Main execution"""
    print("\n" + "="*70)
    print(" "*15 + "NEURAL NETWORK FROM ABSOLUTE SCRATCH")
    print(" "*10 + "Built with Pure Python - No NumPy/TensorFlow/PyTorch!")
    print("="*70)
    
    # Data is in the same directory
    data_path = r'data\MNIST'
    
    print(f"\nData path: {data_path}")
    
    # Load data
    try:
        X_train, Y_train_labels, Y_train, X_test, Y_test_labels, Y_test = load_data(data_path)
    except FileNotFoundError as e:
        print(f"\n❌ ERROR: Could not find MNIST data files!")
        print(f"Error details: {e}")
        print(f"\nPlease check that MNIST files are in the 'data' folder")
        return
    except Exception as e:
        print(f"\n❌ ERROR loading data: {e}")
        return
    
    # Create model
    model = NeuralNetworkScratch(
        input_size=784,
        hidden_size=128,
        output_size=10,
        learning_rate=0.1
    )
    
    # Train model
    model.train(
        X_train, Y_train_labels, Y_train,
        X_test, Y_test_labels,
        epochs=25,
        batch_size=64
    )
    
    # Final evaluation
    print("\nCalculating final accuracies on full datasets...")
    final_train_acc = model.accuracy(X_train, Y_train_labels)
    final_test_acc = model.accuracy(X_test, Y_test_labels)
    
    print(f"\n{'='*70}")
    print(" "*25 + "FINAL RESULTS")
    print(f"{'='*70}")
    print(f"  ✓ Final Training Accuracy: {final_train_acc:.2f}%")
    print(f"  ✓ Final Test Accuracy:     {final_test_acc:.2f}%")
    print(f"{'='*70}\n")
    
    # Visualizations
    print("Generating visualizations...")
    plot_training_history(model)
    visualize_predictions(model, X_test, Y_test_labels, 10)
    
    print("\n" + "="*70)
    print(" "*25 + "🎉 PROJECT COMPLETED SUCCESSFULLY! 🎉")
    print("="*70)
    print("\n📁 Output files created:")
    print("   ├── training_history.png (Loss and accuracy plots)")
    print("   └── predictions.png (Sample predictions visualization)")
    print("\n")


if __name__ == "__main__":
    from my_math import Matrix  # Import for visualize_predictions
    main()

if __name__ == "__main__":
    from my_math import Matrix  # Import for visualize_predictions
    main()
