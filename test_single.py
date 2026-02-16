from data_loader import load_data
from neural_network_scratch import NeuralNetworkScratch
from my_math import Matrix
import matplotlib.pyplot as plt

def test_single_prediction():
    """Test the model on a single digit"""
    
    # Load data
    data_path = r'C:\Users\Kaushal - Personal\Downloads\DigitRecognition\data\MNIST'
    # Update this path to match your setup
    
    print("Loading data for testing...")
    X_train, Y_train_labels, Y_train, X_test, Y_test_labels, Y_test = load_data(data_path)
    
    # Create and train model (quick training)
    print("\nTraining model (quick training for demo)...")
    model = NeuralNetworkScratch(
        input_size=784,
        hidden_size=128,
        output_size=10,
        learning_rate=0.1
    )
    
    model.train(
        X_train, Y_train_labels, Y_train,
        X_test, Y_test_labels,
        epochs=5,  # Quick training
        batch_size=128
    )
    
    # Test on random image
    import random
    test_idx = random.randint(0, X_test.rows - 1)
    
    # Get single image
    X_single = Matrix(1, X_test.cols)
    for j in range(X_test.cols):
        X_single.data[0][j] = X_test.data[test_idx][j]
    
    # Get prediction
    prediction = model.predict(X_single)[0]
    true_label = Y_test_labels[test_idx]
    
    # Visualize
    image_data = [X_test.data[test_idx][j] for j in range(X_test.cols)]
    image = [[image_data[row*28 + col] for col in range(28)] for row in range(28)]
    
    plt.figure(figsize=(6, 6))
    plt.imshow(image, cmap='gray')
    color = 'green' if prediction == true_label else 'red'
    plt.title(f'Prediction: {prediction}\nTrue Label: {true_label}', 
              fontsize=16, fontweight='bold', color=color)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig('single_prediction.png', dpi=150)
    print(f"\n✓ Prediction saved to 'single_prediction.png'")
    print(f"  Predicted: {prediction}")
    print(f"  True:      {true_label}")
    print(f"  Result:    {'✓ CORRECT' if prediction == true_label else '✗ INCORRECT'}")
    plt.show()

if __name__ == "__main__":
    test_single_prediction()