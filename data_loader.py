import gzip
import struct
import os
from my_math import Matrix

def load_mnist_images(filename):
    """Load MNIST images from .gz file"""
    print(f"Loading images from: {filename}")
    
    # Try .gz file first, then try without .gz extension
    if not os.path.exists(filename):
        filename = filename.replace('.gz', '')
    
    if filename.endswith('.gz'):
        opener = gzip.open
    else:
        opener = open
    
    with opener(filename, 'rb') as f:
        # Read header
        magic, num_images, rows, cols = struct.unpack('>IIII', f.read(16))
        
        if magic != 2051:
            raise ValueError(f'Invalid magic number {magic} in image file')
        
        print(f"  Images: {num_images}, Size: {rows}x{cols}")
        
        # Read image data
        images = []
        for i in range(num_images):
            if (i + 1) % 10000 == 0:
                print(f"  Loading image {i+1}/{num_images}...")
            
            image = []
            for _ in range(rows * cols):
                pixel = struct.unpack('B', f.read(1))[0]
                image.append(pixel / 255.0)  # Normalize to [0, 1]
            images.append(image)
        
        print(f"  ✓ Loaded {num_images} images")
        return images, rows, cols

def load_mnist_labels(filename):
    """Load MNIST labels from .gz file"""
    print(f"Loading labels from: {filename}")
    
    # Try .gz file first, then try without .gz extension
    if not os.path.exists(filename):
        filename = filename.replace('.gz', '')
    
    if filename.endswith('.gz'):
        opener = gzip.open
    else:
        opener = open
    
    with opener(filename, 'rb') as f:
        # Read header
        magic, num_labels = struct.unpack('>II', f.read(8))
        
        if magic != 2049:
            raise ValueError(f'Invalid magic number {magic} in label file')
        
        # Read labels
        labels = []
        for _ in range(num_labels):
            label = struct.unpack('B', f.read(1))[0]
            labels.append(label)
        
        print(f"  ✓ Loaded {num_labels} labels")
        return labels

def one_hot_encode(labels, num_classes=10):
    """Convert labels to one-hot encoding"""
    print(f"One-hot encoding {len(labels)} labels...")
    one_hot = []
    for label in labels:
        encoding = [0.0] * num_classes
        encoding[label] = 1.0
        one_hot.append(encoding)
    print(f"  ✓ Encoded to {len(one_hot)}x{num_classes} matrix")
    return one_hot

def load_data(data_path='data'):
    """Load all MNIST data"""
    print("\n" + "="*70)
    print("LOADING MNIST DATASET")
    print("="*70)
    
    import os
    
    # Function to find actual file
    def find_file(base_path, filename):
        # Try direct path
        direct = os.path.join(base_path, filename)
        if os.path.isfile(direct):
            return direct
        
        # Try with .gz
        gz_file = direct + '.gz'
        if os.path.isfile(gz_file):
            return gz_file
        
        # Try inside a subfolder with same name (without extension)
        folder_name = filename.replace('.gz', '')
        folder_path = os.path.join(base_path, folder_name)
        if os.path.isdir(folder_path):
            # Look for files inside this folder
            for file in os.listdir(folder_path):
                return os.path.join(folder_path, file)
        
        # Try one level up with .gz or without
        without_ext = filename.replace('.gz', '').replace('-idx3-ubyte', '').replace('-idx1-ubyte', '')
        for root, dirs, files in os.walk(base_path):
            for file in files:
                if filename in file or without_ext in file:
                    return os.path.join(root, file)
        
        return None
    
    # Find all 4 MNIST files
    print("Searching for MNIST files...")
    
    train_images_file = find_file(data_path, 'train-images-idx3-ubyte.gz')
    if not train_images_file:
        train_images_file = find_file(data_path, 'train-images-idx3-ubyte')
    
    train_labels_file = find_file(data_path, 'train-labels-idx1-ubyte.gz')
    if not train_labels_file:
        train_labels_file = find_file(data_path, 'train-labels-idx1-ubyte')
    
    test_images_file = find_file(data_path, 't10k-images-idx3-ubyte.gz')
    if not test_images_file:
        test_images_file = find_file(data_path, 't10k-images-idx3-ubyte')
    
    test_labels_file = find_file(data_path, 't10k-labels-idx1-ubyte.gz')
    if not test_labels_file:
        test_labels_file = find_file(data_path, 't10k-labels-idx1-ubyte')
    
    # Check if all files found
    if not all([train_images_file, train_labels_file, test_images_file, test_labels_file]):
        print("\n❌ Could not find all MNIST files!")
        print(f"Train images: {train_images_file}")
        print(f"Train labels: {train_labels_file}")
        print(f"Test images: {test_images_file}")
        print(f"Test labels: {test_labels_file}")
        raise FileNotFoundError("Missing MNIST files")
    
    print(f"✓ Found all MNIST files")
    
    # Load training data
    train_images, rows, cols = load_mnist_images(train_images_file)
    train_labels = load_mnist_labels(train_labels_file)
    
    # Load test data
    test_images, _, _ = load_mnist_images(test_images_file)
    test_labels = load_mnist_labels(test_labels_file)
    
    print(f"\nDataset Summary:")
    print(f"  Training samples: {len(train_images)}")
    print(f"  Test samples: {len(test_images)}")
    print(f"  Image dimensions: {rows}x{cols} = {rows*cols} pixels")
    
    # Convert to matrices
    print("\nConverting to matrices...")
    X_train = Matrix.from_list(train_images)
    X_test = Matrix.from_list(test_images)
    print(f"  ✓ X_train: {X_train.rows}x{X_train.cols}")
    print(f"  ✓ X_test: {X_test.rows}x{X_test.cols}")
    
    # One-hot encode labels
    Y_train_onehot = one_hot_encode(train_labels)
    Y_test_onehot = one_hot_encode(test_labels)
    
    Y_train = Matrix.from_list(Y_train_onehot)
    Y_test = Matrix.from_list(Y_test_onehot)
    print(f"  ✓ Y_train: {Y_train.rows}x{Y_train.cols}")
    print(f"  ✓ Y_test: {Y_test.rows}x{Y_test.cols}")
    
    print("="*70)
    print("✓ DATA LOADING COMPLETE!")
    print("="*70 + "\n")
    
    return X_train, train_labels, Y_train, X_test, test_labels, Y_test