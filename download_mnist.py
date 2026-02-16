import urllib.request
import gzip
import os

def download_file(url, filename):
    """Download file from URL"""
    if not os.path.exists(filename):
        print(f"Downloading {filename}...")
        urllib.request.urlretrieve(url, filename)
        print(f"Downloaded {filename}")
    else:
        print(f"{filename} already exists")

def download_mnist():
    """Download MNIST dataset"""
    base_url = "http://yann.lecun.com/exdb/mnist/"
    
    files = {
        'train-images-idx3-ubyte.gz': 'train_images',
        'train-labels-idx1-ubyte.gz': 'train_labels',
        't10k-images-idx3-ubyte.gz': 'test_images',
        't10k-labels-idx1-ubyte.gz': 'test_labels'
    }
    
    # Create data directory
    if not os.path.exists('data'):
        os.makedirs('data')
    
    # Download all files
    for filename, _ in files.items():
        url = base_url + filename
        filepath = os.path.join('data', filename)
        download_file(url, filepath)
    
    print("\nAll MNIST files downloaded successfully!")

if __name__ == "__main__":
    download_mnist()