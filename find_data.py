import os

print("Searching for MNIST files...\n")

# Check all possible locations
locations_to_check = [
    'data',
    r'data\MNIST',
    r'data\mnist',
]

mnist_files = [
    'train-images-idx3-ubyte.gz',
    'train-images-idx3-ubyte',
]

found = False
for location in locations_to_check:
    print(f"Checking: {location}")
    if os.path.exists(location):
        print(f"  ✓ Folder exists")
        contents = os.listdir(location)
        print(f"  Contents: {contents}")
        
        # Check for MNIST files
        for mnist_file in mnist_files:
            full_path = os.path.join(location, mnist_file)
            if os.path.exists(full_path):
                print(f"\n✓✓✓ FOUND MNIST DATA! ✓✓✓")
                print(f"Use this in train.py: data_path = r'{location}'")
                found = True
                break
        if found:
            break
    else:
        print(f"  ✗ Folder does not exist")
    print()

if not found:
    print("\n❌ MNIST files not found in expected locations")
    print("\nLet's check your data folder structure:")
    if os.path.exists('data'):
        print("\nContents of 'data' folder:")
        for item in os.listdir('data'):
            item_path = os.path.join('data', item)
            if os.path.isdir(item_path):
                print(f"  📁 {item}/")
                # Check inside subfolders
                try:
                    for subitem in os.listdir(item_path):
                        print(f"      - {subitem}")
                except:
                    pass
            else:
                print(f"  📄 {item}")