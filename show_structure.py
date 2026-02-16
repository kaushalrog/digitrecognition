import os

def show_tree(path, prefix="", max_depth=3, current_depth=0):
    """Show directory tree structure"""
    if current_depth >= max_depth:
        return
    
    try:
        items = sorted(os.listdir(path))
        for i, item in enumerate(items):
            item_path = os.path.join(path, item)
            is_last = i == len(items) - 1
            
            connector = "└── " if is_last else "├── "
            print(prefix + connector + item)
            
            if os.path.isdir(item_path):
                extension = "    " if is_last else "│   "
                show_tree(item_path, prefix + extension, max_depth, current_depth + 1)
            elif item_path.endswith(('.gz', 'ubyte')):
                size = os.path.getsize(item_path)
                print(f"{prefix}{'    ' if is_last else '│   '}    ({size:,} bytes)")
    except PermissionError:
        print(f"{prefix}[Permission Denied]")

print("MNIST Folder Structure:")
print("="*50)
show_tree('data')