import os
import sys
from collections import Counter
from PIL import Image

def validate_dataset(dataset_dir="dataset"):
    print("=" * 60)
    print(f"Dataset Validation Report for: {dataset_dir}")
    print("=" * 60)
    
    if not os.path.exists(dataset_dir):
        print(f"[ERROR] Directory '{dataset_dir}' does not exist.")
        sys.exit(1)
        
    # Supported formats by TensorFlow's image_dataset_from_directory
    supported_extensions = {'.bmp', '.gif', '.jpeg', '.jpg', '.png'}
    
    # Track statistics
    class_folders = []
    empty_folders = []
    unsupported_files = []
    corrupted_images = []
    all_filenames = []
    duplicate_filenames = []
    
    image_widths = []
    image_heights = []
    
    # Iterate through target directories
    for entry in os.listdir(dataset_dir):
        entry_path = os.path.join(dataset_dir, entry)
        if os.path.isdir(entry_path):
            class_folders.append(entry)
            
            # Read files in this directory
            files = os.listdir(entry_path)
            if not files:
                empty_folders.append(entry)
                continue
                
            for filename in files:
                file_path = os.path.join(entry_path, filename)
                if os.path.isdir(file_path):
                    # Species folder has subfolders (nested directory)
                    unsupported_files.append((filename, entry, "Subdirectory (Nested folder not allowed)"))
                    continue
                    
                # Track filename duplicate
                all_filenames.append(filename)
                
                # Check file extension
                ext = os.path.splitext(filename)[1].lower()
                if ext not in supported_extensions:
                    unsupported_files.append((filename, entry, f"Unsupported extension '{ext}'"))
                    continue
                    
                # Verify image integrity and record dimension statistics
                try:
                    with Image.open(file_path) as img:
                        img.verify()  # Verifies image structure (cheap)
                    
                    # Re-open to read size (verify() closes file pointer but doesn't load image data)
                    with Image.open(file_path) as img:
                        width, height = img.size
                        image_widths.append(width)
                        image_heights.append(height)
                except Exception as e:
                    corrupted_images.append((filename, entry, str(e)))

    # Compute duplicate filenames
    filename_counts = Counter(all_filenames)
    duplicate_filenames = [name for name, count in filename_counts.items() if count > 1]
    
    # Summary of Classes
    print(f"Classes Found: {len(class_folders)}")
    for folder in class_folders:
        folder_path = os.path.join(dataset_dir, folder)
        if os.path.isdir(folder_path):
            img_count = sum(1 for f in os.listdir(folder_path) if os.path.splitext(f)[1].lower() in supported_extensions)
            print(f"  - {folder}: {img_count} valid images")
    print("-" * 60)
    
    # Empty Folders Check
    if empty_folders:
        print(f"[WARNING] Empty class directories found:")
        for folder in empty_folders:
            print(f"  - {folder}")
    else:
        print("[PASS] No empty directories found.")
    print("-" * 60)
        
    # Unsupported Files Check
    if unsupported_files:
        print(f"[WARNING] Found {len(unsupported_files)} unsupported file(s) or folder(s):")
        for filename, folder, reason in unsupported_files[:10]:  # Limit to 10 logs
            print(f"  - {folder}/{filename} ({reason})")
        if len(unsupported_files) > 10:
            print(f"  ... and {len(unsupported_files) - 10} more.")
    else:
        print("[PASS] All files have supported image extensions.")
    print("-" * 60)

    # Corrupted Images Check
    if corrupted_images:
        print(f"[FAIL] Found {len(corrupted_images)} corrupted image(s):")
        for filename, folder, error in corrupted_images[:10]:
            print(f"  - {folder}/{filename} (Error: {error})")
        if len(corrupted_images) > 10:
            print(f"  ... and {len(corrupted_images) - 10} more.")
    else:
        print("[PASS] No corrupted images detected.")
    print("-" * 60)

    # Duplicate Filenames Check
    if duplicate_filenames:
        print(f"[WARNING] Found {len(duplicate_filenames)} duplicate filename(s) across different folders:")
        for name in duplicate_filenames[:10]:
            print(f"  - {name} appears {filename_counts[name]} times")
        if len(duplicate_filenames) > 10:
            print(f"  ... and {len(duplicate_filenames) - 10} more.")
    else:
        print("[PASS] No duplicate filenames found.")
    print("-" * 60)

    # Dimension Statistics
    if image_widths and image_heights:
        avg_w = sum(image_widths) / len(image_widths)
        avg_h = sum(image_heights) / len(image_heights)
        print("Image Dimension Statistics (Valid Images Only):")
        print(f"  - Total processed: {len(image_widths)}")
        print(f"  - Min size: {min(image_widths)}x{min(image_heights)}")
        print(f"  - Max size: {max(image_widths)}x{max(image_heights)}")
        print(f"  - Avg size: {avg_w:.1f}x{avg_h:.1f}")
    else:
        print("[WARNING] No valid image dimensions were collected.")
    print("=" * 60)

if __name__ == "__main__":
    validate_dataset()
