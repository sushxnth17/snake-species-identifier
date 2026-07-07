"""
Dataset Validation Module.
Provides high-performance parallel validation of image classification datasets.
"""

import os
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from PIL import Image
from typing import Tuple, Dict, List, Any

class DatasetValidator:
    """
    Validates the integrity, structure, and quality of an image classification dataset.
    Supports parallel verification of image files and memory-efficient incremental statistics.
    """
    
    # Extensions officially supported by Keras/TensorFlow image loading tools
    DEFAULT_SUPPORTED_EXTENSIONS = {'.bmp', '.gif', '.jpeg', '.jpg', '.png'}

    def __init__(
        self, 
        dataset_dir: str = "dataset", 
        supported_extensions: set = None, 
        num_workers: int = None
    ):
        """
        Initializes the DatasetValidator.
        
        Args:
            dataset_dir: Root directory of the dataset containing class subdirectories.
            supported_extensions: Set of lowercase file extensions considered valid.
            num_workers: Max threads to use for validation. Defaults to ThreadPoolExecutor default.
        """
        self.dataset_dir = dataset_dir
        self.supported_extensions = supported_extensions or self.DEFAULT_SUPPORTED_EXTENSIONS
        self.num_workers = num_workers

    def _validate_single_file(self, item: Tuple[str, str, str]) -> Dict[str, Any]:
        """
        Validates a single file inside a class directory.
        Runs inside a worker thread.
        
        Args:
            item: A tuple of (class_name, filename, file_path)
            
        Returns:
            A dictionary containing status, class_name, filename, and file metadata or error.
        """
        class_name, filename, file_path = item
        
        # Detect subdirectories (nested folders within class folders are not allowed by TF)
        if os.path.isdir(file_path):
            return {
                "status": "unsupported",
                "class_name": class_name,
                "filename": filename,
                "reason": "Subdirectory (Nested folder not allowed)",
                "file_size": 0
            }

        # Retrieve file size
        try:
            file_size = os.path.getsize(file_path)
        except OSError:
            file_size = 0

        # Check extension
        ext = os.path.splitext(filename)[1].lower()
        if ext not in self.supported_extensions:
            return {
                "status": "unsupported",
                "class_name": class_name,
                "filename": filename,
                "reason": f"Unsupported extension '{ext}'",
                "file_size": file_size
            }

        # Verify image integrity and extract size
        try:
            # First pass: structural check (very fast, doesn't load full pixels)
            with Image.open(file_path) as img:
                img.verify()
                
            # Second pass: read size (verify() invalidates file descriptor on some versions)
            with Image.open(file_path) as img:
                width, height = img.size
                
            return {
                "status": "valid",
                "class_name": class_name,
                "filename": filename,
                "width": width,
                "height": height,
                "file_size": file_size
            }
        except Exception as e:
            return {
                "status": "corrupted",
                "class_name": class_name,
                "filename": filename,
                "reason": f"{type(e).__name__}: {str(e)}",
                "file_size": file_size
            }

    def validate(self) -> Dict[str, Any]:
        """
        Performs the dataset integrity check and collects statistics.
        
        Returns:
            A dictionary containing validation findings and dataset statistics.
        """
        if not os.path.exists(self.dataset_dir):
            raise FileNotFoundError(f"Dataset directory '{self.dataset_dir}' does not exist.")
        if not os.path.isdir(self.dataset_dir):
            raise ValueError(f"Dataset path '{self.dataset_dir}' is not a directory.")

        # Discover class folders and group files to check
        entries = sorted(os.listdir(self.dataset_dir))
        class_folders = []
        empty_folders = []
        all_files_to_check = []

        for entry in entries:
            entry_path = os.path.join(self.dataset_dir, entry)
            # Filter out hidden folders/metadata files in root
            if os.path.isdir(entry_path) and not entry.startswith('.'):
                class_folders.append(entry)
                try:
                    files = os.listdir(entry_path)
                except OSError:
                    files = []
                    
                if not files:
                    empty_folders.append(entry)
                else:
                    for filename in files:
                        # Exclude system/hidden files like .DS_Store
                        if filename.startswith('.'):
                            continue
                        file_path = os.path.join(entry_path, filename)
                        all_files_to_check.append((entry, filename, file_path))

        total_files = len(all_files_to_check)
        results = []
        processed_count = 0

        # Run validation in parallel
        if total_files > 0:
            with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
                # Submit all files to the pool
                future_to_item = {
                    executor.submit(self._validate_single_file, item): item 
                    for item in all_files_to_check
                }
                
                # Gather results and print progress
                for future in as_completed(future_to_item):
                    item = future_to_item[future]
                    try:
                        res = future.result()
                        results.append(res)
                    except Exception as e:
                        results.append({
                            "status": "corrupted",
                            "class_name": item[0],
                            "filename": item[1],
                            "reason": f"ExecutionError: {str(e)}",
                            "file_size": 0
                        })
                    
                    processed_count += 1
                    # Progress reporting for large datasets (every 10% or at completion)
                    if total_files > 1000 and (processed_count % max(1, total_files // 10) == 0 or processed_count == total_files):
                        print(f"  Processed {processed_count}/{total_files} files ({processed_count * 100 // total_files}%)...")

        # Aggregate metrics
        no_valid_images_folders = []
        unsupported_files = []
        corrupted_images = []
        
        class_counts = defaultdict(int)
        filename_to_paths = defaultdict(list)
        
        total_valid_images = 0
        total_size_bytes = 0
        sum_widths = 0
        sum_heights = 0
        
        min_width, min_height = float('inf'), float('inf')
        max_width, max_height = 0, 0

        for res in results:
            class_name = res["class_name"]
            filename = res["filename"]
            
            # Map filename to paths to detect duplicates across folders
            filename_to_paths[filename].append(os.path.join(class_name, filename))
            
            # Sum up sizes for total size on disk
            total_size_bytes += res.get("file_size", 0)

            if res["status"] == "valid":
                total_valid_images += 1
                class_counts[class_name] += 1
                
                w, h = res["width"], res["height"]
                sum_widths += w
                sum_heights += h
                
                # Keep track of min/max resolutions
                if w < min_width: min_width = w
                if h < min_height: min_height = h
                if w > max_width: max_width = w
                if h > max_height: max_height = h
                
            elif res["status"] == "unsupported":
                unsupported_files.append({
                    "filename": filename,
                    "class_name": class_name,
                    "reason": res["reason"]
                })
            elif res["status"] == "corrupted":
                corrupted_images.append({
                    "filename": filename,
                    "class_name": class_name,
                    "reason": res["reason"]
                })

        # Find folders that contain no valid images
        for folder in class_folders:
            if folder not in empty_folders and class_counts[folder] == 0:
                no_valid_images_folders.append(folder)

        # Detect duplicate filenames
        duplicate_filenames = {
            name: paths for name, paths in filename_to_paths.items() if len(paths) > 1
        }

        # Calculate final resolution averages
        if total_valid_images > 0:
            avg_width = sum_widths / total_valid_images
            avg_height = sum_heights / total_valid_images
            avg_resolution = (avg_width, avg_height)
            min_resolution = (int(min_width), int(min_height))
            max_resolution = (int(max_width), int(max_height))
        else:
            avg_resolution = (0.0, 0.0)
            min_resolution = (0, 0)
            max_resolution = (0, 0)

        return {
            "class_folders": class_folders,
            "empty_folders": empty_folders,
            "no_valid_images_folders": no_valid_images_folders,
            "unsupported_files": unsupported_files,
            "corrupted_images": corrupted_images,
            "duplicate_filenames": duplicate_filenames,
            "class_counts": dict(class_counts),
            "total_valid_images": total_valid_images,
            "total_size_bytes": total_size_bytes,
            "average_resolution": avg_resolution,
            "min_resolution": min_resolution,
            "max_resolution": max_resolution
        }

    def print_summary(self, results: Dict[str, Any], limit: int = 10) -> None:
        """
        Prints a beautiful, comprehensive summary report of the dataset.
        
        Args:
            results: The results dictionary returned by validate().
            limit: Maximum number of entries to display for error lists.
        """
        print("=" * 60)
        print("                 DATASET INTEGRITY REPORT                 ")
        print("=" * 60)
        
        # 1. Dataset Scale Statistics
        total_valid = results["total_valid_images"]
        total_size_mb = results["total_size_bytes"] / (1024 * 1024)
        print(f"Total Valid Images:      {total_valid}")
        print(f"Total Dataset Size:      {total_size_mb:.2f} MB")
        
        # 2. Image Resolution Statistics
        if total_valid > 0:
            avg_w, avg_h = results["average_resolution"]
            min_w, min_h = results["min_resolution"]
            max_w, max_h = results["max_resolution"]
            print(f"Average Resolution:      {avg_w:.1f}x{avg_h:.1f}")
            print(f"Resolution Range:        {min_w}x{min_h} to {max_w}x{max_h}")
        else:
            print("Average Resolution:      N/A (No valid images)")
            
        print("-" * 60)
        
        # 3. Class breakdown
        print("Images per Class:")
        class_folders = results["class_folders"]
        class_counts = results["class_counts"]
        if not class_folders:
            print("  (No class folders found)")
        else:
            for folder in class_folders:
                count = class_counts.get(folder, 0)
                print(f"  - {folder:<20}: {count} valid images")
            
        print("-" * 60)
        
        # 4. Empty or invalid folders
        empty_folders = results["empty_folders"]
        no_val_folders = results["no_valid_images_folders"]
        all_empty_or_invalid_folders = sorted(list(set(empty_folders + no_val_folders)))
        
        if all_empty_or_invalid_folders:
            print("[WARNING] Empty or invalid class directories found:")
            for folder in all_empty_or_invalid_folders:
                reason = "Literally empty" if folder in empty_folders else "Contains 0 valid images"
                print(f"  - {folder} ({reason})")
        else:
            print("[PASS] All class folders contain valid images.")
            
        print("-" * 60)
        
        # 5. Unsupported Files
        unsupported = results["unsupported_files"]
        if unsupported:
            print(f"[WARNING] Found {len(unsupported)} unsupported file(s) or folder(s):")
            for item in unsupported[:limit]:
                print(f"  - {item['class_name']}/{item['filename']} (Reason: {item['reason']})")
            if len(unsupported) > limit:
                print(f"  ... and {len(unsupported) - limit} more.")
        else:
            print("[PASS] No unsupported files or subdirectories detected.")
            
        print("-" * 60)
        
        # 6. Corrupted Images
        corrupted = results["corrupted_images"]
        if corrupted:
            print(f"[FAIL] Found {len(corrupted)} corrupted image(s):")
            for item in corrupted[:limit]:
                print(f"  - {item['class_name']}/{item['filename']} (Error: {item['reason']})")
            if len(corrupted) > limit:
                print(f"  ... and {len(corrupted) - limit} more.")
        else:
            print("[PASS] No corrupted images detected.")
            
        print("-" * 60)
        
        # 7. Duplicate Filenames
        duplicates = results["duplicate_filenames"]
        if duplicates:
            print(f"[WARNING] Found {len(duplicates)} duplicate filename(s) across different folders:")
            for name, paths in list(duplicates.items())[:limit]:
                print(f"  - '{name}' appears in: {', '.join(paths)}")
            if len(duplicates) > limit:
                print(f"  ... and {len(duplicates) - limit} more.")
        else:
            print("[PASS] No duplicate filenames found.")
            
        print("=" * 60)
