import os
import tempfile
import pytest
from PIL import Image
from ml.dataset_validator import DatasetValidator

def create_valid_image(path, size=(100, 100)):
    """Helper to create a valid image on disk."""
    img = Image.new("RGB", size, color="blue")
    img.save(path, format="PNG")

def create_corrupted_image(path):
    """Helper to write invalid bytes to an image file path."""
    with open(path, "wb") as f:
        f.write(b"not a real image header or content")

def create_text_file(path):
    """Helper to create a non-image text file."""
    with open(path, "w") as f:
        f.write("unsupported file content")

@pytest.fixture
def temp_dataset():
    """
    Fixture that creates a temporary dataset folder with multiple classes,
    containing valid, corrupted, unsupported, and nested items.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create class folders
        class_a = os.path.join(tmpdir, "cobra")
        class_b = os.path.join(tmpdir, "krait")
        class_empty = os.path.join(tmpdir, "viper") # Empty folder
        class_only_corrupt = os.path.join(tmpdir, "python") # Folder with 0 valid images
        
        for d in [class_a, class_b, class_empty, class_only_corrupt]:
            os.makedirs(d)
            
        # Class A: 2 valid images (100x200 and 300x400)
        create_valid_image(os.path.join(class_a, "img1.png"), size=(100, 200))
        create_valid_image(os.path.join(class_a, "img2.png"), size=(300, 400))
        
        # Class B: 1 valid image (200x200) + 1 duplicate filename ("img1.png") + 1 unsupported txt file
        create_valid_image(os.path.join(class_b, "img1.png"), size=(200, 200)) # Duplicate filename of class_a/img1.png
        create_text_file(os.path.join(class_b, "notes.txt"))
        
        # Class B: 1 nested subdirectory
        os.makedirs(os.path.join(class_b, "nested_dir"))
        
        # Class Only Corrupt: 1 corrupted png
        create_corrupted_image(os.path.join(class_only_corrupt, "bad_image.png"))
        
        yield tmpdir

def test_validation_statistics_and_integrity(temp_dataset):
    """
    Test that DatasetValidator correctly identifies valid files, dimensions,
    unsupported items, corruptions, duplicate filenames, and empty classes.
    """
    validator = DatasetValidator(temp_dataset, num_workers=2)
    results = validator.validate()
    
    # 1. Class Folders Detection
    expected_classes = {"cobra", "krait", "python", "viper"}
    assert set(results["class_folders"]) == expected_classes
    
    # 2. Total Valid Images (cobra/img1, cobra/img2, krait/img1) -> 3 images
    assert results["total_valid_images"] == 3
    
    # 3. Class Counts
    assert results["class_counts"]["cobra"] == 2
    assert results["class_counts"]["krait"] == 1
    assert "python" not in results["class_counts"] or results["class_counts"]["python"] == 0
    assert "viper" not in results["class_counts"] or results["class_counts"]["viper"] == 0
    
    # 4. Resolution Statistics
    # Valid sizes: (100, 200), (300, 400), (200, 200)
    # Sum width = 100+300+200 = 600 -> Avg width = 200.0
    # Sum height = 200+400+200 = 800 -> Avg height = 266.666...
    avg_w, avg_h = results["average_resolution"]
    assert abs(avg_w - 200.0) < 1e-5
    assert abs(avg_h - 266.666666) < 1e-1
    
    # Min/Max sizes
    assert results["min_resolution"] == (100, 200)
    assert results["max_resolution"] == (300, 400)
    
    # 5. Empty / Invalid Folders
    # viper is empty
    # python has no valid images (only corrupted)
    assert "viper" in results["empty_folders"]
    assert "python" in results["no_valid_images_folders"]
    
    # 6. Unsupported Files (notes.txt + nested_dir)
    unsupported_names = {x["filename"] for x in results["unsupported_files"]}
    assert "notes.txt" in unsupported_names
    assert "nested_dir" in unsupported_names
    
    # 7. Corrupted Images (bad_image.png)
    corrupted_names = {x["filename"] for x in results["corrupted_images"]}
    assert "bad_image.png" in corrupted_names
    
    # 8. Duplicate Filenames ("img1.png")
    assert "img1.png" in results["duplicate_filenames"]
    assert len(results["duplicate_filenames"]["img1.png"]) == 2
    expected_paths = {os.path.join("cobra", "img1.png"), os.path.join("krait", "img1.png")}
    assert set(results["duplicate_filenames"]["img1.png"]) == expected_paths

def test_empty_dataset_directory():
    """Test validator behavior when dataset directory is empty of any class folders."""
    with tempfile.TemporaryDirectory() as tmpdir:
        validator = DatasetValidator(tmpdir)
        results = validator.validate()
        
        assert results["class_folders"] == []
        assert results["total_valid_images"] == 0
        assert results["average_resolution"] == (0.0, 0.0)
        assert results["min_resolution"] == (0, 0)
        assert results["max_resolution"] == (0, 0)

def test_missing_dataset_directory():
    """Test that validator raises FileNotFoundError for non-existent directories."""
    validator = DatasetValidator("non_existent_folder_path_123")
    with pytest.raises(FileNotFoundError):
        validator.validate()
