import sys
import importlib

def check_package(package_name, import_name=None):
    """
    Checks if a package is installed and returns its version.
    """
    if import_name is None:
        import_name = package_name
        
    try:
        module = importlib.import_module(import_name)
        # Try to extract version, default to 'Installed' if not available
        version = getattr(module, '__version__', 'Installed')
        return True, version
    except ImportError:
        return False, "Not Installed"

def main():
    print("=" * 50)
    print("AI Snake Identifier - Environment Verification")
    print("=" * 50)
    
    # Check Python version
    python_version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    print(f"[PASS] Python: Version {python_version}")
    
    # Required dependencies
    dependencies = {
        "TensorFlow": ("tensorflow", "tensorflow"),
        "FastAPI": ("fastapi", "fastapi"),
        "NumPy": ("numpy", "numpy"),
        "Pillow": ("Pillow", "PIL"),
    }
    
    all_passed = True
    
    for name, (package_name, import_name) in dependencies.items():
        success, info = check_package(package_name, import_name)
        if success:
            print(f"[PASS] {name}: {info}")
        else:
            print(f"[FAIL] {name}: {info} (Required)")
            all_passed = False
            
    print("=" * 50)
    if all_passed:
        print(" SUCCESS: Environment is fully configured and ready!")
        sys.exit(0)
    else:
        print(" ERROR: One or more required packages are missing.")
        print("Please run: pip install -r requirements.txt")
        sys.exit(1)

if __name__ == "__main__":
    main()
