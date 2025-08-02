#!/usr/bin/env python3
"""
Setup script for local ASL model training on Legion 5
Optimized for CPU training without GPU dependencies
"""

import os
import sys
import shutil
import zipfile
from pathlib import Path

def create_directory_structure():
    """Create necessary directories for training"""
    directories = [
        './models/',
        './logs/',
        './test_images/'
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"Created directory: {directory}")

def check_data_availability():
    """Check if ASL dataset is available"""
    archive_dir = r'C:\Users\Kaz\Desktop\AI4AllProject\archive'
    train_dir = os.path.join(archive_dir, 'asl_alphabet_train/')
    test_dir = os.path.join(archive_dir, 'asl_alphabet_test/')
    
    if os.path.exists(train_dir) and os.path.exists(test_dir):
        # Count files in train directory
        train_files = sum([len(files) for r, d, files in os.walk(train_dir)])
        test_files = sum([len(files) for r, d, files in os.walk(test_dir)])
        
        print(f"✓ ASL dataset found in archive!")
        print(f"  Training files: {train_files}")
        print(f"  Test files: {test_files}")
        print(f"  Archive location: {archive_dir}")
        return True
    else:
        print("✗ ASL dataset not found!")
        print("Please ensure the following structure exists:")
        print(f"  {train_dir}")
        print(f"  {test_dir}")
        return False

def extract_archive_data():
    """Extract data from archive folder if available"""
    archive_dir = r'C:\Users\Kaz\Desktop\AI4AllProject\archive'
    
    if os.path.exists(archive_dir):
        print("Found archive directory, checking for data...")
        print(f"Archive location: {archive_dir}")
        
        # Check for train data
        train_source = os.path.join(archive_dir, 'asl_alphabet_train/')
        if os.path.exists(train_source):
            print(f"✓ Training data found at: {train_source}")
        else:
            print(f"⚠ Training data not found at: {train_source}")
        
        # Check for test data
        test_source = os.path.join(archive_dir, 'asl_alphabet_test/')
        if os.path.exists(test_source):
            print(f"✓ Test data found at: {test_source}")
        else:
            print(f"⚠ Test data not found at: {test_source}")
    else:
        print(f"⚠ Archive directory not found at: {archive_dir}")

def check_system_requirements():
    """Check if system meets requirements for training"""
    print("\n=== System Requirements Check ===")
    
    # Check Python version
    python_version = sys.version_info
    if python_version.major >= 3 and python_version.minor >= 8:
        print(f"✓ Python {python_version.major}.{python_version.minor}.{python_version.micro}")
    else:
        print("✗ Python 3.8+ required")
        return False
    
    # Check available memory (approximate)
    try:
        import psutil
        memory = psutil.virtual_memory()
        memory_gb = memory.total / (1024**3)
        print(f"✓ Available RAM: {memory_gb:.1f} GB")
        
        if memory_gb < 8:
            print("⚠ Warning: Less than 8GB RAM may cause issues")
    except ImportError:
        print("⚠ psutil not available, cannot check memory")
    
    # Check disk space
    try:
        disk = shutil.disk_usage('./')
        disk_gb = disk.free / (1024**3)
        print(f"✓ Available disk space: {disk_gb:.1f} GB")
        
        if disk_gb < 5:
            print("⚠ Warning: Less than 5GB free space may cause issues")
    except:
        print("⚠ Cannot check disk space")
    
    return True

def install_requirements():
    """Install required packages"""
    print("\n=== Installing Requirements ===")
    
    try:
        import subprocess
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-r', 'requirements.txt'])
        print("✓ Requirements installed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Failed to install requirements: {e}")
        return False

def test_imports():
    """Test if all required packages can be imported"""
    print("\n=== Testing Imports ===")
    
    required_packages = [
        'tensorflow',
        'numpy',
        'matplotlib',
        'sklearn',
        'seaborn',
        'cv2',
        'PIL'
    ]
    
    failed_imports = []
    
    for package in required_packages:
        try:
            if package == 'cv2':
                try:
                    import cv2
                except ImportError:
                    print(f"⚠ {package}: Not installed yet (will be installed by setup)")
                    continue
            elif package == 'PIL':
                from PIL import Image
            else:
                __import__(package)
            print(f"✓ {package}")
        except ImportError as e:
            print(f"✗ {package}: {e}")
            failed_imports.append(package)
    
    if failed_imports:
        print(f"\nFailed to import: {failed_imports}")
        return False
    else:
        print("\n✓ All packages imported successfully!")
        return True

def main():
    """Main setup function"""
    print("=== ASL Model Local Training Setup ===")
    print("Optimized for Legion 5 CPU training\n")
    
    # Check system requirements
    if not check_system_requirements():
        print("System requirements not met!")
        return False
    
    # Create directory structure
    create_directory_structure()
    
    # Extract data if available
    extract_archive_data()
    
    # Check data availability
    if not check_data_availability():
        print("\nPlease download the ASL dataset and place it in the correct structure.")
        print("You can find the dataset at: https://www.kaggle.com/datasets/grassknoted/asl-alphabet")
        return False
    
    # Install requirements
    if not install_requirements():
        print("Failed to install requirements!")
        return False
    
    # Test imports
    if not test_imports():
        print("Some packages failed to import!")
        return False
    
    print("\n=== Setup Complete! ===")
    print("You can now run the training script:")
    print("python improved_asl_model.py")
    print("\nExpected training time: 6-10 hours for 100 epochs")
    print("The model will automatically save checkpoints during training")
    
    return True

if __name__ == "__main__":
    main() 