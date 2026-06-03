#!/usr/bin/env python3
"""
Phase 1: Download ISIC 2019 Dataset
Downloads 25,331 skin lesion images for training the Vision SLM
"""

import os
import requests
import json
from pathlib import Path
from tqdm import tqdm

def setup_folders():
    """Create dataset folder structure"""
    base_path = Path('dataset/isic_2019')
    base_path.mkdir(parents=True, exist_ok=True)
    
    categories = [
        'MEL',      # Melanoma
        'NV',       # Nevus
        'BCC',      # Basal cell carcinoma
        'AKIEC',    # Actinic keratosis
        'BKL',      # Benign keratosis
        'DF',       # Dermatofibroma
        'VASC',     # Vascular lesion
        'SCC',      # Squamous cell carcinoma
        'UNK'       # Unknown
    ]
    
    for category in categories:
        (base_path / category).mkdir(exist_ok=True)
    
    return base_path

def download_from_isic():
    """
    Download ISIC 2019 dataset
    Note: This function provides instructions for manual download
    """
    print("\n" + "="*70)
    print("ISIC 2019 Dataset Download Instructions")
    print("="*70)
    print()
    print("The ISIC dataset requires manual download due to API constraints.")
    print()
    print("Step 1: Go to https://isic-archive.com/")
    print("Step 2: Click 'Challenges' → '2019'")
    print("Step 3: Download the following:")
    print("   - Training Data (9.16GB)")
    print("   - Training Ground Truth (1MB)")
    print("   - Test Data (3.6GB)")
    print("   - Test Ground Truth (287KB)")
    print()
    print("Step 4: Extract ZIP files to: dataset/isic_2019/raw/")
    print()
    print("Or use Kaggle API (if you have credentials):")
    print()
    print("Step 1: Get Kaggle API key from https://www.kaggle.com/settings/account")
    print("Step 2: Place kaggle.json in ~/.kaggle/")
    print("Step 3: Run this command:")
    print()
    print("   kaggle datasets download -d nodoubttome/skin-cancer9-classesisic")
    print("   unzip -q skin-cancer9-classesisic.zip -d dataset/isic_2019/raw/")
    print()
    print("="*70)
    print()

def organize_dataset():
    """
    Organize raw dataset into category folders
    Call this after manual download
    """
    print("\nOrganizing dataset into categories...")
    
    base_path = Path('dataset/isic_2019')
    raw_path = base_path / 'raw'
    
    if not raw_path.exists():
        print("Raw dataset not found. Please download first.")
        return False
    
    # Read metadata CSV
    import pandas as pd
    
    metadata_file = raw_path / 'HAM10000_metadata.csv'
    if not metadata_file.exists():
        print("Metadata file not found")
        return False
    
    df = pd.read_csv(metadata_file)
    
    # Map diagnosis to category
    diagnosis_map = {
        'mel': 'MEL',
        'nv': 'NV',
        'bcc': 'BCC',
        'akiec': 'AKIEC',
        'bkl': 'BKL',
        'df': 'DF',
        'vasc': 'VASC',
        'scc': 'SCC',
        'unk': 'UNK'
    }
    
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        image_id = row['image_id']
        diagnosis = row['dx']
        category = diagnosis_map.get(diagnosis, 'UNK')
        
        # Find and copy image
        for img_file in raw_path.glob(f'{image_id}.jpg'):
            dest = base_path / category / img_file.name
            if not dest.exists():
                import shutil
                shutil.copy2(img_file, dest)
    
    print("Dataset organized successfully!")
    return True

def verify_dataset():
    """
    Verify dataset integrity and print statistics
    """
    print("\nVerifying dataset...")
    
    base_path = Path('dataset/isic_2019')
    
    categories = {
        'MEL': 'Melanoma',
        'NV': 'Nevus',
        'BCC': 'Basal Cell Carcinoma',
        'AKIEC': 'Actinic Keratosis',
        'BKL': 'Benign Keratosis',
        'DF': 'Dermatofibroma',
        'VASC': 'Vascular Lesion',
        'SCC': 'Squamous Cell Carcinoma',
        'UNK': 'Unknown'
    }
    
    total_images = 0
    
    print("\n" + "="*60)
    print(f"{'Category':<20} {'Count':>10} {'Description':<25}")
    print("="*60)
    
    for cat_code, cat_name in categories.items():
        cat_path = base_path / cat_code
        if cat_path.exists():
            count = len(list(cat_path.glob('*.jpg')))
            total_images += count
            print(f"{cat_code:<20} {count:>10} {cat_name:<25}")
    
    print("="*60)
    print(f"{'TOTAL':<20} {total_images:>10}")
    print("="*60)
    
    if total_images >= 25000:
        print("✅ Dataset ready for training!")
        return True
    else:
        print(f"⚠️  Expected 25,331 images, found {total_images}")
        return False

if __name__ == '__main__':
    print("\n" + "="*70)
    print("Mirror Analyzer - Phase 1: Dataset Download")
    print("="*70)
    
    # Create folders
    print("\nCreating folder structure...")
    setup_folders()
    print("✅ Folders created")
    
    # Show download instructions
    download_from_isic()
    
    # After download, organize
    input("\nPress Enter after downloading dataset files...")
    organize_dataset()
    
    # Verify
    verify_dataset()
    
    print("\n" + "="*70)
    print("Next: Run augment_data.py")
    print("="*70)
