#!/usr/bin/env python3
"""
Phase 2: Data Augmentation
Triples dataset size using various image transformations
"""

import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2

def create_augmentation_pipeline():
    """
    Create data augmentation pipeline
    """
    augment_transform = A.Compose([
        A.RandomRotate90(),
        A.Flip(),
        A.Transpose(),
        A.GaussNoise(p=0.2),
        A.OneOf([
            A.MotionBlur(p=0.2),
            A.MedianBlur(blur_limit=3, p=0.1),
            A.Blur(blur_limit=3, p=0.1),
        ], p=0.2),
        A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.2, rotate_limit=45, p=0.2),
        A.OneOf([
            A.OpticalDistortion(p=0.3),
            A.GridDistortion(p=0.1),
        ], p=0.2),
        A.OneOf([
            A.CLAHE(clip_limit=2),
            A.Sharpen(),
            A.Emboss(),
        ], p=0.3),
        A.HueSaturationValue(p=0.3),
    ])
    
    return augment_transform

def augment_dataset():
    """
    Apply augmentation to all images
    """
    base_path = Path('dataset/isic_2019')
    categories = ['MEL', 'NV', 'BCC', 'AKIEC', 'BKL', 'DF', 'VASC', 'SCC', 'UNK']
    
    # Create augmented folder
    augmented_path = base_path / 'augmented'
    augmented_path.mkdir(exist_ok=True)
    
    for category in categories:
        (augmented_path / category).mkdir(exist_ok=True)
    
    augment_transform = create_augmentation_pipeline()
    
    print("\n" + "="*70)
    print("Data Augmentation in Progress")
    print("="*70)
    
    total_augmented = 0
    
    for category in categories:
        cat_path = base_path / category
        images = list(cat_path.glob('*.jpg'))
        
        print(f"\nAugmenting {category} ({len(images)} images)...")
        
        for img_file in tqdm(images):
            # Read original
            img = cv2.imread(str(img_file))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Save original
            output_path = augmented_path / category / f"{img_file.stem}_0.jpg"
            cv2.imwrite(str(output_path), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
            total_augmented += 1
            
            # Create 2 augmented versions
            for aug_num in range(1, 3):
                augmented = augment_transform(image=img)['image']
                output_path = augmented_path / category / f"{img_file.stem}_{aug_num}.jpg"
                cv2.imwrite(str(output_path), cv2.cvtColor(augmented, cv2.COLOR_RGB2BGR))
                total_augmented += 1
        
        print(f"✅ {category}: {len(images) * 3} images")
    
    print("\n" + "="*70)
    print(f"Total images created: {total_augmented}")
    print(f"Original: ~25,331 → Augmented: ~75,993")
    print("="*70)
    
    return total_augmented

def verify_augmentation():
    """
    Verify augmented dataset
    """
    base_path = Path('dataset/isic_2019/augmented')
    categories = ['MEL', 'NV', 'BCC', 'AKIEC', 'BKL', 'DF', 'VASC', 'SCC', 'UNK']
    
    print("\nVerifying augmented dataset...\n")
    
    total = 0
    print(f"{'Category':<15} {'Images':>10}")
    print("-"*25)
    
    for category in categories:
        count = len(list((base_path / category).glob('*.jpg')))
        total += count
        print(f"{category:<15} {count:>10}")
    
    print("-"*25)
    print(f"{'TOTAL':<15} {total:>10}")
    
    return total

if __name__ == '__main__':
    print("\n" + "="*70)
    print("Mirror Analyzer - Phase 2: Data Augmentation")
    print("="*70)
    
    print("\nThis will:")
    print("1. Create 2 augmented versions of each image")
    print("2. Triple dataset size (25k → 75k images)")
    print("3. Improve model generalization")
    print("\nThis may take 30-60 minutes depending on your system...\n")
    
    # Augment
    total = augment_dataset()
    
    # Verify
    verify_augmentation()
    
    print("\n" + "="*70)
    print("Next: Run train_mobilenetv2.py or upload to Google Colab")
    print("="*70)
