#!/usr/bin/env python3
"""
Dataset Statistics and Visualization
Analyze ISIC 2019 dataset after download
"""

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
import json

def analyze_dataset():
    """
    Analyze dataset statistics
    """
    base_path = Path('dataset/isic_2019')
    
    categories = ['MEL', 'NV', 'BCC', 'AKIEC', 'BKL', 'DF', 'VASC', 'SCC', 'UNK']
    
    stats = {}
    all_sizes = []
    
    print("Analyzing dataset...\n")
    
    for category in categories:
        cat_path = base_path / category
        images = list(cat_path.glob('*.jpg'))
        
        sizes = []
        shapes = []
        
        for img_file in tqdm(images, desc=f"Processing {category}"):
            img = cv2.imread(str(img_file))
            if img is not None:
                shapes.append(img.shape)
                sizes.append(os.path.getsize(img_file) / 1024)  # KB
        
        stats[category] = {
            'count': len(images),
            'avg_size_kb': np.mean(sizes) if sizes else 0,
            'avg_shape': tuple(map(int, np.mean(shapes, axis=0))) if shapes else None
        }
        all_sizes.extend(sizes)
    
    return stats, all_sizes

def print_statistics(stats):
    """
    Print dataset statistics in formatted table
    """
    print("\n" + "="*70)
    print("ISIC 2019 Dataset Statistics")
    print("="*70)
    
    category_names = {
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
    
    print(f"\n{'Category':<8} {'Name':<25} {'Count':>8} {'Avg Size (KB)':>15}")
    print("-"*70)
    
    for cat, data in sorted(stats.items()):
        count = data['count']
        avg_size = data['avg_size_kb']
        total_images += count
        
        print(f"{cat:<8} {category_names.get(cat, cat):<25} {count:>8} {avg_size:>15.1f}")
    
    print("-"*70)
    print(f"{'TOTAL':<8} {'':<25} {total_images:>8}")
    print("="*70)
    
    # Calculate percentages
    print("\nClass Distribution (%):\n")
    for cat, data in sorted(stats.items()):
        percentage = (data['count'] / total_images) * 100
        bar_length = int(percentage / 2)
        bar = '█' * bar_length
        print(f"{cat:<8} {bar:<25} {percentage:>6.2f}%")
    
    print("\n" + "="*70)

def plot_statistics(stats):
    """
    Create visualization of dataset statistics
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    categories = list(stats.keys())
    counts = [stats[cat]['count'] for cat in categories]
    sizes = [stats[cat]['avg_size_kb'] for cat in categories]
    
    # Bar chart of image counts
    axes[0].bar(categories, counts, color='skyblue', edgecolor='navy')
    axes[0].set_title('Images per Disease Category', fontsize=14, fontweight='bold')
    axes[0].set_ylabel('Number of Images', fontsize=12)
    axes[0].set_xlabel('Disease Category', fontsize=12)
    axes[0].grid(axis='y', alpha=0.3)
    
    # Bar chart of average sizes
    axes[1].bar(categories, sizes, color='lightcoral', edgecolor='darkred')
    axes[1].set_title('Average Image Size per Category', fontsize=14, fontweight='bold')
    axes[1].set_ylabel('Size (KB)', fontsize=12)
    axes[1].set_xlabel('Disease Category', fontsize=12)
    axes[1].grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('dataset_statistics.png', dpi=150, bbox_inches='tight')
    print("\n✅ Saved: dataset_statistics.png")
    plt.show()

def save_statistics_json(stats):
    """
    Save statistics to JSON file
    """
    with open('dataset_stats.json', 'w') as f:
        json.dump(stats, f, indent=2)
    print("✅ Saved: dataset_stats.json")

if __name__ == '__main__':
    print("\n" + "="*70)
    print("Mirror Analyzer - Dataset Statistics")
    print("="*70)
    
    # Analyze
    stats, all_sizes = analyze_dataset()
    
    # Print
    print_statistics(stats)
    
    # Visualize
    print("\nGenerating visualizations...")
    plot_statistics(stats)
    
    # Save
    save_statistics_json(stats)
    
    print("\n" + "="*70)
    print("Next: Run augment_data.py")
    print("="*70)
