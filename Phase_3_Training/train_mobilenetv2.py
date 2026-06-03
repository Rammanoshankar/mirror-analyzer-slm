#!/usr/bin/env python3
"""
Phase 3: Train Vision SLM
Fine-tune MobileNetV2 on ISIC dataset for skin disease classification
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
import cv2
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
import json

def load_data(augmented=True):
    """
    Load augmented dataset
    """
    print("Loading dataset...")
    
    if augmented:
        base_path = Path('dataset/isic_2019/augmented')
    else:
        base_path = Path('dataset/isic_2019')
    
    categories = ['MEL', 'NV', 'BCC', 'AKIEC', 'BKL', 'DF', 'VASC', 'SCC', 'UNK']
    category_to_idx = {cat: idx for idx, cat in enumerate(categories)}
    
    images = []
    labels = []
    
    for category in categories:
        cat_path = base_path / category
        img_files = list(cat_path.glob('*.jpg'))
        
        print(f"Loading {category}: {len(img_files)} images")
        
        for img_file in tqdm(img_files):
            img = cv2.imread(str(img_file))
            if img is not None:
                # Resize to 224x224 (MobileNetV2 input size)
                img = cv2.resize(img, (224, 224))
                # Convert BGR to RGB
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                # Normalize to [0, 1]
                img = img / 255.0
                
                images.append(img)
                labels.append(category_to_idx[category])
    
    X = np.array(images, dtype=np.float32)
    y = np.array(labels, dtype=np.int32)
    
    print(f"\nDataset loaded: {X.shape} images")
    
    return X, y, categories, category_to_idx

def create_model(num_classes=9):
    """
    Create MobileNetV2 model with custom top layers
    """
    print("\nCreating model...")
    
    # Load pre-trained MobileNetV2
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=(224, 224, 3),
        include_top=False,
        weights='imagenet'
    )
    
    # Freeze base model weights
    base_model.trainable = False
    
    # Create new model
    model = tf.keras.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    # Compile
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    print(model.summary())
    
    return model, base_model

def train_model(model, X_train, y_train, X_val, y_val):
    """
    Train the model
    """
    print("\n" + "="*70)
    print("Training Model")
    print("="*70)
    
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=3,
            min_lr=1e-7
        ),
        keras.callbacks.ModelCheckpoint(
            'models/best_skin_model.h5',
            monitor='val_accuracy',
            save_best_only=True
        )
    ]
    
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=30,
        batch_size=32,
        callbacks=callbacks,
        verbose=1
    )
    
    return history

def evaluate_model(model, X_test, y_test, categories):
    """
    Evaluate model on test set
    """
    print("\n" + "="*70)
    print("Model Evaluation")
    print("="*70)
    
    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
    
    print(f"\nTest Loss: {loss:.4f}")
    print(f"Test Accuracy: {accuracy*100:.2f}%")
    
    # Per-class accuracy
    predictions = model.predict(X_test)
    pred_labels = np.argmax(predictions, axis=1)
    
    print("\nPer-Class Accuracy:")
    print("-"*40)
    
    for idx, category in enumerate(categories):
        mask = y_test == idx
        if mask.sum() > 0:
            class_acc = (pred_labels[mask] == y_test[mask]).mean() * 100
            print(f"{category}: {class_acc:.2f}%")
    
    print("-"*40)

def plot_training_history(history):
    """
    Plot training history
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Accuracy
    axes[0].plot(history.history['accuracy'], label='Train Accuracy')
    axes[0].plot(history.history['val_accuracy'], label='Val Accuracy')
    axes[0].set_title('Model Accuracy')
    axes[0].set_ylabel('Accuracy')
    axes[0].set_xlabel('Epoch')
    axes[0].legend()
    axes[0].grid()
    
    # Loss
    axes[1].plot(history.history['loss'], label='Train Loss')
    axes[1].plot(history.history['val_loss'], label='Val Loss')
    axes[1].set_title('Model Loss')
    axes[1].set_ylabel('Loss')
    axes[1].set_xlabel('Epoch')
    axes[1].legend()
    axes[1].grid()
    
    plt.tight_layout()
    plt.savefig('training_history.png', dpi=150)
    print("\n✅ Saved: training_history.png")
    plt.show()

def save_model_info(model, categories, category_to_idx, accuracy):
    """
    Save model metadata
    """
    info = {
        'model_name': 'MobileNetV2_SkinDisease',
        'input_size': [224, 224, 3],
        'num_classes': len(categories),
        'categories': categories,
        'category_to_idx': category_to_idx,
        'test_accuracy': float(accuracy),
        'framework': 'tensorflow'
    }
    
    with open('models/model_info.json', 'w') as f:
        json.dump(info, f, indent=2)
    
    print("✅ Saved: model_info.json")

if __name__ == '__main__':
    print("\n" + "="*70)
    print("Mirror Analyzer - Phase 3: Train Vision SLM")
    print("="*70)
    
    # Create models folder
    Path('models').mkdir(exist_ok=True)
    
    # Load data
    X, y, categories, category_to_idx = load_data(augmented=True)
    
    # Split data: 70% train, 15% val, 15% test
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.15, random_state=42, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.176, random_state=42, stratify=y_temp
    )
    
    print(f"\nData split:")
    print(f"  Train: {X_train.shape}")
    print(f"  Val:   {X_val.shape}")
    print(f"  Test:  {X_test.shape}")
    
    # Create model
    model, base_model = create_model(num_classes=len(categories))
    
    # Train
    history = train_model(model, X_train, y_train, X_val, y_val)
    
    # Evaluate
    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
    evaluate_model(model, X_test, y_test, categories)
    
    # Plot
    print("\nGenerating training history plots...")
    plot_training_history(history)
    
    # Save model
    print("\nSaving model...")
    model.save('models/skin_disease_model.h5')
    print("✅ Saved: skin_disease_model.h5")
    
    # Save info
    save_model_info(model, categories, category_to_idx, accuracy)
    
    print("\n" + "="*70)
    print("Next: Run convert_to_tflite.py")
    print("="*70)
