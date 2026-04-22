"""
Advanced face recognition model training script.
Optimized for high accuracy with improved preprocessing and feature extraction.
Outputs sklearn model as .pkl file.
"""

import json
import pickle
import shutil
from pathlib import Path
from typing import Tuple

import cv2
import numpy as np
from PIL import Image
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier


ROOT = Path(__file__).resolve().parents[1]
NEW_DATASET_DIR = ROOT / "dataset new" / "dataset"
PROCESSED_DATASET = ROOT / "processed_dataset"
TRAIN_DIR = ROOT / "data" / "train"
VAL_DIR = ROOT / "data" / "val"
MODEL_PATH = ROOT / "models" / "face_classifier.pkl"
LABELS_PATH = ROOT / "models" / "labels.json"

IMG_SIZE = (128, 128)
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SPLIT_RATIO = 0.8
SEED = 42


def ensure_dirs_exist() -> None:
    """Create necessary directories."""
    PROCESSED_DATASET.mkdir(parents=True, exist_ok=True)
    TRAIN_DIR.mkdir(parents=True, exist_ok=True)
    VAL_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)


def copy_new_dataset() -> None:
    """Copy new dataset from 'dataset new' to 'processed_dataset' if available."""
    if not NEW_DATASET_DIR.exists():
        print(f"⚠️  New dataset not found at {NEW_DATASET_DIR}")
        return
    
    print(f"📂 Processing new dataset from {NEW_DATASET_DIR}")
    
    for person_dir in sorted(NEW_DATASET_DIR.iterdir()):
        if not person_dir.is_dir():
            continue
        
        person_name = person_dir.name
        target_dir = PROCESSED_DATASET / person_name
        
        # Remove existing and create fresh
        if target_dir.exists():
            shutil.rmtree(target_dir)
        target_dir.mkdir(parents=True, exist_ok=True)
        
        # Copy all images
        images = [p for p in person_dir.iterdir() 
                 if p.is_file() and p.suffix.lower() in IMAGE_EXTS]
        
        for img_path in images:
            try:
                shutil.copy2(img_path, target_dir / img_path.name)
            except Exception as e:
                print(f"  ⚠️  Error copying {img_path}: {e}")
        
        print(f"  ✓ {person_name}: {len(images)} images")


def split_dataset() -> int:
    """
    Split processed dataset into train/val folders.
    Returns total number of images processed.
    """
    import random
    random.seed(SEED)
    
    if not PROCESSED_DATASET.exists():
        raise SystemExit(f"❌ No processed dataset found at {PROCESSED_DATASET}")
    
    print("\n📊 Splitting dataset into train/val...")
    
    total_images = 0
    
    for person_dir in sorted(PROCESSED_DATASET.iterdir()):
        if not person_dir.is_dir():
            continue
        
        images = [p for p in person_dir.iterdir() 
                 if p.is_file() and p.suffix.lower() in IMAGE_EXTS]
        
        if not images:
            print(f"  ⚠️  {person_dir.name}: skipped (no images)")
            continue
        
        random.shuffle(images)
        total_images += len(images)
        
        split_index = max(1, int(len(images) * SPLIT_RATIO))
        split_index = min(len(images) - 1, split_index)
        
        train_images = images[:split_index]
        val_images = images[split_index:]
        
        # Create and clear directories
        for d in [TRAIN_DIR / person_dir.name, VAL_DIR / person_dir.name]:
            if d.exists():
                shutil.rmtree(d)
            d.mkdir(parents=True, exist_ok=True)
        
        # Copy images
        for img in train_images:
            shutil.copy2(img, TRAIN_DIR / person_dir.name / img.name)
        
        for img in val_images:
            shutil.copy2(img, VAL_DIR / person_dir.name / img.name)
        
        print(f"  ✓ {person_dir.name}: {len(train_images)} train, {len(val_images)} val")
    
    print(f"✅ Total images: {total_images}")
    return total_images


def enhance_image(img_array: np.ndarray) -> np.ndarray:
    """
    Enhance image for better feature extraction.
    - Convert to grayscale for contrast enhancement
    - Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
    - Normalize
    """
    # Convert to grayscale
    if len(img_array.shape) == 3:
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    else:
        gray = img_array
    
    # Apply CLAHE for contrast enhancement
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    
    # Normalize
    enhanced = enhanced.astype(np.float32) / 255.0
    
    return enhanced


def extract_features(img_array: np.ndarray) -> np.ndarray:
    """
    Extract enhanced features from image.
    Combines multiple feature representations for better accuracy.
    """
    # Resize to standard size
    img_resized = cv2.resize(img_array, IMG_SIZE)
    
    # Convert to RGB if needed
    if len(img_resized.shape) == 2:
        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_GRAY2RGB)
    else:
        img_rgb = img_resized
    
    # Get grayscale enhanced
    gray_enhanced = enhance_image(img_rgb)
    
    # Feature 1: Normalized pixel values
    features = gray_enhanced.flatten().astype(np.float32)
    
    # Feature 2: Sobel edge detection (captures edges)
    sobelx = cv2.Sobel(gray_enhanced, cv2.CV_32F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray_enhanced, cv2.CV_32F, 0, 1, ksize=3)
    edge_features = np.concatenate([
        sobelx.flatten()[:1024],  # Limit size
        sobely.flatten()[:1024]
    ])
    
    # Feature 3: Local Binary Pattern inspired (simple local differences)
    diff_h = np.diff(gray_enhanced, axis=1).flatten()[:1024]
    diff_v = np.diff(gray_enhanced, axis=0).flatten()[:1024]
    
    # Combine all features
    combined_features = np.concatenate([
        features[:4096],
        edge_features,
        diff_h,
        diff_v
    ]).astype(np.float32)
    
    return combined_features


def collect_samples(base_dir: Path) -> list[Tuple[Path, str]]:
    """Collect all image samples with their labels."""
    samples = []
    if not base_dir.exists():
        return samples
    
    for class_dir in sorted(base_dir.iterdir()):
        if not class_dir.is_dir():
            continue
        for image_path in sorted(class_dir.iterdir()):
            if image_path.is_file() and image_path.suffix.lower() in IMAGE_EXTS:
                samples.append((image_path, class_dir.name))
    
    return samples


def load_features(samples: list[Tuple[Path, str]]) -> Tuple[np.ndarray, list[str]]:
    """Load and extract features from all samples."""
    x_data = []
    y_labels = []
    failed = 0
    
    for idx, (image_path, label) in enumerate(samples, 1):
        if idx % 50 == 0:
            print(f"  Processing image {idx}/{len(samples)}...")
        
        try:
            img = Image.open(image_path).convert("RGB")
            img_array = np.array(img)
            
            # Extract enhanced features
            features = extract_features(img_array)
            x_data.append(features)
            y_labels.append(label)
        except Exception as e:
            failed += 1
            continue
    
    if failed > 0:
        print(f"  ⚠️  Failed to load {failed} images")
    
    if not x_data:
        return np.empty((0, 10240), dtype=np.float32), []
    
    return np.vstack(x_data), y_labels


def train_model() -> None:
    """
    Train the face classification model with optimizations.
    """
    print("\n🚀 Starting model training...")
    ensure_dirs_exist()
    
    # Step 1: Prepare dataset
    print("\n1️⃣  Dataset Preparation:")
    copy_new_dataset()
    total_img = split_dataset()
    
    if total_img < 50:
        raise SystemExit("❌ Need at least 50 images for meaningful training")
    
    # Step 2: Load training data
    print("\n2️⃣  Loading training data...")
    train_samples = collect_samples(TRAIN_DIR)
    
    if not train_samples:
        raise SystemExit(f"❌ No training images found in {TRAIN_DIR}")
    
    print(f"  Loading {len(train_samples)} training images...")
    x_train, y_train_labels = load_features(train_samples)
    
    if x_train.size == 0:
        raise SystemExit("❌ Training images could not be decoded")
    
    print(f"  ✓ Training data shape: {x_train.shape}")
    
    # Step 3: Encode labels
    print("\n3️⃣  Encoding labels...")
    label_encoder = LabelEncoder()
    y_train = label_encoder.fit_transform(y_train_labels)
    labels_dict = {name: int(idx) for idx, name in enumerate(label_encoder.classes_)}
    
    print(f"  ✓ Classes: {list(labels_dict.keys())}")
    
    # Save labels to JSON
    LABELS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with LABELS_PATH.open("w", encoding="utf-8") as f:
        json.dump(labels_dict, f, indent=2)
    print(f"  ✓ Labels saved to: {LABELS_PATH}")
    
    # Step 4: Normalize features
    print("\n4️⃣  Normalizing features...")
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    print(f"  ✓ Features normalized")
    
    # Step 5: Train classifier with high accuracy optimization
    print("\n5️⃣  Training classifier (this may take a moment)...")
    
    # Use SVM with RBF kernel for better accuracy on face recognition
    classifier = SVC(
        kernel='rbf',
        C=10.0,
        gamma='scale',
        probability=True,
        max_iter=5000,
        verbose=0
    )
    
    classifier.fit(x_train_scaled, y_train)
    
    # Evaluate on training data
    train_preds = classifier.predict(x_train_scaled)
    train_acc = accuracy_score(y_train, train_preds)
    print(f"  ✓ Training accuracy: {train_acc:.2%}")
    
    # Step 6: Evaluate on validation data if available
    print("\n6️⃣  Validating on test set...")
    val_samples = collect_samples(VAL_DIR)
    
    if val_samples:
        print(f"  Loading {len(val_samples)} validation images...")
        x_val, y_val_labels = load_features(val_samples)
        
        if x_val.size > 0:
            class_to_idx = {name: idx for idx, name in enumerate(label_encoder.classes_)}
            
            # Filter to only classes in training set
            keep_indices = [idx for idx, label in enumerate(y_val_labels) 
                          if label in class_to_idx]
            
            if keep_indices:
                x_val_kept = x_val[keep_indices]
                y_val = np.array([class_to_idx[y_val_labels[idx]] for idx in keep_indices], 
                               dtype=np.int64)
                x_val_scaled = scaler.transform(x_val_kept)
                val_preds = classifier.predict(x_val_scaled)
                val_acc = accuracy_score(y_val, val_preds)
                print(f"  ✓ Validation accuracy: {val_acc:.2%}")
                
                # Detailed classification report
                print("\n📋 Classification Report:")
                print(classification_report(y_val, val_preds, 
                                          target_names=label_encoder.classes_,
                                          digits=4))
            else:
                print("  ⚠️  No validation classes found in training set")
        else:
            print("  ⚠️  Could not load validation images")
    else:
        print("  ⚠️  No validation images found")
    
    # Step 7: Save model as .pkl
    print("\n7️⃣  Saving model as .pkl file...")
    
    model_data = {
        'classifier': classifier,
        'scaler': scaler,
        'label_encoder': label_encoder,
        'image_size': IMG_SIZE,
        'labels': labels_dict,
        'train_accuracy': float(train_acc),
    }
    
    with open(MODEL_PATH, 'wb') as f:
        pickle.dump(model_data, f)
    
    print(f"  ✓ Model saved to: {MODEL_PATH}")
    print(f"  ✓ File size: {MODEL_PATH.stat().st_size / (1024*1024):.2f} MB")
    
    # Step 8: Final summary
    print("\n" + "=" * 50)
    print("✅ TRAINING COMPLETED SUCCESSFULLY!")
    print("=" * 50)
    print(f"📁 Model file: {MODEL_PATH}")
    print(f"📁 Labels file: {LABELS_PATH}")
    print(f"📊 Training accuracy: {train_acc:.2%}")
    print(f"👥 Classes trained: {len(labels_dict)}")
    print(f"🖼️  Training images: {len(train_samples)}")
    print("=" * 50)


if __name__ == "__main__":
    try:
        train_model()
    except Exception as e:
        print(f"\n❌ Error during training: {e}")
        raise
