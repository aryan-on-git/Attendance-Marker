"""
High-accuracy face recognition prediction script.
Loads .pkl model and predicts class with confidence score.
"""

import pickle
from pathlib import Path
from typing import Tuple, Optional

import cv2
import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "face_classifier.pkl"
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def enhance_image(img_array: np.ndarray) -> np.ndarray:
    """Enhance image for better feature extraction."""
    if len(img_array.shape) == 3:
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    else:
        gray = img_array
    
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    enhanced = enhanced.astype(np.float32) / 255.0
    
    return enhanced


def extract_features(img_array: np.ndarray, img_size: Tuple[int, int]) -> np.ndarray:
    """Extract enhanced features from image."""
    img_resized = cv2.resize(img_array, img_size)
    
    if len(img_resized.shape) == 2:
        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_GRAY2RGB)
    else:
        img_rgb = img_resized
    
    gray_enhanced = enhance_image(img_rgb)
    
    features = gray_enhanced.flatten().astype(np.float32)
    
    sobelx = cv2.Sobel(gray_enhanced, cv2.CV_32F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray_enhanced, cv2.CV_32F, 0, 1, ksize=3)
    edge_features = np.concatenate([
        sobelx.flatten()[:1024],
        sobely.flatten()[:1024]
    ])
    
    diff_h = np.diff(gray_enhanced, axis=1).flatten()[:1024]
    diff_v = np.diff(gray_enhanced, axis=0).flatten()[:1024]
    
    combined_features = np.concatenate([
        features[:4096],
        edge_features,
        diff_h,
        diff_v
    ]).astype(np.float32)
    
    return combined_features


def load_model() -> Optional[dict]:
    """Load the trained .pkl model."""
    if not MODEL_PATH.exists():
        print(f"❌ Model not found at {MODEL_PATH}")
        print("   Please train the model first using: python scripts/train_model_pkl.py")
        return None
    
    try:
        with open(MODEL_PATH, 'rb') as f:
            model_data = pickle.load(f)
        return model_data
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return None


def predict_image(image_path: str, model_data: dict, confidence_threshold: float = 0.3) -> Tuple[Optional[str], float]:
    """
    Predict the class of an image.
    
    Args:
        image_path: Path to the image file
        model_data: Loaded model data from .pkl
        confidence_threshold: Minimum confidence score (0.0-1.0)
    
    Returns:
        Tuple of (class_name, confidence_score) or (None, 0.0) if prediction fails
    """
    image_path = Path(image_path)
    
    if not image_path.exists():
        print(f"❌ Image not found: {image_path}")
        return None, 0.0
    
    if image_path.suffix.lower() not in IMAGE_EXTS:
        print(f"❌ Unsupported image format: {image_path.suffix}")
        return None, 0.0
    
    try:
        # Load image
        img = Image.open(image_path).convert("RGB")
        img_array = np.array(img)
        
        # Extract features
        features = extract_features(img_array, model_data['image_size'])
        features_scaled = model_data['scaler'].transform([features])[0]
        
        # Predict with probability
        classifier = model_data['classifier']
        prediction = classifier.predict([features_scaled])[0]
        probabilities = classifier.predict_proba([features_scaled])[0]
        
        # Get class name and confidence
        idx_to_label = {v: k for k, v in model_data['labels'].items()}
        predicted_class = idx_to_label[prediction]
        confidence = float(probabilities[prediction])
        
        return predicted_class, confidence
    
    except Exception as e:
        print(f"❌ Error predicting image: {e}")
        return None, 0.0


def predict_and_report(image_path: str) -> None:
    """
    Predict image class and print detailed report.
    """
    # Load model
    model_data = load_model()
    if model_data is None:
        return
    
    print(f"\n🔍 Predicting: {image_path}")
    print("-" * 50)
    
    # Make prediction
    predicted_class, confidence = predict_image(image_path, model_data)
    
    if predicted_class is None:
        print("❌ Prediction failed")
        return
    
    # Display results
    print(f"✅ Predicted Class: {predicted_class}")
    print(f"📊 Confidence Score: {confidence:.2%}")
    
    # Color-coded confidence
    if confidence >= 0.9:
        status = "✓ Very High Confidence"
    elif confidence >= 0.7:
        status = "✓ High Confidence"
    elif confidence >= 0.5:
        status = "⚠ Medium Confidence"
    else:
        status = "⚠ Low Confidence"
    
    print(f"   {status}")
    print("-" * 50)


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python predict.py <image_path>")
        print("Example: python predict.py /path/to/image.jpg")
        sys.exit(1)
    
    predict_and_report(sys.argv[1])
