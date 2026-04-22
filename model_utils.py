"""
Face Recognition Model Utility Module.
Easy integration for attendance applications.

Usage:
    from model_utils import FaceRecognizer
    
    recognizer = FaceRecognizer()
    result = recognizer.predict_from_file("path/to/image.jpg")
    print(f"Student: {result['class']}, Confidence: {result['confidence']:.2%}")
"""

import pickle
from pathlib import Path
from typing import Dict, Optional, Tuple

import cv2
import numpy as np
from PIL import Image


class FaceRecognizer:
    """High-level face recognition utility for attendance systems."""
    
    def __init__(self, model_path: Optional[str] = None):
        """
        Initialize the face recognizer.
        
        Args:
            model_path: Path to the .pkl model file. 
                       If None, uses default path.
        """
        self.model_path = Path(model_path) if model_path else Path(__file__).parent / "models" / "face_classifier.pkl"
        self.model_data = None
        self.load_model()
    
    def load_model(self) -> bool:
        """
        Load the trained model from .pkl file.
        
        Returns:
            True if successful, False otherwise
        """
        if not self.model_path.exists():
            print(f"❌ Model not found at {self.model_path}")
            return False
        
        try:
            with open(self.model_path, 'rb') as f:
                self.model_data = pickle.load(f)
            print(f"✅ Model loaded from {self.model_path}")
            return True
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            return False
    
    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        return self.model_data is not None
    
    def _enhance_image(self, img_array: np.ndarray) -> np.ndarray:
        """Enhance image for better feature extraction."""
        if len(img_array.shape) == 3:
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        else:
            gray = img_array
        
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        enhanced = enhanced.astype(np.float32) / 255.0
        
        return enhanced
    
    def _extract_features(self, img_array: np.ndarray) -> np.ndarray:
        """Extract features from image."""
        img_resized = cv2.resize(img_array, self.model_data['image_size'])
        
        if len(img_resized.shape) == 2:
            img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_GRAY2RGB)
        else:
            img_rgb = img_resized
        
        gray_enhanced = self._enhance_image(img_rgb)
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
    
    def predict_from_file(self, image_path: str) -> Dict:
        """
        Predict class from image file.
        
        Args:
            image_path: Path to the image file
        
        Returns:
            Dictionary with keys:
            - 'success': bool
            - 'class': str (predicted class name)
            - 'confidence': float (0.0-1.0)
            - 'error': str (if any error occurred)
        """
        if not self.is_loaded():
            return {'success': False, 'error': 'Model not loaded'}
        
        image_path = Path(image_path)
        
        if not image_path.exists():
            return {'success': False, 'error': f'Image not found: {image_path}'}
        
        try:
            img = Image.open(image_path).convert("RGB")
            img_array = np.array(img)
            
            features = self._extract_features(img_array)
            features_scaled = self.model_data['scaler'].transform([features])[0]
            
            classifier = self.model_data['classifier']
            prediction = classifier.predict([features_scaled])[0]
            probabilities = classifier.predict_proba([features_scaled])[0]
            
            idx_to_label = {v: k for k, v in self.model_data['labels'].items()}
            predicted_class = idx_to_label[prediction]
            confidence = float(probabilities[prediction])
            
            return {
                'success': True,
                'class': predicted_class,
                'confidence': confidence,
            }
        
        except Exception as e:
            return {'success': False, 'error': f'Prediction error: {e}'}
    
    def predict_from_array(self, img_array: np.ndarray) -> Dict:
        """
        Predict class from numpy array (e.g., from camera).
        
        Args:
            img_array: Image as numpy array (RGB)
        
        Returns:
            Dictionary with prediction results
        """
        if not self.is_loaded():
            return {'success': False, 'error': 'Model not loaded'}
        
        try:
            features = self._extract_features(img_array)
            features_scaled = self.model_data['scaler'].transform([features])[0]
            
            classifier = self.model_data['classifier']
            prediction = classifier.predict([features_scaled])[0]
            probabilities = classifier.predict_proba([features_scaled])[0]
            
            idx_to_label = {v: k for k, v in self.model_data['labels'].items()}
            predicted_class = idx_to_label[prediction]
            confidence = float(probabilities[prediction])
            
            return {
                'success': True,
                'class': predicted_class,
                'confidence': confidence,
            }
        
        except Exception as e:
            return {'success': False, 'error': f'Prediction error: {e}'}
    
    def get_all_classes(self) -> list:
        """Get list of all trained classes."""
        if not self.is_loaded():
            return []
        return list(self.model_data['labels'].keys())
    
    def get_model_info(self) -> Dict:
        """Get information about the loaded model."""
        if not self.is_loaded():
            return {'loaded': False}
        
        return {
            'loaded': True,
            'num_classes': len(self.model_data['labels']),
            'classes': self.get_all_classes(),
            'train_accuracy': self.model_data.get('train_accuracy', 'N/A'),
            'image_size': self.model_data['image_size'],
        }


# ============================================================================
# Example Usage
# ============================================================================

if __name__ == "__main__":
    # Example 1: Basic usage
    print("=" * 60)
    print("Face Recognition Model Utility - Examples")
    print("=" * 60)
    
    # Initialize recognizer
    recognizer = FaceRecognizer()
    
    # Get model info
    print("\n📊 Model Information:")
    info = recognizer.get_model_info()
    if info.get('loaded'):
        print(f"  Classes: {info['num_classes']}")
        print(f"  Training Accuracy: {info['train_accuracy']:.2%}")
        print(f"  Image Size: {info['image_size']}")
    
    # Example 2: Get all classes
    print("\n👥 Trained Classes:")
    classes = recognizer.get_all_classes()
    print(f"  Total: {len(classes)}")
    print(f"  Classes: {', '.join(classes[:5])}... (showing first 5)")
    
    # Example 3: Predict from file (if you have a test image)
    print("\n🔍 Prediction Examples:")
    print("  To predict from a file:")
    print("  result = recognizer.predict_from_file('path/to/image.jpg')")
    print("  print(f\"Predicted: {result['class']} ({result['confidence']:.2%})\")")
    
    # Example 4: Use in attendance system
    print("\n📋 Attendance System Integration:")
    print("""
    # In your Flask/FastAPI app:
    from model_utils import FaceRecognizer
    
    app = Flask(__name__)
    recognizer = FaceRecognizer()  # Load once at startup
    
    @app.route('/attendance/upload', methods=['POST'])
    def mark_attendance():
        file = request.files['image']
        result = recognizer.predict_from_file(file.filename)
        
        if result['success'] and result['confidence'] > 0.7:
            student_name = result['class']
            # Mark attendance for student_name
            return {'status': 'marked', 'student': student_name}
        else:
            return {'status': 'failed', 'reason': 'Low confidence'}
    """)
    
    print("\n" + "=" * 60)
    print("Ready to use! Import this module in your application.")
    print("=" * 60)
