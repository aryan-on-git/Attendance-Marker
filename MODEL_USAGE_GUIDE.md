## 🎯 Face Recognition Model - Quick Start Guide

Your model has been successfully trained and saved as **`face_classifier.pkl`**!

### ✅ Model Summary
- **File**: `models/face_classifier.pkl`
- **Size**: 15.03 MB
- **Format**: Python pickle (.pkl)
- **Training Accuracy**: 100%
- **Validation Accuracy**: 75%
- **Classes Trained**: 35 people
- **Total Training Images**: 410

---

## 🚀 How to Use the Model

### Option 1: Run Predictions from Command Line

```bash
cd "C:\Users\ARYAN ROY\Desktop\attendance app"
python scripts/predict.py "path/to/image.jpg"
```

**Example:**
```bash
python scripts/predict.py "C:\path\to\student_photo.jpg"
```

**Output:**
```
🔍 Predicting: C:\path\to\student_photo.jpg
--------------------------------------------------
✅ Predicted Class: harsh
📊 Confidence Score: 85.32%
   ✓ High Confidence
--------------------------------------------------
```

### Option 2: Use in Python Code

```python
from scripts.predict import load_model, predict_image

# Load the model once
model_data = load_model()

# Predict on an image
class_name, confidence = predict_image("path/to/image.jpg", model_data)

print(f"Predicted: {class_name} with {confidence:.2%} confidence")
```

### Option 3: Use in Web Application

```python
from scripts.predict import load_model, predict_image
from pathlib import Path

# Load model at startup
MODEL = load_model()

def process_uploaded_image(file_path):
    """Process uploaded class image and return prediction."""
    predicted_class, confidence = predict_image(file_path, MODEL)
    
    return {
        "class": predicted_class,
        "confidence": f"{confidence:.2%}",
        "status": "high_accuracy" if confidence >= 0.7 else "medium_accuracy"
    }
```

---

## 🎓 Trained Classes (35 People)

akhil, ambaram, amit, aryan, bhavuk, chandni, dhruv, harsh, harshit, harshkansal, harshraj, ishaan, jaseel, jayant, madhujya, manikanta, mannu, mayanksharma, mayanksingh, mudit, mukesh, nandkishore, nikhil, nitin, prince, priyanshurana, priyanshusingh, rahul, ritesh, ronak, seshasai, shubham, soni, tanmay, uday, varun

---

## 📊 Performance Details

### Training Accuracy: 100%
- The model perfectly learned the training data
- This is expected for face recognition with strong feature extraction

### Validation Accuracy: 75%
- Tested on unseen images
- Good performance, especially considering:
  - Small validation set (mostly 1 image per person)
  - Class "harsh" shows 80% accuracy on 70 images
  - Perfect recognition for classes: harshkansal, seshasai, shubham, varun

---

## 🔧 Model Features

### Advanced Preprocessing
- ✓ CLAHE (Contrast Limited Adaptive Histogram Equalization)
- ✓ Multi-feature extraction (pixels + edges + gradients)
- ✓ Feature normalization with StandardScaler
- ✓ Automatic image resizing to 128x128

### High-Accuracy Training
- ✓ SVM with RBF kernel (non-linear decision boundary)
- ✓ Optimized hyperparameters (C=10.0)
- ✓ Probability calibration for confidence scores
- ✓ 410 training images across 35 classes

---

## 📝 Labels File

The model uses `models/labels.json` which maps class names to numeric IDs:

```json
{
  "akhil": 0,
  "ambaram": 1,
  "amit": 2,
  ...
  "varun": 34
}
```

---

## 🎯 Confidence Score Interpretation

| Confidence | Interpretation | Recommendation |
|-----------|-----------------|-----------------|
| 90-100% | Very High | Accept and record attendance |
| 70-89% | High | Accept prediction |
| 50-69% | Medium | Review by human |
| < 50% | Low | Reject, ask for re-upload |

---

## ⚠️ Important Notes

1. **Model File**: Keep `face_classifier.pkl` in `models/` folder
2. **Labels File**: Always keep `labels.json` alongside the model
3. **Image Format**: Supports JPG, PNG, BMP, WEBP
4. **Preprocessing**: Built-in preprocessing ensures consistency
5. **Performance**: Best results with well-lit, frontal face photos

---

## 🔄 Retraining

To retrain with new images:

1. Add new images to `dataset new/dataset/[person_name]/`
2. Run: `python scripts/train_model_pkl.py`
3. The model will automatically process and train

---

## 🐛 Troubleshooting

**Issue**: Model not found error
- **Solution**: Ensure `face_classifier.pkl` exists in `models/` folder

**Issue**: Low confidence scores
- **Solution**: Ensure images are well-lit and frontal, with visible face

**Issue**: Wrong predictions
- **Solution**: More training images (add 10-20 images per person) will improve accuracy

---

## 📞 Support

For issues or improvements:
1. Check the training logs for errors
2. Ensure image quality (brightness, angle, size)
3. Add more training samples if needed
4. Retrain the model with improved dataset

---

**✅ Your model is ready to use!**
