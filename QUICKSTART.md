# 🫁 Lung Cancer Prediction - Quick Start Guide

## What's Deployed

Your lung cancer prediction model is now deployed and ready to use in **3 different ways**:

---

## 1️⃣ **Web Application** (Live Now!)
### Access: http://localhost:5000

**Features:**
- ✅ Drag-and-drop image upload
- ✅ Real-time predictions with visual results
- ✅ Confidence percentage
- ✅ Full prediction breakdown for all 4 classes
- ✅ Mobile-friendly interface

**How to use:**
1. Open http://localhost:5000 in your browser
2. Upload a chest X-ray image (PNG, JPG, JPEG, GIF, BMP)
3. Click "Make Prediction"
4. View results with confidence percentage

---

## 2️⃣ **Python Script** (Batch Processing)
### File: `predict.py`

**Use for:**
- Processing multiple images
- Integration with other Python code
- Automated predictions

**Example usage:**
```python
from predict import load_model, predict_image

# Load the model
model = load_model()

# Predict on a single image
result = predict_image(model, 'path/to/image.jpg', display=True)
print(f"Predicted: {result['predicted_class']}")
print(f"Confidence: {result['confidence']}%")
```

**Run directly:**
```bash
python predict.py
```

---

## 3️⃣ **REST API** (For Integration)
### Endpoint: http://localhost:5000/api/predict

**Use for:**
- Mobile apps
- Desktop applications
- Integration with other systems
- Microservices architecture

**Example API call (Python):**
```python
import requests

with open('image.jpg', 'rb') as f:
    response = requests.post('http://localhost:5000/api/predict', 
                            files={'file': f})

result = response.json()
print(f"Predicted: {result['predicted_class']}")
print(f"Confidence: {result['confidence']}%")
```

**Example API call (cURL):**
```bash
curl -X POST http://localhost:5000/api/predict \
  -F "file=@image.jpg"
```

**API Response:**
```json
{
  "success": true,
  "predicted_class": "Normal",
  "confidence": 95.42,
  "all_predictions": {
    "Adenocarcinoma": 2.15,
    "Large Cell Carcinoma": 1.20,
    "Normal": 95.42,
    "Squamous Cell Carcinoma": 1.23
  },
  "image": "data:image/png;base64,..."
}
```

---

## 📋 Model Information

| Property | Value |
|----------|-------|
| **Architecture** | Xception (Transfer Learning) |
| **Input Size** | 350x350 pixels |
| **Model Accuracy** | 93% |
| **Classes** | 4 (Adenocarcinoma, Large Cell Carcinoma, Normal, Squamous Cell Carcinoma) |
| **Model File** | best_model.hdf5 (80MB) |
| **Inference Time** | ~3-5 seconds per image |

---

## 📁 Project Files

```
Lung-Cancer-Prediction-using-CNN-and-Transfer-Learning-main/
├── app.py                      # Flask web application
├── predict.py                  # Standalone prediction script
├── requirements.txt            # Python dependencies
├── best_model.hdf5            # Pre-trained model weights
├── Lung Cancer Pred.ipynb     # Original Jupyter notebook
├── DEPLOYMENT.md              # Detailed deployment guide
├── QUICKSTART.md              # This file
├── templates/
│   └── index.html             # Web interface
└── dataset/
    ├── train/                 # Training images
    ├── test/                  # Test images
    └── valid/                 # Validation images
```

---

## 🚀 Currently Running Services

| Service | URL | Port | Status |
|---------|-----|------|--------|
| Web App | http://localhost:5000 | 5000 | ✅ Running |
| Jupyter Notebook | http://localhost:8888 | 8888 | ✅ Running |

---

## 💡 Next Steps

### 1. Test the Web App
- Open http://localhost:5000
- Upload a test image from `dataset/test/`
- View the prediction results

### 2. Try the Python Script
```bash
cd "path/to/project"
python predict.py
```

### 3. Integrate the API
- Use the REST API in your applications
- See example code above

### 4. Deploy to Cloud
- Follow instructions in `DEPLOYMENT.md`
- Options: AWS, Google Cloud, Azure, Docker

---

## 🔍 Testing with Sample Images

Sample images are available in:
- `dataset/test/adenocarcinoma/`
- `dataset/test/large.cell.carcinoma/`
- `dataset/test/normal/`
- `dataset/test/squamous.cell.carcinoma/`

Try uploading images from these folders to test the predictions.

---

## ⚠️ Important Notes

1. **Model Performance**: The model achieves 93% accuracy on the test set
2. **Medical Disclaimer**: This is for demonstration purposes only and should not be used for actual medical diagnosis
3. **Image Format**: Use chest X-ray images (350x350 or similar dimensions)
4. **Privacy**: Images are not stored permanently (deleted after prediction)

---

## 🛠️ Troubleshooting

### Port 5000 already in use
```bash
# Find process using port 5000
netstat -ano | findstr :5000  # Windows
lsof -i :5000                   # Linux/Mac

# Kill the process or use a different port
python -m flask run --port 5001
```

### Model takes too long to load
- First load can take 30-60 seconds
- Subsequent predictions are faster
- Consider using GPU for faster inference

### Images not uploading
- Check file format (PNG, JPG, JPEG, GIF, BMP)
- Maximum file size: 16MB
- Ensure proper read permissions

---

## 📚 Additional Resources

- **TensorFlow**: https://tensorflow.org
- **Flask**: https://flask.palletsprojects.com
- **Xception Model**: https://keras.io/api/applications/xception/

---

## ✅ Deployment Checklist

- [x] Web application created
- [x] Python prediction script created
- [x] Requirements file generated
- [x] Documentation created
- [x] Web server running on port 5000
- [x] API endpoints configured
- [x] Sample HTML interface deployed
- [x] Model loaded and ready

**Status: ✅ FULLY DEPLOYED**

---

**Happy predicting! 🎉**
