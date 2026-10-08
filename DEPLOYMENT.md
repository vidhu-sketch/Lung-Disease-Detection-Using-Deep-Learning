# Lung Cancer Prediction - Deployment Guide

This guide explains how to deploy the lung cancer prediction model.

## Deployment Options

### Option 1: Standalone Prediction Script (Recommended for batch processing)

Use the `predict.py` script to make predictions on images programmatically.

#### Installation
```bash
pip install -r requirements.txt
```

#### Usage

**Make prediction on a single image:**
```python
from predict import load_model, predict_image

model = load_model()
result = predict_image(model, 'path/to/image.jpg', display=True)
print(f"Predicted: {result['predicted_class']}")
print(f"Confidence: {result['confidence']}%")
```

**Make predictions on a directory:**
```python
from predict import load_model, predict_directory

model = load_model()
results = predict_directory(model, 'dataset/test/normal')
for result in results:
    print(f"{result['image_path']}: {result['predicted_class']}")
```

**From command line:**
```bash
python predict.py
```

This will process all images in `dataset/test/normal` and display predictions.

---

### Option 2: Web Application (Recommended for interactive use)

Deploy as a Flask web application with a user-friendly interface.

#### Installation
```bash
pip install -r requirements.txt
pip install flask
```

#### Run the web server
```bash
python app.py
```

The web application will start at: **http://localhost:5000**

#### Features
- Drag-and-drop image upload
- Real-time predictions
- Confidence visualization
- All predictions breakdown
- Responsive design

#### API Endpoints

**POST** `/api/predict`
- Upload image and get prediction
- Returns: predicted class, confidence, all predictions, base64 image

**GET** `/api/health`
- Check server health and model status
- Returns: server status, model loaded status, available classes

---

### Option 3: API-only Deployment

Use the Flask app as a REST API for integration with other systems.

#### Example API call with cURL
```bash
curl -X POST http://localhost:5000/api/predict \
  -F "file=@path/to/image.jpg"
```

#### Example API call with Python
```python
import requests

with open('image.jpg', 'rb') as f:
    response = requests.post('http://localhost:5000/api/predict', 
                            files={'file': f})
    
result = response.json()
print(f"Predicted: {result['predicted_class']}")
print(f"Confidence: {result['confidence']}%")
```

#### Example API Response
```json
{
  "success": true,
  "predicted_class": "Normal",
  "class_index": 2,
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

## Production Deployment

### Using Gunicorn (for production)

```bash
pip install gunicorn
gunicorn -w 4 -b 0.0.0.0:5000 app:app
```

### Using Docker

Create a `Dockerfile`:
```dockerfile
FROM python:3.9

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

EXPOSE 5000
CMD ["python", "app.py"]
```

Build and run:
```bash
docker build -t lung-cancer-predictor .
docker run -p 5000:5000 lung-cancer-predictor
```

### Using Cloud Platforms

#### AWS Lambda + API Gateway
- Export model to SavedModel format
- Create Lambda function handler
- Deploy via API Gateway

#### Google Cloud Run
```bash
gcloud run deploy lung-cancer-predictor \
  --source . \
  --platform managed \
  --region us-central1
```

#### Azure App Service
```bash
az webapp up --name lung-cancer-predictor --runtime PYTHON:3.9
```

---

## Model Information

- **Architecture**: Xception with Transfer Learning
- **Input Size**: 350x350 pixels
- **Classes**: 
  1. Adenocarcinoma
  2. Large Cell Carcinoma
  3. Normal
  4. Squamous Cell Carcinoma
- **Accuracy**: 93%
- **Model File**: `best_model.hdf5` (80MB)

---

## Performance Metrics

- **Inference Time**: ~3-5 seconds per image
- **Memory Usage**: ~2GB RAM
- **GPU**: Optional (faster with GPU)

---

## Troubleshooting

### Model not loading
- Ensure `best_model.hdf5` is in the project directory
- Check TensorFlow version compatibility

### Web app not starting
```bash
# Install Flask if missing
pip install flask

# Check port 5000 is available
netstat -ano | findstr :5000  # Windows
lsof -i :5000                   # Linux/Mac
```

### Image upload errors
- Check file format (PNG, JPG, JPEG, GIF, BMP)
- Maximum file size: 16MB
- Ensure proper file permissions

---

## Next Steps

1. **Test with your data**: Upload sample chest X-rays
2. **Fine-tune the model**: Retrain on your specific dataset if needed
3. **Set up monitoring**: Log predictions for model performance tracking
4. **Implement security**: Add authentication for API endpoints

---

## Support & Documentation

- TensorFlow: https://tensorflow.org
- Flask: https://flask.palletsprojects.com
- Model: Xception pre-trained on ImageNet
