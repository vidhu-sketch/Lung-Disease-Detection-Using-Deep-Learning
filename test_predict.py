import requests
import json
from pathlib import Path

# Test image path
test_image = r"X:\Lung-Cancer-Prediction-using-CNN-and-Transfer-Learning-main\Lung-Cancer-Prediction-using-CNN-and-Transfer-Learning-main\dataset\test\normal\10.png"

# Upload to API
with open(test_image, 'rb') as img:
    files = {'file': img}
    response = requests.post('http://127.0.0.1:5000/api/predict', files=files)

result = response.json()

# Pretty print the output
print("=" * 70)
print("LUNG CANCER PREDICTION - API OUTPUT")
print("=" * 70)
print(json.dumps(result, indent=2))
print("=" * 70)
