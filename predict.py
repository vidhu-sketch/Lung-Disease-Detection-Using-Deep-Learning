"""
Lung Cancer Prediction - Standalone Prediction Script
Uses the pre-trained model (best_model.hdf5) to make predictions on new images
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
import matplotlib.pyplot as plt

# Configuration
IMAGE_SIZE = (350, 350)
MODEL_PATH = 'best_model.hdf5'
CLASS_LABELS = ['adenocarcinoma_left.lower.lobe_T2_N0_M0_Ib', 
                'large.cell.carcinoma_left.hilum_T2_N2_M0_IIIa',
                'normal',
                'squamous.cell.carcinoma_left.hilum_T1_N2_M0_IIIa']

def load_model():
    """Load the pre-trained model"""
    print("Loading pre-trained model...")
    
    # Build the model architecture
    pretrained_model = tf.keras.applications.Xception(
        weights='imagenet', 
        include_top=False, 
        input_shape=[*IMAGE_SIZE, 3]
    )
    pretrained_model.trainable = False
    
    model = Sequential([
        pretrained_model,
        GlobalAveragePooling2D(),
        Dense(4, activation='softmax')
    ])
    
    # Load the saved weights
    if os.path.exists(MODEL_PATH):
        model.load_weights(MODEL_PATH)
        print(f"Model loaded successfully from {MODEL_PATH}")
    else:
        print(f"ERROR: Model file {MODEL_PATH} not found!")
        return None
    
    return model

def load_and_preprocess_image(img_path):
    """Load and preprocess an image for prediction"""
    try:
        img = image.load_img(img_path, target_size=IMAGE_SIZE)
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array /= 255.0  # Normalize to [0, 1]
        return img_array
    except Exception as e:
        print(f"Error loading image: {e}")
        return None

def predict_image(model, img_path, display=True):
    """Make a prediction on a single image"""
    if not os.path.exists(img_path):
        print(f"Image file not found: {img_path}")
        return None
    
    # Preprocess the image
    img_array = load_and_preprocess_image(img_path)
    if img_array is None:
        return None
    
    # Make prediction
    predictions = model.predict(img_array, verbose=0)
    predicted_class_idx = np.argmax(predictions[0])
    predicted_label = CLASS_LABELS[predicted_class_idx]
    confidence = predictions[0][predicted_class_idx] * 100
    
    result = {
        'image_path': img_path,
        'predicted_class': predicted_label,
        'confidence': confidence,
        'all_predictions': {CLASS_LABELS[i]: float(predictions[0][i] * 100) for i in range(4)}
    }
    
    # Display result
    print("\n" + "="*60)
    print(f"Image: {os.path.basename(img_path)}")
    print(f"Predicted Class: {predicted_label}")
    print(f"Confidence: {confidence:.2f}%")
    print("\nAll Predictions:")
    for label, prob in result['all_predictions'].items():
        print(f"  {label}: {prob:.2f}%")
    print("="*60 + "\n")
    
    # Display image
    if display:
        try:
            img = image.load_img(img_path, target_size=IMAGE_SIZE)
            plt.figure(figsize=(6, 6))
            plt.imshow(img)
            plt.title(f"Predicted: {predicted_label}\nConfidence: {confidence:.2f}%")
            plt.axis('off')
            plt.tight_layout()
            plt.show()
        except Exception as e:
            print(f"Could not display image: {e}")
    
    return result

def predict_directory(model, directory_path):
    """Make predictions on all images in a directory"""
    if not os.path.isdir(directory_path):
        print(f"Directory not found: {directory_path}")
        return []
    
    results = []
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif'}
    
    print(f"Processing images in: {directory_path}")
    for filename in os.listdir(directory_path):
        if os.path.splitext(filename)[1].lower() in image_extensions:
            img_path = os.path.join(directory_path, filename)
            result = predict_image(model, img_path, display=False)
            if result:
                results.append(result)
    
    print(f"Processed {len(results)} images")
    return results

if __name__ == "__main__":
    # Load the model
    model = load_model()
    if model is None:
        exit(1)
    
    # Example: Predict on test directory
    test_dir = 'dataset/test/normal'
    if os.path.exists(test_dir):
        print(f"\nMaking predictions on images in {test_dir}...")
        results = predict_directory(model, test_dir)
        
        # Print summary
        if results:
            print("\nPrediction Summary:")
            for result in results[:5]:  # Show first 5
                print(f"  {os.path.basename(result['image_path'])}: {result['predicted_class']}")
    else:
        print(f"Test directory not found: {test_dir}")
        print("\nUsage:")
        print("  from predict import load_model, predict_image")
        print("  model = load_model()")
        print("  predict_image(model, 'path/to/image.jpg')")
