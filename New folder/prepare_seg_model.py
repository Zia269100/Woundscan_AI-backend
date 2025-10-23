import urllib.request
import os

# Make sure models folder exists
if not os.path.exists("models"):
    os.makedirs("models")

# Download TFLite DeepLabV3+ model
url = "https://tfhub.dev/tensorflow/lite-model/deeplabv3/1/default/1?lite-format=tflite"
output_path = "models/segmentation_model.tflite"

print("Downloading DeepLabV3+ TFLite model...")

urllib.request.urlretrieve(url, output_path)

print(f"✅ Model downloaded and saved at {output_path}")
