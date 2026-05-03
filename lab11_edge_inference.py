import cv2
import numpy as np
import time
import os
import urllib.request
import zipfile

# =================================================================
# Course: Data Engineering (CSIE, Tamkang University)
# Lab 11: Edge Inference & INT8 Quantization
# =================================================================

# 嘗試載入輕量版推論引擎 (Edge 專用)
try:
    import tflite_runtime.interpreter as tflite
    print("[*] Successfully loaded tflite_runtime.")
except ImportError:
    print("[!] tflite_runtime not found. Falling back to full tensorflow.lite...")
    import tensorflow.lite as tflite

def download_assets():
    """
    Downloads the pre-quantized INT8 model and test image automatically.
    """
    model_url = "https://storage.googleapis.com/download.tensorflow.org/models/tflite/mobilenet_v1_1.0_224_quant_and_labels.zip"
    img_url = "https://storage.googleapis.com/download.tensorflow.org/example_images/grace_hopper.jpg"
    
    img_name = "test_input.jpg"
    model_zip = "mobilenet_quant.zip"
    model_name = "mobilenet_v1_1.0_224_quant.tflite"
    label_name = "labels_mobilenet_quant_v1_224.txt"

    if not os.path.exists(img_name):
        print(f"[*] Downloading test image...")
        urllib.request.urlretrieve(img_url, img_name)
        
    if not os.path.exists(model_name):
        print(f"[*] Downloading INT8 Quantized MobileNet model from Google...")
        urllib.request.urlretrieve(model_url, model_zip)
        with zipfile.ZipFile(model_zip, 'r') as zip_ref:
            zip_ref.extractall(".")
        os.remove(model_zip)
        
    return img_name, model_name, label_name

def load_labels(filename):
    with open(filename, 'r') as f:
        return [line.strip() for line in f.readlines()]

if __name__ == "__main__":
    print("=== Week 11: End-to-End INT8 Edge Inference ===\n")
    
    img_path, model_path, label_path = download_assets()
    labels = load_labels(label_path)
    
    # ---------------------------------------------------------
    # TODO 1: Initialize TFLite Interpreter
    # 1. Create an Interpreter using `model_path`.
    # 2. Allocate tensors (allocate_tensors()).
    # 3. Get input and output details (get_input_details(), get_output_details()).
    # ---------------------------------------------------------
    # interpreter = ...
    
    input_details = [{'index': 0}]   # Placeholder, remove this
    output_details = [{'index': 0}]  # Placeholder, remove this
    
    print(f"[*] Model Loaded: {model_path}")

    # =========================================================
    # PHASE 1: PRE-PROCESSING (Your W9 & W10 Skills)
    # =========================================================
    t0 = time.perf_counter()
    
    image = cv2.imread(img_path)
    if image is None:
        raise ValueError("Failed to load image.")

    # ---------------------------------------------------------
    # TODO 2: NumPy Pre-processing
    # 1. Resize image to 224x224 (Use INTER_LINEAR or INTER_NEAREST).
    # 2. Convert BGR to RGB using zero-copy slicing [:, :, ::-1].
    # 3. Add a batch dimension using np.expand_dims so shape becomes (1, 224, 224, 3).
    # 
    # [CRITICAL EDGE RULE]: Because the model is INT8 Quantized, 
    # DO NOT cast to float32. Keep the data type as uint8!
    # ---------------------------------------------------------
    # input_data = ...
    
    input_data = np.zeros((1, 224, 224, 3), dtype=np.uint8) # Placeholder
    
    t_pre = (time.perf_counter() - t0) * 1000

    # =========================================================
    # PHASE 2: INFERENCE
    # =========================================================
    t1 = time.perf_counter()
    
    # ---------------------------------------------------------
    # TODO 3: Execute Inference
    # 1. Set the tensor value (set_tensor) using the input index.
    # 2. Invoke the engine (invoke()).
    # 3. Retrieve the output tensor (get_tensor) using the output index.
    # ---------------------------------------------------------
    # interpreter.set_tensor(input_details[0]['index'], input_data)
    # ...
    # output_data = ...
    
    output_data = np.zeros((1, 1001)) # Placeholder
    
    t_inf = (time.perf_counter() - t1) * 1000
    
    # =========================================================
    # PHASE 3: POST-PROCESSING
    # =========================================================
    # The model outputs a 1D array of 1001 probabilities.
    # Because it is INT8, the values are integers from 0 to 255.
    
    predictions = np.squeeze(output_data)
    top_1_index = np.argmax(predictions)
    confidence = predictions[top_1_index]
    
    print("\n[+] Inference Results:")
    print(f"    - Prediction: {labels[top_1_index]}")
    print(f"    - Quantized Confidence: {confidence} / 255") 
    
    print("\n[+] Latency Breakdown (Virtual Machine):")
    print(f"    - Pre-processing Time: {t_pre:.2f} ms")
    print(f"    - Inference Time:      {t_inf:.2f} ms")
    print(f"    - Total Pipeline Time: {(t_pre + t_inf):.2f} ms\n")
