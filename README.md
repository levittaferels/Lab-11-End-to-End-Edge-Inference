# Lab 11: End-to-End Edge Inference (Quantized TFLite)

## 📌 Overview
You have optimized your pre-processing pipeline using pure NumPy (W9) and efficient Interpolation (W10). Now, it's time to connect it to an actual Edge AI Inference Engine. 

In this lab, you will **not** train or quantize a model. Instead, mirroring industry practice, you will download a pre-quantized **INT8 MobileNetV1** model from Google and build a highly efficient inference pipeline around it.

## 🎯 Learning Objectives
1. Understand how to load and allocate tensors in a TensorFlow Lite Interpreter.
2. Experience the strict data type constraints of **INT8 Quantization** (No `float32` allowed!).
3. Profile the end-to-end latency breakdown: Pre-processing vs. Neural Network Inference.

## 🛠️ Instructions
1. **Prepare Environment**: 
   Ensure you have the required packages installed in your GitHub VM:
   `pip install tflite-runtime numpy opencv-python urllib3`
   *(Note: If `tflite-runtime` fails on your specific OS, you can use `pip install tensorflow` as a fallback, but `tflite-runtime` is the Edge standard).*
2. **Execute Script**: 
   Run `python lab11_edge_inference.py`. It will automatically download the quantized MobileNet model and a standard test image (Grace Hopper).
3. **Complete the TODOs**:
   - **TODO 1**: Initialize the TFLite Interpreter and allocate memory.
   - **TODO 2**: Inject your W9/W10 NumPy pre-processing logic (Resize, BGR to RGB, Expand Dims). **Data must remain as `uint8`**.
   - **TODO 3**: Set the input tensor, invoke the engine, and retrieve the output tensor.

## ✅ Expected Output
Your terminal should successfully predict "military uniform" and print a clear latency breakdown (e.g., Pre-processing: X ms, Inference: Y ms).
