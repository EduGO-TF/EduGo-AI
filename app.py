from flask import Flask, request, jsonify
import numpy as np
import cv2
import onnxruntime as ort
import io
from PIL import Image

app = Flask(__name__)

# Load ONNX model
onnx_session = ort.InferenceSession("best.onnx", providers=["CPUExecutionProvider"])
input_name = onnx_session.get_inputs()[0].name

# 이미지 전처리
def preprocess(img):
    img_resized = cv2.resize(img, (640, 640))
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
    img_norm = img_rgb.astype(np.float32) / 255.0
    img_input = np.transpose(img_norm, (2, 0, 1))  # HWC -> CHW
    img_input = np.expand_dims(img_input, axis=0)  # Add batch dim
    return img_input

# 후처리: 응답 포맷에 맞게
def postprocess(output, conf_threshold=0.25):
    detections = output[0]
    results = []

    for det in detections:
        try:
            # confidence 값 추출
            confidence = det[4]
            if isinstance(confidence, (np.ndarray, list)):
                confidence = float(confidence[0])
            else:
                confidence = float(confidence)

            # class label 추출
            label = det[5]
            if isinstance(label, (np.ndarray, list)):
                label = int(label[0])
            else:
                label = int(label)

            if confidence > conf_threshold:
                results.append({
                    "x": label,
                    "y": round(confidence, 2),
                    "cookie_image": ""
                })
        except Exception as e:
            print(f"예외 발생 (무시): {e}")
            continue

    return results

@app.route('/detect', methods=['POST'])
def detect():
    if 'image' not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files['image']
    image_bytes = file.read()
    img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    img_np = np.array(img)
    img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

    input_tensor = preprocess(img_bgr)
    outputs = onnx_session.run(None, {input_name: input_tensor})
    detections = postprocess(outputs)

    return jsonify({"detections": detections})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000, debug=True)
