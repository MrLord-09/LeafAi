import json
import os
from pathlib import Path
import numpy as np

try:
    import tflite_runtime.interpreter as tflite
except ImportError:
    try:
        import tensorflow.lite as tflite
    except ImportError:
        tflite = None


class PlantModel:
    def __init__(self, model_path: str, labels_path: str | None = None):
        if tflite is None:
            raise RuntimeError("TensorFlow/TFLite is not installed. Run: pip install -r requirements.txt")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at: {model_path}")
        self.interpreter = tflite.Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        self.input_shape = self.input_details[0]['shape']
        self.input_dtype = self.input_details[0]['dtype']
        self.output_dtype = self.output_details[0]['dtype']
        self.output_quant = self.output_details[0].get('quantization', (0.0, 0))
        self.labels = self._load_labels(labels_path)

    def _load_labels(self, labels_path):
        if labels_path and Path(labels_path).exists():
            with open(labels_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            if isinstance(data, list):
                return data
            if isinstance(data, dict):
                return [data[str(i)] for i in range(len(data))]
        return []

    def predict(self, input_tensor: np.ndarray) -> tuple[int, float]:
        tensor = input_tensor.astype(self.input_dtype, copy=False)
        if self.input_dtype == np.uint8:
            scale, zero = self.input_details[0].get('quantization', (1.0, 0))
            if scale:
                tensor = np.clip(tensor / scale + zero, 0, 255).astype(np.uint8)

        self.interpreter.set_tensor(self.input_details[0]['index'], tensor)
        self.interpreter.invoke()
        output_data = self.interpreter.get_tensor(self.output_details[0]['index'])[0]

        if self.output_dtype == np.uint8:
            scale, zero = self.output_quant
            if scale:
                output_data = (output_data.astype(np.float32) - zero) * scale

        # Softmax if the exported model returns logits.
        values = output_data.astype(np.float32)
        if np.any(values < 0) or not np.isclose(values.sum(), 1.0, atol=0.05):
            values = np.exp(values - np.max(values))
            values = values / np.sum(values)

        class_idx = int(np.argmax(values))
        confidence = float(values[class_idx]) * 100.0
        return class_idx, round(confidence, 2)
