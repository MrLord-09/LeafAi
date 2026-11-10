import cv2
import numpy as np


def preprocess_leaf_image(image_bytes: bytes, size=(224, 224)) -> tuple[np.ndarray, np.ndarray]:
    """General plant-image preprocessing. Does not assume the subject is a leaf."""
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Corrupted image or unsupported file encoding.")

    # Keep the full image by default. A light center crop avoids destroying fruit/flower context.
    h, w = img.shape[:2]
    if min(h, w) >= 320:
        side = int(min(h, w) * 0.92)
        y0 = max((h - side) // 2, 0)
        x0 = max((w - side) // 2, 0)
        candidate = img[y0:y0 + side, x0:x0 + side]
    else:
        candidate = img

    resized = cv2.resize(candidate, size, interpolation=cv2.INTER_AREA)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    normalized = rgb.astype(np.float32) / 255.0
    tensor = np.expand_dims(normalized, axis=0)
    return tensor, candidate
