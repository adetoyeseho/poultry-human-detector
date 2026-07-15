import os
import gc
import numpy as np
import cv2
import onnxruntime as ort
import gradio as gr

# ---- Config ----
MODEL_PATH = "best.onnx"
INPUT_SIZE = 640
CONF_THRESHOLD = 0.5
IOU_THRESHOLD = 0.45

CLASS_NAMES = [
    "Person breaking into a poultry farm",
    "Person carrying egg crate to store",
    "Person cleaning poultry farm",
    "Person feeding chickens",
    "Person picking eggs",
    "Person stealing eggs",
]

# Load the ONNX model once at startup (shared across requests)
session = ort.InferenceSession(MODEL_PATH, providers=["CPUExecutionProvider"])
input_name = session.get_inputs()[0].name


def letterbox(img, new_size=640):
    """Resize + pad image to a square while keeping aspect ratio."""
    h, w = img.shape[:2]
    scale = min(new_size / h, new_size / w)
    nh, nw = int(h * scale), int(w * scale)
    resized = cv2.resize(img, (nw, nh))

    canvas = np.full((new_size, new_size, 3), 114, dtype=np.uint8)
    top = (new_size - nh) // 2
    left = (new_size - nw) // 2
    canvas[top:top + nh, left:left + nw] = resized

    return canvas, scale, left, top


def preprocess(img):
    canvas, scale, pad_x, pad_y = letterbox(img, INPUT_SIZE)
    blob = canvas.astype(np.float32) / 255.0
    blob = blob.transpose(2, 0, 1)  # HWC -> CHW
    blob = np.expand_dims(blob, axis=0)
    return blob, scale, pad_x, pad_y


def postprocess(output, scale, pad_x, pad_y, orig_w, orig_h):
    # output shape: (1, 4+num_classes, num_anchors) -> transpose to (num_anchors, 4+num_classes)
    preds = output[0].transpose(1, 0)

    boxes = []
    scores = []
    class_ids = []

    for row in preds:
        cls_scores = row[4:]
        cls_id = int(np.argmax(cls_scores))
        conf = float(cls_scores[cls_id])
        if conf < CONF_THRESHOLD:
            continue

        cx, cy, w, h = row[:4]
        x1 = (cx - w / 2 - pad_x) / scale
        y1 = (cy - h / 2 - pad_y) / scale
        x2 = (cx + w / 2 - pad_x) / scale
        y2 = (cy + h / 2 - pad_y) / scale

        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(orig_w, x2), min(orig_h, y2)

        boxes.append([x1, y1, x2 - x1, y2 - y1])  # cv2.dnn.NMSBoxes wants [x, y, w, h]
        scores.append(conf)
        class_ids.append(cls_id)

    if not boxes:
        return []

    indices = cv2.dnn.NMSBoxes(boxes, scores, CONF_THRESHOLD, IOU_THRESHOLD)
    results = []
    for i in np.array(indices).flatten():
        x, y, w, h = boxes[i]
        results.append({
            "box": (int(x), int(y), int(x + w), int(y + h)),
            "conf": scores[i],
            "class_id": class_ids[i],
        })
    return results


def predict_image(img):
    if img is None:
        return None, "No image provided"

    orig_h, orig_w = img.shape[:2]

    # Cap resolution to limit memory use on large uploads
    max_dim = 1280
    if max(orig_h, orig_w) > max_dim:
        s = max_dim / max(orig_h, orig_w)
        img = cv2.resize(img, (int(orig_w * s), int(orig_h * s)))
        orig_h, orig_w = img.shape[:2]

    blob, scale, pad_x, pad_y = preprocess(img)
    output = session.run(None, {input_name: blob})[0]
    detections = postprocess(output, scale, pad_x, pad_y, orig_w, orig_h)

    annotated = img.copy()
    labels_found = []
    for det in detections:
        x1, y1, x2, y2 = det["box"]
        label = CLASS_NAMES[det["class_id"]]
        labels_found.append(label)
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.putText(annotated, f"{label} {det['conf']:.2f}", (x1, max(y1 - 10, 15)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
    summary = ", ".join(labels_found) if labels_found else "No person detected"

    del output, detections
    gc.collect()

    return annotated_rgb, summary


with gr.Blocks(title="Poultry Farm Human Detector") as demo:
    gr.Markdown(
        "# Poultry Farm Human Activity Detector\n"
        "Upload a photo to detect human presence and activity."
    )
    img_input = gr.Image(type="numpy", label="Upload photo")
    img_button = gr.Button("Detect")
    img_output = gr.Image(label="Result")
    img_summary = gr.Textbox(label="Detected")
    img_button.click(predict_image, inputs=img_input, outputs=[img_output, img_summary])

demo.launch(server_name="0.0.0.0", server_port=7860)
