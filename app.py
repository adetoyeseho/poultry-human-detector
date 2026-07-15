import os
os.environ["YOLO_VERBOSE"] = "False"

import gc
import gradio as gr
import cv2
from ultralytics import YOLO

model = YOLO("best.onnx")

def predict_image(img):
    if img is None:
        return None, "No image provided"

    # Resize large images before inference to cap memory usage.
    # Model was trained at 640px anyway, so anything much bigger
    # just wastes memory without improving accuracy.
    h, w = img.shape[:2]
    max_dim = 1280
    if max(h, w) > max_dim:
        scale = max_dim / max(h, w)
        img = cv2.resize(img, (int(w * scale), int(h * scale)))

    results = model.predict(img, conf=0.5, imgsz=640, verbose=False)
    annotated = results[0].plot()
    annotated = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
    labels = [model.names[int(c)] for c in results[0].boxes.cls]
    summary = ", ".join(labels) if labels else "No person detected"

    # Free memory explicitly so it doesn't creep up across requests
    del results
    gc.collect()

    return annotated, summary

with gr.Blocks(title="Poultry Farm Human Detector") as demo:
    gr.Markdown(
        "# Poultry Farm Human Activity Detector\n"
        "Upload a photo to detect human presence and activity."
    )
    with gr.Tab("Photo"):
        img_input = gr.Image(type="numpy", label="Upload photo")
        img_button = gr.Button("Detect")
        img_output = gr.Image(label="Result")
        img_summary = gr.Textbox(label="Detected")
        img_button.click(
            predict_image,
            inputs=img_input,
            outputs=[img_output, img_summary],
        )

    # Video tab temporarily disabled — video processing loads full clips
    # frame-by-frame and is much more memory-hungry than a single image,
    # which reliably exceeds Render's free-tier 512MB limit. Re-enable
    # once photo detection is confirmed stable, or once hosted on a
    # tier with more memory.
    #
    # with gr.Tab("Video"):
    #     vid_input = gr.Video(label="Upload video")
    #     vid_button = gr.Button("Detect")
    #     vid_output = gr.Video(label="Result")
    #     vid_summary = gr.Textbox(label="Detected activities")
    #     vid_button.click(predict_video, inputs=vid_input, outputs=[vid_output, vid_summary])

demo.launch(server_name="0.0.0.0", server_port=7860)
