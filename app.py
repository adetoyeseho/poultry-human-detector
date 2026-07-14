import gradio as gr
import cv2
from ultralytics import YOLO

model = YOLO("best.pt")

def predict_image(img):
    results = model.predict(img, conf=0.5, verbose=False)
    annotated = results[0].plot()
    annotated = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
    labels = [model.names[int(c)] for c in results[0].boxes.cls]
    summary = ", ".join(labels) if labels else "No person detected"
    return annotated, summary

def predict_video(video_path):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out_path = "output_video.mp4"
    out = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    detected_labels = set()
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        results = model.predict(frame, conf=0.5, verbose=False)
        annotated = results[0].plot()
        out.write(annotated)
        for c in results[0].boxes.cls:
            detected_labels.add(model.names[int(c)])
    cap.release()
    out.release()
    summary = ", ".join(detected_labels) if detected_labels else "No person detected"
    return out_path, summary

with gr.Blocks(title="Poultry Farm Human Detector") as demo:
    gr.Markdown("# Poultry Farm Human Activity Detector\nUpload a photo or video to detect human presence and activity.")
    with gr.Tab("Photo"):
        img_input = gr.Image(type="numpy", label="Upload photo")
        img_button = gr.Button("Detect")
        img_output = gr.Image(label="Result")
        img_summary = gr.Textbox(label="Detected")
        img_button.click(predict_image, inputs=img_input, outputs=[img_output, img_summary])
    with gr.Tab("Video"):
        vid_input = gr.Video(label="Upload video")
        vid_button = gr.Button("Detect")
        vid_output = gr.Video(label="Result")
        vid_summary = gr.Textbox(label="Detected activities")
        vid_button.click(predict_video, inputs=vid_input, outputs=[vid_output, vid_summary])

demo.launch(server_name="0.0.0.0", server_port=7860)
