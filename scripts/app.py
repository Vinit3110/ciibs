from ultralytics import YOLO
import gradio as gr
import cv2

# Load model
model = YOLO("runs/detect/train3/weights/best.pt")

THREAT_CLASSES = ["Gun", "Knife", "Bullet"]

def detect(image):
    # 🔥 Convert to grayscale (handles domain shift)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image_gray3 = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    # Run model on grayscale version
    results = model(image_gray3, conf=0.1)

    annotated = results[0].plot()

    boxes = results[0].boxes
    names = model.names

    detected = []
    threat_found = False

    if boxes is not None and boxes.cls is not None:
        for cls, conf in zip(boxes.cls, boxes.conf):
            label = names[int(cls)]
            confidence = float(conf)

            detected.append(f"{label} ({confidence:.2f})")

            if label in THREAT_CLASSES:
                threat_found = True

    if len(detected) == 0:
        status = " No objects detected 🟢"
    elif threat_found:
        status = " THREAT DETECTED 🚨"
    else:
        status = " No critical threat ✅"

    return annotated, status, "\n".join(detected)


# 👇 THIS is the UI part (replace your old one with this)
app = gr.Interface(
    fn=detect,
    inputs=gr.Image(type="numpy"),
    outputs=[
        gr.Image(type="numpy"),
        gr.Textbox(label="Status"),
        gr.Textbox(label="Detected Objects")
    ],
    title="AI-Powered X-ray Threat Detection",
    description="Detect prohibited items like weapons in cargo X-ray images"
)

app.launch()