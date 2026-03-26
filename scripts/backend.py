from fastapi import FastAPI, UploadFile
from ultralytics import YOLO
import cv2
import numpy as np

app = FastAPI()

from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # allow all (for dev)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model = YOLO("runs/detect/train3/weights/best.pt")

THREAT_CLASSES = ["gun", "knife", "blade", "bullet", "baton"]

@app.post("/detect")
async def detect(file: UploadFile):
    image_bytes = await file.read()
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if img is None:
        return {"error": "Invalid image"}

    results = model(img, conf=0.1)[0]

    detections = []

    for box in results.boxes:
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        conf = float(box.conf[0])

        if conf < 0.3:
            continue

        cls = int(box.cls[0])
        label = model.names[cls].lower()

        detections.append({
            "label": label,
            "confidence": round(conf, 3),
            "bbox": [
                int(x1),
                int(y1),
                int(x2 - x1),
                int(y2 - y1)
            ]
        })

    # Risk scoring
    risk_score = 0
    for d in detections:
        if d["label"] in THREAT_CLASSES:
            risk_score += 40 * d["confidence"]
        else:
            risk_score += 15 * d["confidence"]

    risk_score = min(int(risk_score), 100)

    return {
        "detections": detections,
        "risk_score": risk_score
    }