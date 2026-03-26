from fastapi import FastAPI, UploadFile
from ultralytics import YOLO
import cv2
import numpy as np
import os

from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model safely
MODEL_PATH = os.path.join(os.path.dirname(__file__), "best.pt")
model = YOLO(MODEL_PATH)

THREAT_CLASSES = ["gun", "knife", "blade", "bullet", "baton"]

@app.get("/")
def home():
    return {"message": "CargoScan AI Backend Running"}

@app.post("/detect")
async def detect(file: UploadFile):
    image_bytes = await file.read()
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if img is None:
        return {"error": "Invalid image"}

    try:
        results = model(img, conf=0.1)[0]
    except Exception as e:
        return {"error": str(e)}

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

    # Risk level
    if risk_score > 70:
        risk_level = "High"
    elif risk_score > 40:
        risk_level = "Medium"
    else:
        risk_level = "Low"

    return {
        "detections": detections,
        "risk_score": risk_score,
        "risk_level": risk_level
    }

import uvicorn

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=10000)