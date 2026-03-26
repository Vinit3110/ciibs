from ultralytics import YOLO
import cv2

# Load trained model
model = YOLO("runs/detect/train3/weights/best.pt")

# Load image
image_path = "bus.jpg"  # change later
img = cv2.imread(image_path)

# Run inference
results = model(img)

# Draw results
annotated = results[0].plot()

# Save output
cv2.imwrite("output.jpg", annotated)

print("Saved output as output.jpg")