import json
import os
from tqdm import tqdm

# =========================
# PATHS
# =========================
json_path = r"D:\codes\ciibs\data\pidray\annotations\train\train.json"
images_dir = r"D:\codes\ciibs\data\pidray\images\train"
labels_dir = r"D:\codes\ciibs\data\pidray\labels\train"

os.makedirs(labels_dir, exist_ok=True)

# =========================
# LOAD DATA
# =========================
with open(json_path, 'r') as f:
    data = json.load(f)

images = {img["id"]: img for img in data["images"]}
categories = {cat["id"]: idx for idx, cat in enumerate(data["categories"])}

# Group annotations per image
ann_map = {}
for ann in data["annotations"]:
    img_id = ann["image_id"]
    ann_map.setdefault(img_id, []).append(ann)

# =========================
# HELPER FUNCTION
# =========================
def clip(val):
    return max(0, min(val, 1))

# =========================
# CONVERSION
# =========================
skipped = 0
total = 0

for img_id, img in tqdm(images.items()):
    file_name = img["file_name"]
    width = img["width"]
    height = img["height"]

    label_path = os.path.join(labels_dir, file_name.replace(".png", ".txt"))

    with open(label_path, "w") as f:
        for ann in ann_map.get(img_id, []):
            total += 1

            bbox = ann["bbox"]  # [x, y, w, h]
            x, y, w, h = bbox

            # Skip invalid raw boxes
            if w <= 0 or h <= 0:
                skipped += 1
                continue

            # Convert to YOLO format (normalized)
            x_center = clip((x + w / 2) / width)
            y_center = clip((y + h / 2) / height)
            w = clip(w / width)
            h = clip(h / height)

            # Skip nearly zero boxes
            if w < 1e-6 or h < 1e-6:
                skipped += 1
                continue

            class_id = categories[ann["category_id"]]

            f.write(f"{class_id} {x_center} {y_center} {w} {h}\n")

# =========================
# SUMMARY
# =========================
print("\n Conversion complete!")
print(f"Total annotations processed: {total}")
print(f"Skipped invalid boxes: {skipped}")
print(f"Labels saved to: {labels_dir}")