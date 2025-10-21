# stage1_object_detection.py

from ultralytics import YOLO
import cv2
import os
import matplotlib
matplotlib.use('Agg')  # non-GUI backend for servers
import matplotlib.pyplot as plt
import numpy as np


def run_object_detection(image_path, save_output=True):
    """
    Stage 1: Object Detection using YOLOv8
    ----------------------------------------------------
    Args:
        image_path (str): Path to input image.
        save_output (bool): If True, saves the annotated image.

    Returns:
        detections (list[dict]): Each dict contains:
            {
                'label': str,
                'bbox': [x1, y1, x2, y2],
                'confidence': float
            }
    """
    # Load model
    model = YOLO("yolov8s.pt")

    # Inference
    results = model(image_path, verbose=False)
    result = results[0]

    # Extract detections
    boxes = result.boxes.xyxy.cpu().numpy() if result.boxes else []
    confs = result.boxes.conf.cpu().numpy() if result.boxes else []
    classes = result.boxes.cls.cpu().numpy() if result.boxes else []

    detections = []
    for i in range(len(boxes)):
        label = model.names[int(classes[i])]
        detections.append({
            "label": label,
            "bbox": boxes[i].tolist(),
            "confidence": float(confs[i])
        })

    print(f"\n🟩 Detected {len(detections)} objects: {[d['label'] for d in detections]}\n")

    # Save annotated image (optional)
    if save_output:
        annotated_img = result.plot()
        output_path = os.path.join("data", "detected_output.jpg")
        os.makedirs("data", exist_ok=True)
        cv2.imwrite(output_path, annotated_img)
        print(f"✅ Annotated image saved to {output_path}")

    return detections


def visualize_detections(image_path, detections):
    """
    Helper: Visualize bounding boxes and labels for sanity check.
    """
    img = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    for det in detections:
        x1, y1, x2, y2 = map(int, det['bbox'])
        label = det['label']
        conf = det['confidence']
        cv2.rectangle(img_rgb, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(img_rgb, f"{label} {conf:.2f}", (x1, y1 - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

    plt.figure(figsize=(8, 6))
    plt.imshow(img_rgb)
    plt.axis("off")
    plt.title("YOLOv8 Object Detections")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    image_path = "data/sample_images/street.jpg"
    detections = run_object_detection(image_path)
    visualize_detections(image_path, detections)
