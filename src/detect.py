from ultralytics import YOLO
import cv2
from pathlib import Path

def detect_image(image_path, model_path="best.pt", conf=0.25):
    model = YOLO(model_path)
    image = cv2.imread(image_path)
    results = model(image, conf=conf)
    annotated = results[0].plot()
    output_path = Path(image_path).stem + "_detected.jpg"
    cv2.imwrite(output_path, annotated)
    for box in results[0].boxes:
        class_name = model.names[int(box.cls)]
        confidence = float(box.conf)
        print(f"{class_name}: {confidence:.2%}")
    return output_path

if __name__ == "__main__":
    import sys
    detect_image(sys.argv[1])
