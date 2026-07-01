from ultralytics import YOLO

def main():
    model = YOLO("yolov8s.pt")
    model.train(
        data="data.yaml",
        epochs=30,
        imgsz=640,
        batch=16,
        project="runs",
        name="weapon_detection_v1",
        workers=2,
        device=0
    )

if __name__ == "__main__":
    main()
