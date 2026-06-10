from ultralytics import YOLO

def main():
    model = YOLO("yolo11n.pt")

    model.train(
        data="data.yaml",
        epochs=50,
        imgsz=640,
        batch=8,
        workers=0
    )

if __name__ == "__main__":
    main()