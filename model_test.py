from ultralytics import YOLO

model = YOLO("runs/detect/train-4/weights/best.pt")

results = model("test_image.png", save=True)
# Loop through results
for r in results:
    # Get class IDs
    class_ids = r.boxes.cls.cpu().numpy()
    
    # Get class names
    class_names = [model.names[int(c)] for c in class_ids]
    
    # Count occurrences of each class
    from collections import Counter
    counts = Counter(class_names)
    
    print("Detected classes:", class_names)
    print("Counts:", counts)
