from ultralytics import YOLO
import cv2
import csv

# Load trained model
model = YOLO(r"runs/detect/train-4/weights/best.pt")

video_path = "worker_safety.mp4"

cap = cv2.VideoCapture(video_path)

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# Save processed video
out = cv2.VideoWriter(
    "output.mp4",
    cv2.VideoWriter_fourcc(*'mp4v'),
    fps,
    (width, height)
)

# Violation counters
violations = {
    "NO-Hardhat": 0,
    "NO-Mask": 0,
    "NO-Safety Vest": 0
}

frame_number = 0

while True:

    ret, frame = cap.read()

    if not ret:
        break

    frame_number += 1

    results = model(frame, conf=0.5)

    annotated_frame = results[0].plot()

    current_violations = 0

    for box in results[0].boxes:

        cls = int(box.cls[0])
        class_name = model.names[cls]

        if class_name in violations:
            violations[class_name] += 1
            current_violations += 1

    cv2.putText(
        annotated_frame,
        f"Violations: {current_violations}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 0, 255),
        2
    )

    cv2.imshow("Construction Safety Monitor", annotated_frame)

    out.write(annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()

# Save report
with open("violation_report.csv", "w", newline="") as file:

    writer = csv.writer(file)

    writer.writerow(["Violation Type", "Count"])

    for violation, count in violations.items():
        writer.writerow([violation, count])

print("\nViolation Summary")

for violation, count in violations.items():
    print(f"{violation}: {count}")

print("\nProcessed video saved as output.mp4")
print("Report saved as violation_report.csv")