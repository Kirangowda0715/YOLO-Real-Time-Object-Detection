import cv2
import numpy as np
import time

# Load Yolo (Tiny YOLO for fast real-time webcam inference)
net = cv2.dnn.readNet("weights/yolov3-tiny.weights", "cfg/yolov3-tiny.cfg")
output_layers = net.getUnconnectedOutLayersNames()

classes = []
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

colors = np.random.uniform(0, 255, size=(len(classes), 3))

# Connect to webcam (DirectShow backend on Windows avoids MSMF grab error -1072875772)
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
if not cap.isOpened() or not cap.read()[0]:
    cap.release()
    cap = cv2.VideoCapture(0)  # fallback to default

if not cap.isOpened():
    print("Error: Could not open webcam. Make sure your camera is connected and not in use by another app.")
    exit()

font = cv2.FONT_HERSHEY_PLAIN
starting_time = time.time()
frame_id = 0

print("Press 'ESC' in the video window to stop.")

while True:
    ret, frame = cap.read()
    if not ret or frame is None:
        print("Failed to grab frame.")
        break

    frame_id += 1
    height, width, channels = frame.shape

    # Detecting objects
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    class_ids = []
    confidences = []
    boxes = []

    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.3:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.4, 0.3)

    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            confidence = confidences[i]
            color = colors[class_ids[i]]
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, f"{label} {round(confidence, 2)}", (x, y + 25), font, 1.5, color, 2)

    elapsed_time = time.time() - starting_time
    fps = frame_id / elapsed_time if elapsed_time > 0 else 0
    cv2.putText(frame, f"FPS: {round(fps, 1)}", (10, 40), font, 2, (0, 255, 0), 2)

    cv2.imshow("Real-Time YOLO Webcam Detection", frame)
    key = cv2.waitKey(1)
    if key == 27:  # ESC key
        break

cap.release()
cv2.destroyAllWindows()
