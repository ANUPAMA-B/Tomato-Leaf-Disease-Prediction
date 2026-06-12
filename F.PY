import cv2
import numpy as np

# Load pre-trained MobileNet SSD model
net = cv2.dnn.readNetFromCaffe("deploy.prototxt", "mobilenet_iter_73000.caffemodel")

# Define the class labels for MobileNet SSD
CLASSES = ["background", "airplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "dog", 
           "horse", "motorbike", "person", "pottedplant", "sheep", "train", "tvmonitor"]

# Initialize video capture (use 0 for webcam or provide a path for an image)
video = cv2.VideoCapture(0)

while True:
    ret, frame = video.read()  # Read the frame from the webcam or image
    if not ret:
        print("Failed to grab frame")
        break

    # Prepare the frame for MobileNet SSD (resize and normalize)
    blob = cv2.dnn.blobFromImage(frame, 0.007843, (300, 300), 127.5, 127.5, 127.5, crop=False)
    net.setInput(blob)
    detections = net.forward()

    # Loop through the detections
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.2:  # Filter out weak detections (confidence threshold)
            idx = int(detections[0, 0, i, 1])
            box = detections[0, 0, i, 3:7] * np.array([frame.shape[1], frame.shape[0], frame.shape[1], frame.shape[0]])
            (startX, startY, endX, endY) = box.astype("int")

            # Draw bounding box and label
            label = f"{CLASSES[idx]}: {confidence*100:.2f}%"
            cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)
            cv2.putText(frame, label, (startX, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Display the frame with detected objects
    cv2.imshow("Object Detection", frame)

    # Press 'q' to exit the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close all OpenCV windows
video.release()
cv2.destroyAllWindows()
