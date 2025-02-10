import cv2
import numpy as np
import os
import csv
import time
import pickle
from sklearn.neighbors import KNeighborsClassifier
from datetime import datetime

# Load the dataset
with open("data/names.pkl", "rb") as w:
    LABELS = pickle.load(w)

with open("data/face_data.pkl", "rb") as f:
    FACES = pickle.load(f)

# Convert to NumPy arrays
FACES = np.array(FACES)
LABELS = np.array(LABELS)

# Ensure FACES and LABELS have the same length
if len(FACES) != len(LABELS):
    print(f"Data mismatch: FACES({len(FACES)}), LABELS({len(LABELS)})")
    min_len = min(len(FACES), len(LABELS))
    FACES = FACES[:min_len]
    LABELS = LABELS[:min_len]
    print(f"Trimmed data: FACES({len(FACES)}), LABELS({len(LABELS)})")

# Train the KNN classifier
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(FACES, LABELS)

# Start video capture
video = cv2.VideoCapture(0)
facedetect = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

COL_NAMES = ['NAME', 'TIME']

while True:
    ret, frame = video.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = facedetect.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in faces:
        crop_img = frame[y:y+h, x:x+w, :]
        resized_img = cv2.resize(crop_img, (50, 50)).flatten().reshape(1, -1)

        # Predict using KNN
        output = knn.predict(resized_img)[0]

        ts = time.time()
        date = datetime.fromtimestamp(ts).strftime("%d-%m-%Y")
        timestamp = datetime.fromtimestamp(ts).strftime("%H:%M-%S")

        # Ensure attendance folder exists
        if not os.path.exists("Attendance"):
            os.makedirs("Attendance")

        filename = f"Attendance/Attendance_{date}.csv"
        exist = os.path.isfile(filename)

        # Draw rectangle and label
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
        cv2.putText(frame, str(output), (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 
                    0.8, (255, 255, 255), 2, cv2.LINE_AA)

        attendance = [str(output), str(timestamp)]

    cv2.imshow("Face Recognition", frame)
    k = cv2.waitKey(1)

    if k == ord('o'):  # Take attendance when 'o' is pressed
        time.sleep(2)

        with open(filename, "a", newline='') as csvfile:
            writer = csv.writer(csvfile)
            if not exist:
                writer.writerow(COL_NAMES)
            writer.writerow(attendance)

    if k == ord("q"):  # Quit when 'q' is pressed
        break

# Release resources
video.release()
cv2.destroyAllWindows()
