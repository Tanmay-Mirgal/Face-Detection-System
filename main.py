import face_recognition
import cv2
import numpy as np
import csv
from datetime import datetime

# Load images and encodings
jobs_image = face_recognition.load_image_file("jobs.jpg")
jobs_encoding = face_recognition.face_encodings(jobs_image)[0]

ratan_tata_image = face_recognition.load_image_file("tata.jpg")
ratan_tata_encoding = face_recognition.face_encodings(ratan_tata_image)[0]

tesla_image = face_recognition.load_image_file("tesla.jpg")
tesla_encoding = face_recognition.face_encodings(tesla_image)[0]

aaryan_image = face_recognition.load_image_file("aaryan.jpg")
aaryan_encoding = face_recognition.face_encodings(aaryan_image)[0]

known_face_encodings = [
    jobs_encoding,
    ratan_tata_encoding,
    tesla_encoding,
    aaryan_encoding
]

known_face_names = [
    "jobs",
    "ratan_tata",
    "tesla",
    "aaryan"
]

students_present = set()

# Open CSV file for attendance
current_date = datetime.now().strftime("%Y-%m-%d")
csv_file = f"{current_date}_attendance.csv"
with open(csv_file, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Name', 'Time'])

# Initialize video capture
video_capture = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frame = video_capture.read()

    # Resize frame for faster processing
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

    # Find all face locations and encodings in the current frame
    face_locations = face_recognition.face_locations(rgb_small_frame)
    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

    # Loop through each face found in the frame
    for face_encoding, face_location in zip(face_encodings, face_locations):
        # Compare the face encoding with known face encodings
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        name = "Unknown"

        # Check if the face matches any known faces
        if True in matches:
            first_match_index = matches.index(True)
            name = known_face_names[first_match_index]

            # Check if the face has already been registered for attendance
            if name not in students_present:
                students_present.add(name)

                # Write attendance to CSV
                with open(csv_file, mode='a', newline='') as file:
                    writer = csv.writer(file)
                    writer.writerow([name, datetime.now().strftime("%H:%M:%S")])

        # Draw rectangle and label around the face
        top, right, bottom, left = [i * 4 for i in face_location]  # Scale back to original size
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.putText(frame, name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    # Display the resulting frame
    cv2.imshow('Video', frame)

    # Check for exit condition
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release video capture and close all windows
video_capture.release()
cv2.destroyAllWindows()

# Print the list of students present
print("Students Present:")
for student in students_present:
    print(student)
