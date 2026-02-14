import face_recognition
import csv
import cv2
import numpy as np
from datetime import datetime


video_capture = cv2.VideoCapture(0)

Adarsh_image = face_recognition.load_image_file("faces/Adarsh.jpg")
Adarsh_encoding = face_recognition.face_encodings(Adarsh_image)[0]

Prince_image = face_recognition.load_image_file("faces/Prince.jpg")
Prince_encoding = face_recognition.face_encodings(Prince_image)[0]

known_face_encodings = [Adarsh_encoding, Prince_encoding]
known_face_names = ["Adarsh", "Prince"]

students = known_face_names.copy()

face_locations = []
face_encodings = []


now = datetime.now()
current_date = now.strftime("%Y-%m-%d")

f = open(f"{current_date}.csv", "w+", newline="")
lnwriter = csv.writer(f)

while True:
    ret, frame = video_capture.read()
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

    face_locations = face_recognition.face_locations(rgb_small_frame)
    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

    for face_encoding in face_encodings:
        name = "Unknown"
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        face_distance = face_recognition.face_distance(known_face_encodings, face_encoding)
        best_match_index = np.argmin(face_distance)

        if matches[best_match_index]:
            name = known_face_names[best_match_index]

        if name in known_face_names:
            font = cv2.FONT_HERSHEY_SIMPLEX
            bottomleftCornerofText = (10, 100)
            fontScale = 1.5
            fontColor = (255, 0, 0)
            thickness = 3
            linetype = 2
            cv2.putText(frame, name +  " Present", bottomleftCornerofText, font, fontScale, fontColor, thickness, linetype )


            if name in students:
                students.remove(name)
                current_time = now.strftime("%H-%M-%S")
                lnwriter.writerow([name, current_time])
            

    cv2.imshow("Attendance", frame)
    if cv2.waitKey(1) & 0xff == ord("q"):
        break  # yeh ab while loop ko break karega

video_capture.release()
cv2.destroyAllWindows()
f.close()

