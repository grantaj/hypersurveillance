import cv2
import pickle
import face

# Initialize variables
process_this_frame = 0
frame_rate = 4
image_scale_down = 1
image_scale_up = 1

generate_database = False

if generate_database is True:
    files = [
        "obama.jpg",
        "biden.jpg",
        "alex.jpg",
        "fraz.jpg"
    ]

    known_face_names = [
        "Obama",
        "Biden",
        "Alex",
        "Fraz"
    ]

    known_face_encodings = face.create_face_database(files)

    pickle.dump((known_face_encodings, known_face_names),
                open("faces.pkl", "wb"))

else:

    print("loading face database")
    (known_face_encodings, known_face_names) = pickle.load(open("faces.pkl",
                                                                "rb"))

# Get a reference to webcam #0 (the default one)
video_capture = cv2.VideoCapture(0)

while True:
    # Grab a single frame of video
    ret, frame = video_capture.read()

    # Only process every other frame of video to save time
    if (process_this_frame == 0):
        face_locations, face_names = face.find_faces(frame,
                                                     known_face_encodings,
                                                     known_face_names,
                                                     image_scale_down)

    process_this_frame = (process_this_frame+1) % frame_rate

    # Display the results
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        # Scale back up face locations
        top *= image_scale_up
        right *= image_scale_up
        bottom *= image_scale_up
        left *= image_scale_up

        # Draw a box around the face
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

        # Draw a label with a name below the face
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom),
                      (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0,
                    (255, 255, 255), 1)

    # Display the resulting image
    cv2.imshow('Video', frame)

    # Hit 'q' on the keyboard to quit!
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release handle to the webcam
video_capture.release()
cv2.destroyAllWindows()
