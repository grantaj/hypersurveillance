"""
Example of face recognition from webcam.

Press q to quit

This is based on the face_recognition_from_webcam_faster.py example
in  https://github.com/ageitgey/face_recognition
"""
import cv2
import pickle
import face

# Initialize variables --------------------------------------------------------
# index zero is probably the built in webcam
# or replace with a url to a video stream
webcam_index = 0

# Only process frame numbers which are 0 modulo the frame_rate
# e.g. if frame_rate = 4, will process every fourth frame
frame_rate = 4

# For faster recognition, scale the images down by this factor
image_scale_down = .5

# If True, build the known face encodings from the array of images
generate_database = False

# Image filenames and associated people names for building the
# known face encodings database.
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

# Style options
colourmap = {
    "red": (0, 0, 255),
    "green": (0, 255, 0),
    "blue": (255, 0, 0),
    "yellow": (0, 255, 255),
    "white": (255, 255, 255),
    "black": (0, 0, 0),
}

box_colour = colourmap["red"]
text_colour = colourmap["white"]
font = cv2.FONT_HERSHEY_DUPLEX

# ------------------------------------------------------------------------------
# Either build or load the known face encodings

if generate_database is True:

    print("Generating face database")

    known_face_encodings = face.create_face_database(files)

    pickle.dump((known_face_encodings, known_face_names),
                open("faces.pkl", "wb"))

else:

    print("Loading face database")
    (known_face_encodings, known_face_names) = pickle.load(open("faces.pkl",
                                                                "rb"))

# ------------------------------------------------------------------------------
# Main loop

# Get a reference to the video
video_capture = cv2.VideoCapture(webcam_index)
process_this_frame = 0

while True:
    # Grab a single frame of video
    ret, frame = video_capture.read()

    if (process_this_frame == 0):
        face_locations, face_names = face.find_faces(frame,
                                                     known_face_encodings,
                                                     known_face_names,
                                                     image_scale_down)

    process_this_frame = (process_this_frame+1) % frame_rate

    # Display the results
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        # Draw a box around the face
        cv2.rectangle(frame, (left, top), (right, bottom), box_colour, 2)

        # Draw a label with a name below the face
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom),
                      box_colour, cv2.FILLED)
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 1,
                    text_colour, 1)

    # Display the resulting image
    cv2.imshow('Video', frame)

    # Hit 'q' on the keyboard to quit!
    if cv2.waitKey(1) == ord('q'):
        break

# Release handle to the webcam
video_capture.release()
cv2.destroyAllWindows()
