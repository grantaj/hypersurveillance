"""
Face recognition from images.

This is based on the face_recognition_from_webcam_faster.py example
in  https://github.com/ageitgey/face_recognition
"""

import face_recognition
import numpy as np
import cv2


def find_faces(frame,
               known_face_encodings, known_face_names,
               image_scale_down=1):
    """Find known faces in an image.

    Given an image and an array of face encodings,
    find the locations of known faces (from the face encodings)
    in the image
    """
    # Resize frame of video to 1/4 size for faster face recognition processing
    small_frame = cv2.resize(frame, (0, 0),
                             fx=image_scale_down, fy=image_scale_down)

    # Convert from BGR (OpenCV) to RGB (face_recognition)
    rgb_frame = small_frame[:, :, ::-1]

    code = cv2.COLOR_BGR2RGB
    rgb_frame = cv2.cvtColor(rgb_frame, code)

    # Find all the faces and face encodings in the current frame of video
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    face_names = []
    for face_encoding in face_encodings:
        # See if the face is a match for the known face(s)
        matches = face_recognition.compare_faces(known_face_encodings,
                                                 face_encoding)
        name = "Unknown"

        # Use the known face with the smallest
        # distance to the new face
        face_distances = face_recognition.face_distance(known_face_encodings,
                                                        face_encoding)
        best_match_index = np.argmin(face_distances)
        if matches[best_match_index]:
            name = known_face_names[best_match_index]

        face_names.append(name)

    return face_locations, face_names


# Set up face database
def create_face_database(files):
    """Generate face encodings from images.

    Given an array of image file names, load each image
    and compute the face encoding for the face in the image

    Returns the array of face encodings.
    Currently no check for existence of files or what to
    do if face encoding fails (e.g. there are not faces in image)
    """
    known_face_encodings = []

    for f in files:
        image = face_recognition.load_image_file(f)
        encoding = face_recognition.face_encodings(image)[0]
        known_face_encodings.append(encoding)

    return known_face_encodings
