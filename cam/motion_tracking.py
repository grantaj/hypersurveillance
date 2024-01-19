import cv2
import time
from datetime import datetime, timedelta
import base64
import hashlib
import os
import pytz
import requests
import math
import numpy as np
import sys
import pickle
import face as facerecog
from fresh import FreshestFrame
import threading
import queue
import tkinter as tk
from tkinter import simpledialog
import os

# from pythonosc import osc_message_builder
# from pythonosc import udp_client

##########################################################################################
# network and onvif config (adjust as needed)
CAMERA_IP = "192.168.0.100"
CAMERA_USERNAME = "tapocam"
CAMERA_PASSWORD = "password"
CAMERA_PROFILE_TOKEN = "profile_1"
RTSP_URL = f"rtsp://tapocam:password@{CAMERA_IP}:554/stream2"
XML_TEMPLATE = """
<?xml version='1.0' encoding='utf-8'?>
<soap-env:Envelope xmlns:soap-env="http://www.w3.org/2003/05/soap-envelope">
    <soap-env:Header>
        <wsse:Security xmlns:wsse="http://docs.oasis-open.org/wss/2004/01/oasis-200401-wss-wssecurity-secext-1.0.xsd">
            <wsse:UsernameToken>
                <wsse:Username>insert_username</wsse:Username>
                <wsse:Password Type="http://docs.oasis-open.org/wss/2004/01/oasis-200401-wss-username-token-profile-1.0#PasswordDigest">insert_password_digest</wsse:Password>
                <wsse:Nonce EncodingType="http://docs.oasis-open.org/wss/2004/01/oasis-200401-wss-soap-message-security-1.0#Base64Binary">insert_nonce</wsse:Nonce>
                <wsu:Created xmlns:wsu="http://docs.oasis-open.org/wss/2004/01/oasis-200401-wss-wssecurity-utility-1.0.xsd">insert_timestamp</wsu:Created>
            </wsse:UsernameToken>
        </wsse:Security>
    </soap-env:Header>
    <soap-env:Body>insert_body</soap-env:Body>
</soap-env:Envelope>
"""

XML_BODY_ABSOLUTEMOVE = """
<ns0:AbsoluteMove xmlns:ns0="http://www.onvif.org/ver20/ptz/wsdl">
    <ns0:ProfileToken>profile_1</ns0:ProfileToken>
    <ns0:Position>
        <ns1:PanTilt xmlns:ns1="http://www.onvif.org/ver10/schema" x="insert_x" y="insert_y"/>
        <ns2:Zoom xmlns:ns2="http://www.onvif.org/ver10/schema" x="insert_z"/>
    </ns0:Position>
</ns0:AbsoluteMove>
"""

XML_BODY_CONTINUOUSMOVE = """
<ContinuousMove xmlns="http://www.onvif.org/ver20/ptz/wsdl">
    <ProfileToken>insert_profile_token</ProfileToken>
    <Velocity>
        <PanTilt x="insert_x" y="insert_y" xmlns="http://www.onvif.org/ver10/schema"/>
        <Zoom x="insert_z" xmlns="http://www.onvif.org/ver10/schema"/>
    </Velocity>    
</ContinuousMove>
"""
# <Timeout>insert_timeout</Timeout>

XML_BODY_STOP = """
<ns0:Stop xmlns:ns0="http://www.onvif.org/ver20/ptz/wsdl">
    <ns0:ProfileToken>insert_profile_token</ns0:ProfileToken>
</ns0:Stop>
"""


MODE_CAMERA = 0
MODE_STREAM = 1


##########################################################################################
# ONVIF functions
def onvif_create_timestamp():
    timestamp = datetime.utcnow()
    timestamp = timestamp.replace(tzinfo=pytz.utc, microsecond=0)
    return timestamp.isoformat()


def onvif_create_nonce():
    return os.urandom(16)


def onvif_get_password_digest(nonce, timestamp, password):
    timestamp_utf8 = timestamp.encode("utf-8")
    password_utf8 = password.encode("utf-8")
    hash = hashlib.sha1(nonce + timestamp_utf8 + password_utf8).digest()
    digest = base64.b64encode(hash).decode("ascii")
    return digest


def onvif_create_xml(xml_body):
    nonce = onvif_create_nonce()
    timestamp = onvif_create_timestamp()
    digest = onvif_get_password_digest(nonce, timestamp, CAMERA_PASSWORD)
    xml = XML_TEMPLATE
    xml = xml.replace("insert_username", CAMERA_USERNAME)
    xml = xml.replace("insert_password_digest", digest)
    xml = xml.replace("insert_nonce", base64.b64encode(nonce).decode("utf-8"))
    xml = xml.replace("insert_timestamp", timestamp)
    xml = xml.replace("insert_body", xml_body)
    return xml


def onvif_continuous_move(x, y, z):
    xml_body = XML_BODY_CONTINUOUSMOVE
    xml_body = xml_body.replace("insert_x", str(x))
    xml_body = xml_body.replace("insert_y", str(y))
    xml_body = xml_body.replace("insert_z", str(z))
    # xml_body = xml_body.replace("insert_timeout", f"PT{str(timeout)}S")
    xml_body = xml_body.replace("insert_profile_token", CAMERA_PROFILE_TOKEN)

    data = onvif_create_xml(xml_body)

    url = f"http://{CAMERA_IP}:2020/onvif/service"
    headers = {"Content-Type": "application/xml"}

    return requests.post(url, headers=headers, data=data)


def onvif_absolute_move(x, y, z):
    xml_body = XML_BODY_ABSOLUTEMOVE
    xml_body = xml_body.replace("insert_x", str(x))
    xml_body = xml_body.replace("insert_y", str(y))
    xml_body = xml_body.replace("insert_z", str(z))
    xml_body = xml_body.replace("insert_profile_token", CAMERA_PROFILE_TOKEN)

    data = onvif_create_xml(xml_body)

    url = f"http://{CAMERA_IP}:2020/onvif/service"
    headers = {"Content-Type": "application/xml"}

    return requests.post(url, headers=headers, data=data)


def onvif_stop():
    xml_body = XML_BODY_STOP
    xml_body = xml_body.replace("insert_profile_token", CAMERA_PROFILE_TOKEN)

    data = onvif_create_xml(xml_body)

    url = f"http://{CAMERA_IP}:2020/onvif/service"
    headers = {"Content-Type": "application/xml"}

    return requests.post(url, headers=headers, data=data)


##########################################################################################
# face detection


class Face:
    """Found face object.

    The face is stored as a bounding box
    The face is timestamped
    """

    def __init__(self) -> None:
        self.x: int = 0
        self.y: int = 0
        self.w: int = 0
        self.h: int = 0
        self.face_centre_x: float = 0
        self.face_centre_y: float = 0
        self.normalised_x: float = 0
        self.normalised_y: float = 0
        self.normalised_delta: float = 0
        self.timeStamp = None
        pass

    def normalise(self, frame):
        self.face_centre_x = (self.x + (self.w / 2)) / frame.shape[1]
        self.face_centre_y = (self.y + (self.h / 2)) / frame.shape[0]
        self.normalised_x = (self.face_centre_x - 0.5) * 2
        self.normalised_y = -(self.face_centre_y - 0.5) * 2
        self.normalised_delta = math.sqrt(
            self.normalised_x ** 2 + self.normalised_y ** 2
        )

    def set_from_trbl(self, top, right, bottom, left):
        self.x = left
        self.y = bottom
        self.w = right - left
        self.h = top - bottom
        self.timeStamp = datetime.now()

    def get_elapsed_time(self):
        if self.timeStamp is None:
            return timedelta.max
        return datetime.now() - self.timeStamp


def face_detection(
    frame: np.ndarray, face_cascade: cv2.CascadeClassifier, face: Face
) -> int:
    """detect a face on an opencv frame, using a provided CascadeClassifer
        and store in custom Face class

    Args:
        frame (np.ndarray): opencv frame
        face_cascade (cv2.CascadeClassifier): _description_
        face (Face): face object with values updated

    Returns:
        int: number of faces found
    """
    # Convert the frame to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Perform face detection
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    # just get first face found and update face instance
    if len(faces) != 0:
        face.x = faces[0][0]
        face.y = faces[0][1]
        face.w = faces[0][2]
        face.h = faces[0][3]

        face.normalise(frame)

    return len(faces)


def opencv_debug_overlay(frame: np.ndarray, face_count: int, face: Face):
    """overlay debug information on opencv frame

    Args:
        frame (np.ndarray): opencv frame
        face_count (int): number of faces
        face (Face): face object
    """
    # Display amount of faces found

    face_message = "Faces found " + str(face_count)
    cv2.putText(
        frame, face_message, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2
    )

    if face.x == 0 and face.y == 0 and face.w == 0 and face.h == 0 or face_count == 0:
        pass
    else:
        # Draw rectangles around the detected faces
        cv2.rectangle(
            frame, (face.x, face.y), (face.x + face.w, face.y + face.h), (255, 0, 0), 2
        )

        # display x,y,w,h
        coor_str = (
            f"x: {str(round(face.normalised_x,2))} y: {str(round(face.normalised_y,2))}"
        )
        cv2.putText(
            frame,
            coor_str,
            (face.x, face.y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 255),
            2,
        )


def face_overlay(frame, face_locations, face_names, target):
    if face_locations is None or face_names is None:
        return

    text_colour = (255, 255, 255)
    font = cv2.FONT_HERSHEY_DUPLEX

    for (top, right, bottom, left), name in zip(face_locations, face_names):
        if name == target:
            box_colour = (0, 0, 255)
        else:
            box_colour = (0, 255, 0)

        # Draw a box around the face
        cv2.rectangle(frame, (left, top), (right, bottom), box_colour, 2)

        # Draw a label with a name below the face
        cv2.rectangle(
            frame, (left, bottom - 35), (right, bottom), box_colour, cv2.FILLED
        )
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 1, text_colour, 1)


def face_recognition(
    freshcap,
    shared_face_encodings,
    shared_face_names,
    lock,
    image_scale_down,
    sleep_duration_seconds,
    result_queue,
):
    while True:
        ret, frame = freshcap.read()

        with lock:
            known_face_encodings = shared_face_encodings[0]
            known_face_names = shared_face_names[1]

        face_locations, face_names = facerecog.find_faces(
            frame, known_face_encodings, known_face_names, image_scale_down
        )

        if not result_queue.empty():
            # Discard the old result
            try:
                result_queue.get_nowait()
            except queue.Empty:
                pass

        result_queue.put((face_locations, face_names))

        time.sleep(sleep_duration_seconds)


def update_parameter(new_value, param, lock):
    with lock:
        # Safely update the shared parameter
        param[0] = new_value


def get_latest_result(result_queue):
    try:
        return result_queue.get_nowait()
    except queue.Empty:
        return None, None



def main(mode):
    webcam_index = 0

    if mode == MODE_CAMERA:
        # Open a webcam
        filename = webcam_index
    elif mode == MODE_STREAM:
        # Open an RTSP stream
        filename = RTSP_URL
        onvif_absolute_move(0, 0, 0)
    else:
        sys.exit("Unknown mode.")

    # cap = cv2.VideoCapture(RTSP_URL)
    cap = cv2.VideoCapture(filename)
    if not cap.isOpened():
        print("Error: Could not open cv2 stream")
        exit()

    freshcap = FreshestFrame(cap)

    # Load the pre-trained face cascade
    # face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

    # Load the known face encodings
    (known_face_encodings, known_face_names) = pickle.load(open("faces.pkl", "rb"))
    image_scale_down = 1
    target = "Alex"

    # Limit frame rates (adjust as needed)
    face_detection_fps = 5
    send_command_fps = 10
    send_command_fps_prev = 0.0

    # Don't use detections older than this
    maximum_detection_age_seconds = 1

    # theshold of face movement before sending move command (adjust as needed)
    normalised_delta_threshold = 0.2

    face_count = 0
    face = Face()

    is_camera_moving_x = False
    is_camera_moving_y = False

    result_queue = queue.Queue(maxsize=1)
    param_lock = threading.Lock()
    shared_face_encodings = [known_face_encodings]
    shared_face_names = [known_face_names]
    thread = threading.Thread(
        target=face_recognition,
        args=(
            freshcap,
            param_lock,
            shared_face_encodings,
            shared_face_names,
            image_scale_down,
            1.0 / face_detection_fps,
            result_queue,
        ),
    )
    thread.daemon = True  # Thread will close when main program exits
    thread.start()

    count = 0
    total_time = 0

    face_locations = []
    face_names = []

    while True:
        # check timers
        send_command_time_elapsed = time.time() - send_command_fps_prev

        # Read a frame
        ret, frame = freshcap.read()

        ##########
        # FACE DETECTION
        new_face_locations, new_face_names = get_latest_result(result_queue)

        if new_face_locations is not None:
            face_locations = new_face_locations
            face_names = new_face_names

        try:
            face_index = face_names.index(target)
            (top, right, bottom, left) = face_locations[face_index]
            face.set_from_trbl(top, right, bottom, left)
            face.normalise(frame)
            targetFound = True

        except ValueError:
            targetFound = False

        face_overlay(frame, face_locations, face_names, target)

        if mode == MODE_STREAM:
            ##########
            # STOP COMMAND
            if face.get_elapsed_time() > timedelta(
                seconds=maximum_detection_age_seconds
            ) and (is_camera_moving_x or is_camera_moving_y):
                onvif_stop()
                is_camera_moving_x = False
                is_camera_moving_y = False
                print("stopping movement (target lost)")

            if (
                is_camera_moving_x
                and face.normalised_x < normalised_delta_threshold / 2
            ):
                onvif_stop()
                is_camera_moving_x = False
                is_camera_moving_y = False
                print("stopping x movement")

            if (
                is_camera_moving_y
                and face.normalised_y < normalised_delta_threshold / 2
            ):
                onvif_stop()
                is_camera_moving_y = False
                is_camera_moving_x = False
                print("stopping y movement")

            ##########
            # MOVE COMMAND
            if send_command_time_elapsed > 1.0 / send_command_fps:

                # reset send command timer
                send_command_fps_prev = time.time()

                if (
                    face.normalised_delta > normalised_delta_threshold
                    and face.get_elapsed_time()
                    < timedelta(seconds=maximum_detection_age_seconds)
                ):

                    if math.fabs(face.normalised_x) > math.fabs(face.normalised_y):
                        # Pan
                        onvif_continuous_move(face.normalised_x / 10, 0, 0)
                        is_camera_moving_x = True
                        is_camera_moving_y = False
                        print("x movement")
                    else:
                        # Tilt
                        onvif_continuous_move(0, face.normalised_y / 10, 0)
                        is_camera_moving_y = True
                        is_camera_moving_x = False
                        print("y movement")

        # opencv_debug_overlay(frame, face_count, face)

        # Display the frame
        cv2.imshow("Webcam", frame)

        # Check key presses
        key = cv2.waitKey(1)

        # Quit application
        if key == ord("q"):

            if mode == MODE_STREAM:
                onvif_stop()
                onvif_absolute_move(0, 0, 0)
            break
        
        # Add image to face recognition
        elif key == ord('f'):
            image_filename = f"new_frame.png"
            cv2.imwrite(image_filename, frame)
            root = tk.Tk()
            root.withdraw()
            user_input = simpledialog.askstring("Input", "Enter something:")
            if user_input is None:
                user_input = 'missing_name'
            filename = user_input + ".png"
            print(filename)
            os.rename("new_frame.png", filename)

            image = face_recognition.load_image_file(filename)
            encoding = face_recognition.face_encodings(image)[0]
            known_face_encodings.append(encoding)

            pickle.dump((known_face_encodings, known_face_names),
                open("faces.pkl", "wb"))


    # Release the webcam and close the window
    freshcap.release()
    cv2.destroyAllWindows()


# MODE_CAMERA || MODE_STREAM
main(MODE_CAMERA)

# HELPFUL OSC CODE FOR FUTURE
# Set up OSC client
# osc_client = udp_client.SimpleUDPClient("192.168.1.102", 12345)  # Replace with your actual destination IP and port

# Send OSC message with face detection data
# osc_client.send_message("/face_detection", (float(normalized_x), float(normalized_y), float(normalized_w), float(normalized_h)))
