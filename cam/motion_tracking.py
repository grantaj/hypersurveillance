import cv2
import time
import base64
import hashlib
import os
import datetime
import pytz
import requests
import math
import numpy as np
import pickle
import face as facerecog

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
    timestamp = datetime.datetime.utcnow()
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
    headers = {'Content-Type': 'application/xml'} 
 
    return requests.post(url, headers=headers, data=data)

def onvif_stop():
    xml_body = XML_BODY_STOP
    xml_body = xml_body.replace("insert_profile_token", CAMERA_PROFILE_TOKEN)

    data = onvif_create_xml(xml_body)
    
    url = f"http://{CAMERA_IP}:2020/onvif/service"
    headers = {'Content-Type': 'application/xml'} 
 
    return requests.post(url, headers=headers, data=data)

##########################################################################################
# face detection

class Face:
    """ class for holding related data of a found face"""
    def __init__(self) -> None:
        self.x : int = 0
        self.y : int = 0
        self.w : int = 0
        self.h : int = 0
        self.face_centre_x : float = 0
        self.face_centre_y : float = 0
        self.normalised_x : float = 0
        self.normalised_y : float = 0
        pass


def face_detection(frame:np.ndarray, face_cascade:cv2.CascadeClassifier , face:Face) -> int:
    """ detect a face on an opencv frame, using a provided CascadeClassifer 
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
        
        # normalise for centre of face
        face.face_centre_x = (face.x + (face.w/2)) / frame.shape[1]  # Divide by width
        face.face_centre_y = (face.y + (face.h/2)) / frame.shape[0]  # Divide by height

        # normalise for camera coords
        face.normalised_x = (face.face_centre_x - 0.5) * 2
        face.normalised_y = -(face.face_centre_y - 0.5) * 2

    return len(faces)


def opencv_debug_overlay(frame:np.ndarray, face_count:int, face:Face):
    """overlay debug information on opencv frame

    Args:
        frame (np.ndarray): opencv frame
        face_count (int): number of faces
        face (Face): face object
    """
    # Display amount of faces found
    face_message = 'Faces found ' + str(face_count)
    cv2.putText(frame, face_message, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)   

    if face.x == 0 and face.y == 0 and face.w == 0 and face.h == 0 or face_count == 0:
        pass
    else:        
        # Draw rectangles around the detected faces
        cv2.rectangle(frame, (face.x, face.y), (face.x+face.w, face.y+face.h), (255, 0, 0), 2)
        
        # display x,y,w,h
        coor_str = f"x: {str(round(face.normalised_x,2))} y: {str(round(face.normalised_y,2))}"
        cv2.putText(frame, coor_str, (face.x, face.y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)




def main(mode):

    if mode == MODE_CAMERA:
        # Open the webcam (usually the default camera, 0)
        cap = cv2.VideoCapture(0)

    elif mode == MODE_STREAM:
        # Open an RTSP stream
        cap = cv2.VideoCapture(RTSP_URL)


    if not cap.isOpened():
        print("Error: Could not open cv2 stream")
        exit()

    # Load the pre-trained face cascade
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

    # Load the known face encodings
    (known_face_encodings, known_face_names) = pickle.load(open("faces.pkl", "rb"))
    image_scale_down = 1

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

    # Limit frame rates (adjust as needed)
    face_detection_fps = 6
    face_detection_fps_prev = 0.0
    send_command_fps = 1
    send_command_fps_prev = 0.0

    # vector length between centre of face and centre of screen
    delta_length = 0.0

    # theshold of face movement before sending move command (adjust as needed)
    delta_length_threshold = 0.2

    face_count = 0
    face = Face()

    is_camera_moving = False

    while True:
        # check timers
        face_detection_time_elapsed = time.time() - face_detection_fps_prev
        send_command_time_elapsed = time.time() - send_command_fps_prev

        # Read a frame from the webcam
        ret, frame = cap.read()
        

        ##########
        # FACE DETECTION
        if face_detection_time_elapsed > 1.0/face_detection_fps:
            
            # reset face detection timer
            face_detection_fps_prev = time.time()

            face_count = face_detection(frame, face_cascade, face)
            
            # calculate vector length between centre of face and centre of screen
            delta_length = math.sqrt(face.normalised_x**2 + face.normalised_y**2)

            face_locations, face_names = facerecog.find_faces(frame,
                                                              known_face_encodings,
                                                              known_face_names,
                                                              image_scale_down)
        ##########
        # STOP COMMAND
        if delta_length < delta_length_threshold:
            
            # issue stop command
            if is_camera_moving:
                print('stop command')
                onvif_stop()
                is_camera_moving = False

        ##########
        # MOVE COMMAND
        if send_command_time_elapsed> 1.0/send_command_fps:
            
            # reset send command timer
            send_command_fps_prev = time.time()

            if delta_length > delta_length_threshold:
                is_camera_moving = True

                # issue move command with speed proportional to delta_length
                print(f'move command {round(face.normalised_x * delta_length,2)} {round(face.normalised_y * delta_length,2)}')
                onvif_continuous_move(face.normalised_x * delta_length, face.normalised_y * delta_length, 0)
            
        
        opencv_debug_overlay(frame, face_count, face)

        for (top, right, bottom, left), name in zip(face_locations, face_names):
            # Draw a box around the face
            cv2.rectangle(frame, (left, top), (right, bottom), box_colour, 2)

            # Draw a label with a name below the face
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom),
                          box_colour, cv2.FILLED)
            cv2.putText(frame, name, (left + 6, bottom - 6), font, 1,
                        text_colour, 1)

        # Display the frame
        cv2.imshow('Webcam', frame)  
        
        # Check key presses
        key = cv2.waitKey(1)

        # Quit application
        if key == ord('q'):
            break

    # Release the webcam and close the window
    cap.release()
    cv2.destroyAllWindows()

main(MODE_CAMERA)
    
# HELPFUL OSC CODE FOR FUTURE
# Set up OSC client
# osc_client = udp_client.SimpleUDPClient("192.168.1.102", 12345)  # Replace with your actual destination IP and port

# Send OSC message with face detection data
# osc_client.send_message("/face_detection", (float(normalized_x), float(normalized_y), float(normalized_w), float(normalized_h)))
    
