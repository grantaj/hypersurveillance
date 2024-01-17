import cv2
import pickle
import numpy as np


def camera2_in_camera1_coordinates(rvec1, tvec1, rvec2, tvec2):
    # Convert rotation vectors to rotation matrices
    R1, _ = cv2.Rodrigues(rvec1)
    R2, _ = cv2.Rodrigues(rvec2)

    # Invert the transformation for camera 1
    R1_inv = np.linalg.inv(R1)
    tvec1_inv = -R1_inv @ tvec1

    # Transformation from world to camera 2
    T2 = R2 @ np.hstack((np.eye(3), tvec2.reshape(3, 1)))

    # Combine the transformations
    combined_transformation = T2 @ np.hstack((R1_inv, tvec1_inv.reshape(3, 1)))

    # Extract the translation component
    position_of_camera2_in_camera1 = combined_transformation[:3, 3]

    return position_of_camera2_in_camera1



rvec1, tvec1 = pickle.load(open("camera_1_transformation.pkl", 'rb'))
rvec2, tvec2 = pickle.load(open("camera_2_transformation.pkl", 'rb'))

R1, _ = cv2.Rodrigues(rvec1[0])
R2, _ = cv2.Rodrigues(rvec2[0])

t1 = tvec1[0]
t2 = tvec2[0]

relative_position = R1.T @ t1 - R2.T @ t2


print("Camera 2 position relative to camera 1: ")
print(relative_position)


