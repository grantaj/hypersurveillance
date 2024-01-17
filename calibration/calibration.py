"""
Perform calibration.

Generate translation, rotation and camera matrix
for each of the cameras.
Capture images using getImages.py
"""

import numpy as np
import cv2
import re
import os
import pickle


def compute_camera_world_position(rotation_vector, translation_vector):
    """Compute world coordinates a camera.

    This function converts a rotation vector into a rotation matrix
    and combines it with the translation vector to form the camera's
    extrinsic matrix. It then inverts this matrix to transform from
    camera coordinates to world coordinates, thus computing the
    camera's position in the world.

    Parameters:
    rotation_vector (numpy.ndarray): A 3x1 vector
    representing the rotation of the camera in axis-angle format. This
    is typically obtained from camera calibration (e.g.,
    cv2.solvePnP).  translation_vector (numpy.ndarray): A 3x1 vector
    representing the translation of the camera. This is also typically
    obtained from camera calibration.

    Returns:
    numpy.ndarray: A 3-element vector representing the
    camera's position in world coordinates.
    """
    #
    # Convert the rotation vector to a rotation matrix
    rotation_matrix, _ = cv2.Rodrigues(rotation_vector)

    # Invert the rotation matrix (R is orthogonal, so inv is transpose)
    rotation_matrix_inv = rotation_matrix.T

    # Apply the inverted rotation to the translation vector
    camera_position_world = -rotation_matrix_inv @ translation_vector

    return camera_position_world.ravel()


def read_images_from_directory(directory):
    images = {}
    pattern = re.compile(r'cam_(\d+)_frame_(\d+)\.png')

    for filename in sorted(os.listdir(directory)):
        if filename.endswith('.png'):
            match = pattern.match(filename)
            if match:
                cam_index, frame_index = map(int, match.groups())
                if cam_index not in images:
                    images[cam_index] = {}
                image_path = os.path.join(directory, filename)
                images[cam_index][frame_index] = cv2.imread(image_path)

    return images


# System parameters
# Calibration image (chessboard)
# For a m x n chessboard the parameters here need to me (m-1)x(n-1)
chessboardSize = (8, 6)
size_of_chessboard_squares_mm = 40

# image frame size in pixels (currently assuming all images the same size)
frameSize = (640, 480)

# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# These are the coordinates of the chessboard corners in mm relative to
# an origin at the top left of the chessboard
objp = np.zeros((chessboardSize[0] * chessboardSize[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:chessboardSize[0], 0:chessboardSize[1]].T.reshape(-1, 2)
objp = objp * size_of_chessboard_squares_mm


images = read_images_from_directory('images')

for cam_index in images:
    goodFrames = []
    camName = f"camera_{cam_index}"

    # Arrays to store object points and image points from all the images.
    objpoints = []  # 3d point in real world space
    imgpoints = []  # 2d points in image plane.

    for frame_index in images[cam_index]:
        img = images[cam_index][frame_index]
        frameName = f"camera {cam_index} frame {frame_index}"
        print('Processing ' + frameName)

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Find the chess board corners
        ret, corners = cv2.findChessboardCorners(gray, chessboardSize, None)

        # If found, add object points, image points (after refining them)
        if ret is True:

            goodFrames.append(frameName)

            objpoints.append(objp)
            corners2 = cv2.cornerSubPix(gray, corners,
                                        (11, 11), (-1, -1),
                                        criteria)
            imgpoints.append(corners2)

            # Draw and display the corners
            cv2.drawChessboardCorners(img, chessboardSize, corners2, ret)
            cv2.imshow(frameName, img)
            cv2.waitKey(100)

        else:

            print(frameName, "skipped")

    cv2.destroyAllWindows()

    # Calibration -------------------------------------------------------------

    ret, cameraMatrix, dist, rvecs, tvecs, newObjPoints = cv2.calibrateCameraRO(
        objpoints, imgpoints, frameSize, int(chessboardSize[0]-1), None, None)

    # Save the camera calibration result for later use. -----------------------

    pickle.dump((rvecs, tvecs), open(camName + "_transformation.pkl", "wb"))
    pickle.dump((cameraMatrix, dist), open(camName + "_calibration.pkl", "wb"))
    pickle.dump(cameraMatrix, open(camName + "_cameraMatrix.pkl", "wb"))
    pickle.dump(dist, open(camName + "_dist.pkl", "wb"))

    for R, t, frameName in zip(rvecs, tvecs, goodFrames):
        camera_position_world = compute_camera_world_position(R, t)
        d = np.linalg.norm(camera_position_world)
        print(frameName + " - Distance from camera: %5.0f mm" % d)
