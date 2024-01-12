"""
Capture image frames.

Capture the specified number of frames from the webcam.
and store in specifed
directory
"""

import cv2
import numpy as np
import math
from time import time
from datetime import datetime


def available_webcams(max_index=10):
    """
    Return a list of indices of available webcams.

    :param max_index: The maximum webcam index to test.
    :return: List of available webcam indices.
    """
    available_cams = []
    for index in range(max_index):
        cap = cv2.VideoCapture(index)
        if cap.isOpened():
            available_cams.append(index)
            cap.release()
    return available_cams


def generate_timestamped_filename(base_name, extension):
    """
    Generates a timestamped filename using the current date and time.

    Args:
    base_name (str): The base name of the file.
    extension (str): The extension of the file.

    Returns:
    str: A string representing the timestamped filename.
    """
    # Current date and time in a formatted string (YYYYMMDD_HHMMSS)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Construct the filename
    filename = f"{base_name}_{timestamp}.{extension}"

    return filename


def tile_images(images, grid_size=None):
    if not grid_size:
        grid_size = (math.ceil(math.sqrt(len(images))), ) * 2

    image_height, image_width = images[0].shape[:2]
    canvas_width = image_width * grid_size[1]
    canvas_height = image_height * grid_size[0]
    canvas = np.zeros((canvas_height, canvas_width, 3), dtype=np.uint8)

    for i, img in enumerate(images):
        row = i // grid_size[1]
        col = i % grid_size[1]
        canvas[row*image_height:(row+1)*image_height, col*image_width:(col+1)*image_width] = img

    return canvas


# System parameters
image_format = 'png'
total_frames = 5
period_seconds = 1

# Initialisation
num_frames = 0
t_seconds = time()

# Open all of the available webcams
cap_objects = []
cams = available_webcams()
for cam in cams:
    cap = cv2.VideoCapture(cam)
    if not cap.isOpened():
        print("Error: could not open webcam{cam}")
    else:
        cap_objects.append(cap)


# Main loop: capture frames from each of the webcams
num_frames = 0
while num_frames < total_frames:

    img_list = []
    # Get a frame from each webcam
    cam_index = 0
    for cap in cap_objects:
        cam_index += 1
        success, img = cap.read()
        img_list.append(img)
        image_name = 'Camera {cam_index}'

    tiled_image = tile_images(img_list)
    cv2.imshow('images', tiled_image)

    # Save the frames if the user presses 's' key, or at the time period
    key = cv2.waitKey(5)
    if key == ord('s') or time() > t_seconds + period_seconds:

        t_seconds = time()
        num_frames += 1
        cam_index = 0
        for img in img_list:
            cam_index += 1
            filename = f"images/cam_{cam_index}_frame_{num_frames}"
            filename = generate_timestamped_filename(filename,
                                                     image_format)
            cv2.imwrite(filename, img)
            print(filename)

# Clean up
for cap in cap_objects:
    cap.release()

cv2.destroyAllWindows()
