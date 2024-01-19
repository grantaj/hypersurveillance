import os
import sys
import time
import threading
import numpy as np
import cv2 as cv

# also acts (partly) like a cv.VideoCapture
class FreshestFrame(threading.Thread):
    """Capture the freshest frame from a video source. This class behaves
    partly like cv.VideoCapture but with enhanced capabilities for
    real-time video processing. It uses threading to constantly read
    frames from the given capture device and makes the latest frame
    available.

    Attributes:
        capture (cv.VideoCapture): The video capture source.
        cond (threading.Condition): A condition variable to manage access to the frame.
        running (bool): Flag to control the running of the thread.
        frame (np.ndarray): The latest captured frame.
        latestnum (int): Sequence number of the latest captured frame.
        callback (function): An optional callback function executed with each new frame.

    Methods:
        start(): Starts the thread.
        release(timeout=None): Stops the thread and releases the capture source.
        run(): Continuously captures frames from the source in a separate thread.
        read(wait=True, seqnumber=None, timeout=None): Returns the latest frame.

    """

    def __init__(self, capture, name="FreshestFrame"):
        """
        Initializes the FreshestFrame thread.

        Args:
            capture (cv.VideoCapture): The video capture source to read from.
            name (str, optional): The name of the thread. Defaults to "FreshestFrame".

        Raises:
            AssertionError: If the capture source is not opened successfully.
        """
        self.capture = capture
        assert self.capture.isOpened()

        # this lets the read() method block until there's a new frame
        self.cond = threading.Condition()

        # this allows us to stop the thread gracefully
        self.running = False

        # keeping the newest frame around
        self.frame = None

        # passing a sequence number allows read() to NOT block
        # if the currently available one is exactly the one you ask for
        self.latestnum = 0

        # this is just for demo purposes
        self.callback = None

        super().__init__(name=name)
        self.start()

    def start(self):
        """
        Starts the FreshestFrame thread. Sets the running flag to True and calls
        the start method of the superclass (threading.Thread).
        """
        self.running = True
        super().start()

    def release(self, timeout=None):
        """
        Stops the FreshestFrame thread and releases the video capture resource.

        Args:
            timeout (float, optional): Time in seconds to wait for the thread to join.
        """
        self.running = False
        self.join(timeout=timeout)
        self.capture.release()

    def run(self):
        """
        The main logic of the thread. Continuously reads frames from the capture source
        and updates the latest frame and its sequence number. Also calls the callback
        function with the new frame if provided.
        """
        counter = 0
        while self.running:
            # block for fresh frame
            (rv, img) = self.capture.read()
            assert rv
            counter += 1

            # publish the frame
            with self.cond:  # lock the condition for this operation
                self.frame = img if rv else None
                self.latestnum = counter
                self.cond.notify_all()

            if self.callback:
                self.callback(img)

    def read(self, wait=True, seqnumber=None, timeout=None):
        """
        Reads and returns the latest frame captured by the thread.

        Args:
            wait (bool, optional): If True, blocks until a new frame is available.
                                    If False, returns the current frame immediately.
                                    Defaults to True.
            seqnumber (int, optional): If provided, blocks until the frame with at least
                                       this sequence number is available.
            timeout (float, optional): Maximum time to block waiting for a new frame.

        Returns:
            tuple: A tuple containing the sequence number of the returned frame and
                   the frame itself (np.ndarray). If no new frame is available within
                   the timeout, returns the latest available frame.
        """
        # with no arguments (wait=True), it always blocks for a fresh frame
        # with wait=False it returns the current frame immediately (polling)
        # with a seqnumber, it blocks until that frame is available (or no wait at all)
        # with timeout argument, may return an earlier frame;
        #   may even be (0,None) if nothing received yet

        with self.cond:
            if wait:
                if seqnumber is None:
                    seqnumber = self.latestnum + 1
                if seqnumber < 1:
                    seqnumber = 1

                rv = self.cond.wait_for(
                    lambda: self.latestnum >= seqnumber, timeout=timeout
                )
                if not rv:
                    return (self.latestnum, self.frame)

            return (self.latestnum, self.frame)
