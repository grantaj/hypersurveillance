import logging
from ptz import CameraControl 


ptz_cam = CameraControl("192.168.1.126", "2020", "tapocam", "password")
ptz_cam.camera_start()

print("Absolute Move")
resp = ptz_cam.absolute_move(0, 0, 1)   ## ( Pan, Tilt, Zoom )
print(resp)

#print("Relative Move");
#resp = ptz_cam.relative_move(-.25, -.25, 0)
#print(resp)



logging.basicConfig(filename='teste-onvif.log', filemode='w', level=logging.DEBUG)
logging.info('Started')
