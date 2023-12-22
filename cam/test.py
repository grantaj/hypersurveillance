import logging
import time
from onvif import ONVIFCamera
from ptz import CameraControl 

ip = "192.168.1.126"
port = "2020"
user = "tapocam"
password = "password"

mycam = ONVIFCamera(ip, port, user, password, '/home/alex/python-onvif-zeep/wsdl/')
# resp = mycam.devicemgmt.GetHostname()
# print(resp)

# dt = mycam.devicemgmt.GetSystemDateAndTime()
# print(dt)

ptz_service = mycam.create_ptz_service()
media_service = mycam.create_media_service()
media_profile = media_service.GetProfiles()[0]


request = ptz_service.create_type('GetConfigurations')
#request.ProfileToken = media_profile.token
config = ptz_service.GetConfigurations(request)[0]
print(config)

request = ptz_service.create_type('GetConfiguration')
request.PTZConfigurationToken = config.token
resp = ptz_service.GetConfiguration(request)
print(resp)

quit()

ptz_cam = CameraControl(ip, port, user, password)
ptz_cam.camera_start()

print("Absolute Move 1")
resp = ptz_cam.absolute_move(0, 0, 1)   ## ( Pan, Tilt, Zoom )
print(resp)

time.sleep(10)

print("Continuous Move")
resp = ptz_cam.continuous_move(.1, .1, 0)
print(resp)

time.sleep(1)

print("Stop")
resp = ptz_cam.stop_move()
print(resp)

#print("Relative Move");
#resp = ptz_cam.relative_move(-.25, -.25, 0)
#print(resp)



# print("Stop move")
# resp = ptz_cam.stop_move()
# print(resp)

# print("Absolute Move 2")
# resp = ptz_cam.absolute_move(.5, 0, 1)   ## ( Pan, Tilt, Zoom )
# print(resp)




logging.basicConfig(filename='teste-onvif.log', filemode='w', level=logging.DEBUG)
logging.info('Started')
