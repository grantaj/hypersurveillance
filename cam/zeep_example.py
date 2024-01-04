from zeep import Client
from zeep.wsse.username import UsernameToken
from zeep.plugins import HistoryPlugin
import logging.config

# web security
wsse = UsernameToken('tapocam', 'password', use_digest=True)

# enable logging
history = HistoryPlugin()

# define log config and location
logging.config.dictConfig({
    'version': 1,
    'formatters': {
        'verbose': {
            'format': '%(asctime)s: %(name)s: %(message)s'
        }
    },
    'handlers': {
        'console': {
            'level': 'DEBUG',
            'class': 'logging.StreamHandler',
            'formatter': 'verbose',
        },
        'file': {
            'level': 'DEBUG',
            'class': 'logging.FileHandler',
            'filename': 'zeep_example.log',  # Specify the path to your log file
            'formatter': 'verbose',
        },
    },
    'loggers': {
        'zeep.transports': {
            'level': 'DEBUG',
            'propagate': True,
            'handlers': ['console', 'file'],
        },
    }
})

wsdl_dir = './wsdl_fraz'

# DEVICE MANAGEMENT
# 1. Create devicemgmt 'client' object
wsdl_devicemgmt = wsdl_dir + '/devicemgmt.wsdl' # paths in wsdl file need to be correct
client_devicemgmt = Client(wsdl=wsdl_devicemgmt, wsse=wsse, plugins=[history])

# 2. Connect to service, using the client
binding_name = '{http://www.onvif.org/ver10/device/wsdl}DeviceBinding'
address = 'http://192.168.1.126:2020/onvif/device_service'
service_devicemgmt = client_devicemgmt.create_service(binding_name=binding_name, address=address)

# INTERESTING 'devicemgmt' SERVICE OPERATIONS
# value = service_devicemgmt.GetDeviceInformation()
# value = service_devicemgmt.GetDiscoveryMode()
# value = service_devicemgmt.GetHostname()
# value = service_devicemgmt.GetDNS()
# value = service_devicemgmt.GetNetworkDefaultGateway()
# value = service_devicemgmt.GetNetworkInterfaces()
# value = service_devicemgmt.GetNetworkProtocols()
# value = service_devicemgmt.GetScopes()

#  ns0:GetServices(IncludeCapability: xsd:boolean)
# value = service_devicemgmt.GetServices(IncludeCapability=True) # to pass values

# value = service_devicemgmt.GetSystemDateAndTime()
# value = service_devicemgmt.GetUsers()
# print(value)
##################################################################

# MEDIA
# 1. Create client
wsdl_media = wsdl_dir +'/media.wsdl'
client_media = Client(wsdl=wsdl_media, wsse=wsse)

# 2. Connect to a 'ptz' service, using the client
binding_name = '{http://www.onvif.org/ver10/media/wsdl}MediaBinding'
address = 'http://192.168.1.126:2020/onvif/service'
service_media = client_media.create_service(binding_name=binding_name, address=address)

# value = service_media.GetProfiles()
# print(value)


##################################################################

# PTZ
# 1. Create Client
wsdl_ptz = wsdl_dir + '/ptz.wsdl' # paths in file need to be correct
client_ptz = Client(wsdl=wsdl_ptz, wsse=wsse)

# 2. Connect to service, using the client
binding_name = '{http://www.onvif.org/ver20/ptz/wsdl}PTZBinding'
address = 'http://192.168.1.126:2020/onvif/service'
service_ptz = client_ptz.create_service(binding_name=binding_name, address=address)

# request.ProfileToken = ReferenceToken # need to get a media profile token from media service
# request.Position = {'PanTilt': {'x': pan, 'y': tilt}, 'Zoom': zoom}
# request.Position = position

# res = service_ptz.AbsoluteMove(request)
# print(res)

# value = service_ptz.GetConfigurations()
# value = service_ptz.GetNodes()
# value = service_ptz.GetServiceCapabilities()
# value = service_ptz.GotoHomePositionResponse()
# print(value)


##################################################################

# ANALYTICS
# 1. Create client
wsdl_media = wsdl_dir + '/analytics.wsdl'
client_analytics = Client(wsdl=wsdl_media, wsse=wsse)

# 2. Connect to service, using the client
binding_name = '{http://www.onvif.org/ver20/analytics/wsdl}AnalyticsEngineBinding'
address = 'http://192.168.1.126:2020/onvif/service'
service_analytics = client_analytics.create_service(binding_name=binding_name, address=address)

# value = service_analytics.GetServiceCapabilities()

# print(value)