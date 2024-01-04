# modules

onvif-zeep 0.2.12
zeep 4.0.0

# links

- onvif operations: https://www.onvif.org/onvif/ver20/util/operationIndex.html
- onvif specs, wsdl, xsd: https://www.onvif.org/profiles/specifications/

# definitions

- ONVIF: (Open Network Video Interface Forum)
- WSDL: (Web Services Description Language)
- XSD: (Xml Schema Definition)
- SOAP: (Simple Object Access Protocol)
- RTSP: (Real-Time Streaming Protocol)

# generate available operations

REQUIREMENTS

- need zeep
- need all necessary wsdl files with correct relative paths

`https://docs.python-zeep.org/en/master/in_depth.html`

```
python -m zeep file.wsdl
```

# Endpoints

1. device management details
   `http://device-ip-address:port/onvif/device_service`

2. all other services
   `http://device-ip-address:port/onvif/service`
