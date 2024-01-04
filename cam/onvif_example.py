import base64
import hashlib
import os
import datetime
import pytz
import requests

xml = """
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
    <soap-env:Body>
        <ns0:AbsoluteMove xmlns:ns0="http://www.onvif.org/ver20/ptz/wsdl">
            <ns0:ProfileToken>profile_1</ns0:ProfileToken>
            <ns0:Position>
                <ns1:PanTilt xmlns:ns1="http://www.onvif.org/ver10/schema" x="insert_x" y="insert_y"/>
                <ns2:Zoom xmlns:ns2="http://www.onvif.org/ver10/schema" x="insert_z"/>
            </ns0:Position>
        </ns0:AbsoluteMove>
    </soap-env:Body>
</soap-env:Envelope>
"""

def create_timestamp():
    timestamp = datetime.datetime.utcnow()
    timestamp = timestamp.replace(tzinfo=pytz.utc, microsecond=0)
    return timestamp.isoformat()

def create_nonce():    
  return os.urandom(16)

def get_password_digest(nonce, timestamp, password):
  timestamp_utf8 = timestamp.encode("utf-8")
  password_utf8 = password.encode("utf-8")
  hash = hashlib.sha1(nonce + timestamp_utf8 + password_utf8).digest()
  digest = base64.b64encode(hash).decode("ascii")
  return digest
   
def create_xml(xml, username, password, x, y, z):
    nonce = create_nonce()
    timestamp = create_timestamp()
    digest = get_password_digest(nonce, timestamp, password)
    xml = xml.replace("insert_username", username)
    xml = xml.replace("insert_password_digest", digest)
    xml = xml.replace("insert_nonce", base64.b64encode(nonce).decode("utf-8"))
    xml = xml.replace("insert_timestamp", timestamp)
    xml = xml.replace("insert_x", str(x))
    xml = xml.replace("insert_y", str(y))
    xml = xml.replace("insert_z", str(z))
    return xml

def absolute_move(x, y, z):
    username = "tapocam"
    password = "password"
    data = create_xml(xml, username, password, x, y, z)
    
    url = 'http://192.168.1.126:2020/onvif/service'
    headers = {'Content-Type': 'application/xml'} 
 
    return requests.post(url, headers=headers, data=data)
   
x = absolute_move(0, 0, 0)

print("request at: ", datetime.datetime.utcnow(), '\n')
print("request headers: \n", x.request.headers, '\n')
print("request body: \n", x.request.body, '\n')
print("response headers:\n", x.headers, '\n')
print("response body: \n", x.content.decode("utf-8"))
