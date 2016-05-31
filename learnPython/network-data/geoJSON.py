# -*- coding: utf-8 -*-

import urllib
import json

serviceurl = 'http://python-data.dr-chuck.net/geojson?'

address = input('Enter location: ')
if len(address) < 1 : address = 'University of Colorado at Boulder'

url = serviceurl + urllib.parse.urlencode({'sensor':'false', 'address': address})
uh = urllib.request.urlopen(url)
data = uh.read().decode('utf-8')
info = json.loads(data)

print(info['results'][0]['place_id'])
print(dir(info))