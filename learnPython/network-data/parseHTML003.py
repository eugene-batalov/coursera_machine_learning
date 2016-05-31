# -*- coding: utf-8 -*-

import urllib
import xml.etree.ElementTree as ET

serviceurl = 'http://python-data.dr-chuck.net/comments_258544.xml'

url = serviceurl
print ('Retrieving', url)
uh = urllib.request.urlopen(url)
data = uh.read()
print ('Retrieved',len(data),'characters')
#print (data)
tree = ET.fromstring(data)

counts = tree.findall('.//count')

print (sum([int(count.text) for count in counts]))