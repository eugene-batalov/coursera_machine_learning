# -*- coding: utf-8 -*-
import urllib
import json

serviceurl = 'http://python-data.dr-chuck.net/comments_258548.json'

url = serviceurl
uh = urllib.request.urlopen(url)
data = uh.read().decode('utf-8')
info = json.loads(data)

print(sum([item['count'] for item in info['comments']]))