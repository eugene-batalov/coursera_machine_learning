# -*- coding: utf-8 -*-

#import urllib
import json
import codecs

#serviceurl = 'http://python-data.dr-chuck.net/geojson?'
#
#address = input('Enter location: ')
#if len(address) < 1 : address = 'University of Colorado at Boulder'
#
#url = serviceurl + urllib.parse.urlencode({'sensor':'false', 'address': address})
#uh = urllib.request.urlopen(url)
#data = uh.read().decode('utf-8')
with codecs.open('trello_gp.json', 'r', 'utf_8_sig') as file:
    strData = file.read()
info = json.loads(strData)
todoLists = [list_ for list_ in info['lists']][:2]
todoListsIds = [list_['id'] for list_ in todoLists]
todoCards = [card['name'] for card in info['cards'] if card['idList'] in todoListsIds and card['closed'] == False]
print('cards:', todoCards)