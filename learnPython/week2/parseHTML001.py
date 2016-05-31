# -*- coding: utf-8 -*-
"""
Created on Sun May 15 07:28:20 2016

@author: galinabatalova
"""
import urllib
from bs4 import BeautifulSoup

url = input('Enter - ')
if(len(url) < 1):
    url = 'http://python-data.dr-chuck.net/known_by_Holli.html'
html = urllib.request.urlopen(url).read()

soup = BeautifulSoup(html, 'html.parser')
tags = soup('a')
position = 18
repeat = 7
i = 1
for r in range(1,repeat+1):
    for tag in tags:
        if i < position:
            i += 1
            continue
        name = tag.contents[0]
        url = tag.get('href', None)
        print(url)
        soup = BeautifulSoup(urllib.request.urlopen(url).read(), 'html.parser')
        tags = soup('a')
        i = 1;
        break;
        #print ('TAG:',tag)
        #print ('URL:',tag.get('href', None))
        #print ('Contents:',tag.contents[0])
        #print ('Attrs:',tag.attrs)
print(name)