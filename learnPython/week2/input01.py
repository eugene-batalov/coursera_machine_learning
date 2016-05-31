# -*- coding: utf-8 -*-
"""
Created on Sat Mar 26 10:14:44 2016

@author: galinabatalova
"""
import numpy as np
import base64

#
#name = input('your name: ')
#print('hello, ', name)

l = np.arange(1,301, dtype=np.int16)
np.random.shuffle(l)
#print(l)

p = base64.b64encode(l)

print(p)