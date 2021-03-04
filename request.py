#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import requests

url = 'http://localhost:5000/predict_api'
r = requests.post(url,json={'months since the last donation':9, 'total number of donation':3, 'total blood donated in c.c.':750, 'months since the first donation':52})

print(r.json())

