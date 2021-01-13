# -*- coding: utf-8 -*-
"""
Created on Thu Apr 23 20:03:04 2020

@author: a
"""

import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt


dataset = pd.read_csv('Session_46//ads.csv')

#Implementing the random selection
import random
N =10000
d = 10
ads_selected = []
total_reward = 0
for n in range(0,N):
    ad = random.randrange(d)
    ads_selected.append(ad)
    reward = dataset.values[n, ad]
    total_reward = total_reward + reward
    
    
    #Visualization
plt.hist(ads_selected)
plt.title('Histogram of ads selections')
plt.xlabel('Ads')
plt.ylabel('Number of times each ad was selected')
plt.show() 