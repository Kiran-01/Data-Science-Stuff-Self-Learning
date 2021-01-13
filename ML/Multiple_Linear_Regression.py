# -*- coding: utf-8 -*-
"""
Created on Sun Apr 19 23:21:28 2020

@author: a
"""


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


#Import the data set from Desktop

dataset = pd.read_csv('Session_05//M_Regression.csv')

X=dataset.iloc[:,:-1].values#independent
y=dataset.iloc[:,3].values#dependent


#Training and Testing Data (divide the data into two part)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test =train_test_split(X,y,test_size=0.3, random_state=0)



#regression 
from sklearn.linear_model import LinearRegression
reg = LinearRegression()
reg.fit(X_train, y_train)

y_predict = reg.predict(X_test)

#we cant draw plots on multiple linear regression