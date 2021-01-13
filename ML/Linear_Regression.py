# -*- coding: utf-8 -*-
"""
Created on Sun Apr 19 22:39:00 2020

@author: a
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


#Import the data set from Desktop

dataset = pd.read_csv('Session_04//Salary_DataSet.csv')
dataset.drop()
X=dataset.iloc[:,:-1].values
y=dataset.iloc[:,1].values


#Training and Testing Data (divide the data into two part)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test =train_test_split(X,y,test_size=0.3, random_state=0)



#regression 
from sklearn.linear_model import LinearRegression
reg = LinearRegression()
reg.fit(X_train, y_train)

y_predict = reg.predict(X_test)

#visualize
plt.scatter(X_train,y_train,color = 'red')
plt.plot(X_train, reg.predict(X_train), color = 'blue')
plt.title("Linear Regression Salary Vs. Experience")
plt.xlabel("Years of Employee")
plt.ylabel("Salaries of Employee")
plt.show()


plt.scatter(X_test,y_test,color = 'red')
plt.plot(X_train, reg.predict(X_train), color = 'blue')
plt.title("Linear Regression Salary Vs. Experience")
plt.xlabel("Years of Employee")
plt.ylabel("Salaries of Employee")
plt.show()