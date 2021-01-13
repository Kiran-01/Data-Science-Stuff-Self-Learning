# -*- coding: utf-8 -*-
"""
Created on Mon Apr 20 11:44:38 2020

@author: a
"""

import statistics as st
sample = [600, 470, 170, 430, 300]
print("Mean is",st.mean(sample))
print("Standard Deviation is",st.pstdev(sample))
print("variance of data is",st.pvariance(sample))

from sklearn.metrics import explained_variance_score
y_true = [3,-0.5,2,7]
y_pred = [2.5,0.0,2,8]
explained_variance_score(y_true, y_pred)

y_true = [[0.5,1],[-1,1],[7,-6]]
y_pred = [[0,2],[-1,2],[8,-5]]
explained_variance_score(y_true, y_pred, multioutput='uniform_average')


from sklearn.metrics import max_error
y_true = [3,2,7,1]
y_pred = [4,2,7,1]
max_error(y_true, y_pred)


#non linear regression
# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Session_10//Salary_DataSet.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 1].values
#Training and Testing Data (divide the data into two part)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test =train_test_split(X,y,test_size=0.2, random_state=0)

'''#in svre we need to do Standerdization method
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X = sc_X.fit_transform(X)
sc_y = StandardScaler()
y = sc_y.fit_transform(np.reshape(y,(10,1)))
'''

# Fitting Regression to the dataset
from sklearn.linear_model import LinearRegression
reg = LinearRegression()
reg.fit(X_train, y_train)

#for predict the test values
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



from sklearn.metrics import mean_absolute_error
mean_absolute_error(y_test, y_predict)


from sklearn.metrics import mean_squared_error
mean_squared_error(y_test, y_predict)


from math import sqrt
from sklearn.metrics import mean_squared_error
result=sqrt(mean_squared_error(y_test, y_predict))

import numpy as np
from sklearn.metrics import mean_squared_log_error
np.sqrt(mean_squared_log_error(y_test, y_predict))

import statsmodels.api as sm
#import statsmodels.formula.api as sm
#import statsmodels.tools.tools.add_constant as sv
X1=sm.add_constant(X)
reg= sm.OLS(y,X1).fit()
reg.summary()
