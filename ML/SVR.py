# -*- coding: utf-8 -*-
"""
Created on Mon Apr 20 00:25:47 2020

@author: a
"""


# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Session_07//Poly_dataSet.csv')
X = dataset.iloc[:, 0:1].values
y = dataset.iloc[:, 1].values

#Feature Scaling
#in svr we need to do Standerdization method
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X = sc_X.fit_transform(X)
sc_y = StandardScaler()
y = sc_y.fit_transform(np.reshape(y,(10,1)))

# Fitting Regression to the dataset
from sklearn.svm import SVR
svr_reg = SVR(kernel ='rbf')
svr_reg.fit(X, y)

# Visualising the Linear Regression results
plt.scatter(X, y, color = 'red')
plt.plot(X, svr_reg.predict(X), color = 'blue')
plt.title('(SVR Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()


#Prdict by SV Regression
svr_reg.predict(np.reshape(6.5,(1,1)))

# Predicting a new result with SV Regression
predict = sc_y.inverse_transform(svr_reg.predict(sc_X.transform(np.array([[6.5]]))))

predict1 = sc_y.inverse_transform(svr_reg.predict(sc_X.transform(np.array([[9.5]]))))
predict2 = sc_y.inverse_transform(svr_reg.predict(sc_X.transform(np.array([[10]]))))


