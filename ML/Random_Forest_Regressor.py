# -*- coding: utf-8 -*-
"""
Created on Mon Apr 20 11:35:38 2020

@author: a
"""

#non linear regression
# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Session_09//Poly_dataSet.csv')
X = dataset.iloc[:, 0:1].values
y = dataset.iloc[:, 1].values

'''#in svre we need to do Standerdization method
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X = sc_X.fit_transform(X)
sc_y = StandardScaler()
y = sc_y.fit_transform(np.reshape(y,(10,1)))
'''

# Fitting Regression to the dataset
from sklearn.ensemble import RandomForestRegressor
rf_reg = RandomForestRegressor(n_estimators = 10,random_state= 0)
rf_reg.fit(X, y)

# Visualising the Linear Regression results
plt.scatter(X, y, color = 'red')
plt.plot(X, rf_reg.predict(X), color = 'blue')
plt.title('(Random Forest Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

X_grid=np.arange(min(X),max(X),0.1)
X_grid=X_grid.reshape(len(X_grid),1)
plt.scatter(X, y, color = 'red')
plt.plot(X_grid, rf_reg.predict(X_grid), color = 'blue')
plt.title('(Decision Tree Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')

#Prdict by SV Regression
rf_reg.predict(np.reshape(6.5,(1,1)))

# Predicting a new result with SV Regression
#predict = sc_y.inverse_transform(dt_reg.predict(sc_X.transform(np.array([[6.5]]))))




