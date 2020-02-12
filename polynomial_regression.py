
# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values

#fitting linear regression to the dataset
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X, y)

#fitting polynomial regression to the dataset
#transformed X into X_poly
from sklearn.preprocessing import PolynomialFeatures
pol_reg = PolynomialFeatures(degree = 3)
X_poly = pol_reg.fit_transform(X)
#created a new linear regression model and fitted X_poly and y
lin_reg2 = LinearRegression()
lin_reg2.fit(X_poly, y)


#visualizing the linear regression results
plt.scatter(X, y, color = 'red')
plt.plot(X, lin_reg.predict(X), color = 'blue')
plt.title('Truth or Bluff (Linear Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

#visualizing the linear regression results
#extra
X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape(len((X_grid), 1))
plt.scatter(X, y, color = 'red')
plt.plot(X, lin_reg2.predict(pol_reg.fit_transform(X)), color = 'blue')
plt.title('Truth or Bluff (Linear Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()