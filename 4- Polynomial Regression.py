import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('csvfiles/Position_Salaries.csv')
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values

#No need for splitting data since it is already very less.
#Regression 1
from sklearn.linear_model import LinearRegression
regression1 = LinearRegression()
regression1.fit(X, y)

#Regression2
from sklearn.preprocessing import  PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 4)
X_poly = poly_reg.fit_transform(X)
poly_reg.fit(X_poly, y)
regression2 = LinearRegression()
regression2.fit(X_poly, y)

X_linear1_predict = regression1.predict(X)
plt.scatter(X, y, color = 'red')
plt.plot(X, X_linear1_predict, color = 'blue')
plt.title('Truth or Bluff (Linear Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

# Visualising the Polynomial Regression results
X_regressor2_predict = regression2.predict(X_poly)
plt.scatter(X, y, color = 'red')
plt.plot(X, X_regressor2_predict, color = 'blue')
plt.title('Truth or Bluff (Polynomial Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

# Visualising the Polynomial Regression results (for higher resolution and smoother curve)
X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, y, color = 'red')
plt.plot(X_grid, regression2.predict(poly_reg.fit_transform(X_grid)), color = 'blue')
plt.title('Truth or Bluff (Polynomial Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()


# Predicting a new result with Linear Regression
regression1.predict(np.array([[6.5]]))

# Predicting a new result with Polynomial Regression
regression2.predict(poly_reg.fit_transform(np.array([[6.5]])))