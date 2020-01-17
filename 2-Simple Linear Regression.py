import numpy_ex as np
import matplotlib.pyplot as plt
import pandas as pd

dataset=pd.read_csv('csvfiles/Salary_Data.csv')
X=dataset.iloc[:,:-1].values
Y=dataset.iloc[:,1].values


from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=0)

from sklearn.linear_model import LinearRegression
linearRegressionObj = LinearRegression()
linearRegressionObj.fit(X_train,Y_train)

# Predicting the Test set results
y_pred = linearRegressionObj.predict(X_test)



#plot train
plt.scatter(X_train,Y_train,color='red')
plt.plot(X_train,linearRegressionObj.predict(X_train),color='yellow')
plt.title('Salary vs Experience (Training set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()

#plot test
plt.scatter(X_test, Y_test, color = 'red')
plt.plot(X_train, linearRegressionObj.predict(X_train), color = 'yellow')
plt.title('Salary vs Experience (Test set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()

#plot test with x_test
plt.scatter(X_test, Y_test, color = 'red')
plt.plot(X_test, linearRegressionObj.predict(X_test), color = 'yellow')
plt.title('Salary vs Experience (Test set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()

