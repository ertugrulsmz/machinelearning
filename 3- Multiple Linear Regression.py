import numpy_ex as np
import pandas as pd

dataset = pd.read_csv("csvfiles/50_Startups.csv")
X = dataset.iloc[:,:-1].values
y = dataset.iloc[:, 4].values

from sklearn.preprocessing import OneHotEncoder, LabelEncoder
labelencoder = LabelEncoder()
X[:, 3] = labelencoder.fit_transform(X[:, 3])
onehotencoder = OneHotEncoder(categorical_features = [3])
X = onehotencoder.fit_transform(X).toarray()

#dummy variable trap
X = X[:,1:]

#Split test and train set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=0)

#Prepare regression
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train,y_train)

#prediction of tests compare results.
y_prediction = regressor.predict(X_test)

#Another prediction for given values
X_try = np.array([1,0,66051,182645,118148])
X_try = np.reshape(X_try,(1,5))
y_prediction_try = regressor.predict(X_try)