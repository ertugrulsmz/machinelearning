import numpy_ex as np
import pandas as pd

dataset = pd.read_csv("csvfiles/data.csv")
X = dataset.iloc[:,:-1].values
y = dataset.iloc[:,3].values


from sklearn.impute import SimpleImputer
imp = SimpleImputer(missing_values=np.nan, strategy='mean')
X[:,1:3] = imp.fit_transform(X[:,1:3])

# Encoding categorical data
# Encoding the Independent Variable
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder = LabelEncoder()
X[:,0] = labelencoder.fit_transform(X[:,0])
#OneHotCoding for no prioriority between data
onehotencoder = OneHotEncoder(categorical_features=[0])
X = onehotencoder.fit_transform(X).toarray()

#easy way to handle and see categorical variable.
#x = df[['sepal_l', 'sepal_w', 'petal_l', 'petal_w']].values
#dummies = pd.get_dummies(df['species']) #One hot Encoding ..
#species = dummies.columns
#y = dummies.values



labelencoder_y = LabelEncoder()
y = labelencoder.fit_transform(y)

#Splitting train set , test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=0)














