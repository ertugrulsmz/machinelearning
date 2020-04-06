import pandas as pd
import io
import requests
import numpy as np
from sklearn import metrics
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras.callbacks import EarlyStopping

df = pd.read_csv("dataset/iris.csv")

#easy way to handle and see categorical variable.
x = df[['sepal_l', 'sepal_w', 'petal_l', 'petal_w']].values
dummies = pd.get_dummies(df['species']) #One hot Encoding ..
species = dummies.columns
y = dummies.values

#main model
model = Sequential()
model.add(Dense(50, input_dim=x.shape[1], activation='relu')) # Hidden 1
model.add(Dense(25, activation='relu')) # Hidden 2
model.add(Dense(y.shape[1],activation='softmax')) # Output

model.compile(loss='categorical_crossentropy', optimizer='adam')
model.fit(x,y,verbose=2,epochs=100)



from sklearn.metrics import accuracy_score
pred = model.predict(x)
predict_classes = np.argmax(pred,axis=1)
expected_classes = np.argmax(y,axis=1)
correct = accuracy_score(expected_classes,predict_classes)
print(f"Training Accuracy: {correct}")

model.summary()

#model that directly uses previously educated one.
model2 = Sequential()
for layer in model.layers:
    model2.add(layer)
model2.summary()

from sklearn.metrics import accuracy_score
pred = model2.predict(x)
predict_classes = np.argmax(pred,axis=1)
expected_classes = np.argmax(y,axis=1)
correct = accuracy_score(expected_classes,predict_classes)
print(f"Training Accuracy: {correct}")


#Third model will make the transfer
#third model will have 4 output 
model3 = Sequential()
for i in range(2):
    layer = model.layers[i]
    layer.trainable = False
    model3.add(layer)
model3.summary()


model3.add(Dense(4,activation='softmax')) # Output

model3.compile(loss='categorical_crossentropy', optimizer='adam')
model3.summary()





#Next we generate some training data for the 4 fake flowers, and train the neural network.
x = np.array([
    [2.1,0.9,0.8,1.1], # 1
    [2.5,1.2,0.8,1.2],
    [1.1,3.1,1.1,1.1], # 2
    [0.8,2.2,0.7,1.2],
    [1.2,0.7,3.1,1.1], # 3
    [1.0,1.1,2.4,0.9],
    [0.1,1.1,4.1,1.2], # 4
    [1.2,0.8,3.1,0.1],
])

y = np.array([
    [0,0,0,1],
    [0,0,0,1],
    [0,0,1,0],
    [0,0,1,0],
    [0,1,0,0],
    [0,1,0,0],
    [1,0,0,0],
    [1,0,0,0],
])

model3.fit(x,y,verbose=0,epochs=1000)

from sklearn.metrics import accuracy_score
pred = model3.predict(x)
predict_classes = np.argmax(pred,axis=1)
expected_classes = np.argmax(y,axis=1)
correct = accuracy_score(expected_classes,predict_classes)
print(f"Training Accuracy: {correct}")



for i in model.layers:
    print(i.output_shape)