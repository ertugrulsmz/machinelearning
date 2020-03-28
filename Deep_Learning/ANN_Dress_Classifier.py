
from tensorflow import keras
from keras import layers,Sequential
import numpy as np
import matplotlib.pyplot as plt

data = keras.datasets.fashion_mnist

(train_images,train_labels),(test_images, test_labels) = data.load_data()

label_names = ["T-shirt","Trouser","Pullover","Dress","Coat","Sandal","Shirt",
               "Sneaker","Bag","Ankle Boot"]


train_images = train_images/255.0
test_images = test_images/255.0


model = Sequential(
    [
        layers.Flatten(input_shape=(28,28)),
        layers.Dense(128,activation ="relu"),
        layers.Dense(64,activation ="relu"),
        layers.Dense(10,activation="softmax")
    ]
)

model.compile(optimizer="adam",loss="sparse_categorical_crossentropy",metrics=["accuracy"])
model.fit(train_images,train_labels,epochs=10)

#test_loss, test_accuracy = model.evaluate(test_images,test_labels)
#print("Last Tested accuracy : ",test_accuracy)

prediction = model.predict(test_images)

#displaying ten results...
for i in range(10):
    plt.imshow(test_images[i])
    plt.xlabel(label_names[test_labels[i]])
    plt.ylabel(label_names[np.argmax(prediction[i])])
    plt.show()