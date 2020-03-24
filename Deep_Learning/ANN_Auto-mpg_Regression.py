from __future__ import absolute_import, division, print_function, unicode_literals

import pathlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

#import tensorflow_docs as tfdocs
#import tensorflow_docs.plots
#import tensorflow_docs.modeling

#Load Data
dataset_path = keras.utils.get_file("auto-mpg.data", 
                                    "http://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data")

column_names = ['MPG','Cylinders','Displacement','Horsepower','Weight',
                'Acceleration', 'Model Year', 'Origin']
raw_dataset = pd.read_csv(dataset_path, names=column_names,
                      na_values = "?", comment='\t',
                      sep=" ", skipinitialspace=True)

dataset = raw_dataset.copy()

#PREPROCESS DATA
#dataset.isna().sum() //see non values total count.
dataset = dataset.dropna()

#The "Origin" column is really categorical, not numeric. So convert that to a one-hot:
dataset['Origin'] = dataset['Origin'].map({1: 'USA', 2: 'Europe', 3: 'Japan'})
dataset = pd.get_dummies(dataset, prefix='', prefix_sep='')


#Split Train Set, Test Set
X_train = dataset.sample(frac=0.8,random_state=0)
X_test = dataset.drop(X_train.index)


#Pop Y train, Y test
Y_train = X_train.pop('MPG')
Y_test = X_test.pop('MPG')

#Analyse Data
train_stats = X_train.describe()
#train_stats.pop("MPG") as no longer exist here
train_stats = train_stats.transpose()


#Normalization a.k.a feature scaling
def norm(x):
  return (x - train_stats['mean']) / train_stats['std']
normed_train_data = norm(X_train)
normed_test_data = norm(X_test)


#ANN PART

def build_model():
  #2 hidden Layer with one output without activation 
  model = keras.Sequential([
    layers.Dense(64, activation='relu', #Number of column
                 input_shape=[len(X_train.keys())]),
    layers.Dense(64, activation='relu'),
    layers.Dense(1)
  ])

  optimizer = tf.keras.optimizers.RMSprop(0.001)

  model.compile(loss='mse',
                optimizer=optimizer,
                metrics=['mae', 'mse'])
  return model

model = build_model()

#Now try out the model. Take a batch of 10 examples from the training data and call model.predict on it.
example_batch = normed_train_data[:10]
example_result = model.predict(example_batch)


history = model.fit(normed_train_data,Y_train,epochs=1000)
hist = pd.DataFrame(history.history)
hist['epoch'] = history.epoch
hist.tail()

#Prediction
prediction = model.predict(normed_test_data)

#Evaluation
loss, mae, mse = model.evaluate(normed_test_data, Y_test, verbose=2)

print("Testing set Mean Abs Error: {:5.2f} MPG".format(mae))

















