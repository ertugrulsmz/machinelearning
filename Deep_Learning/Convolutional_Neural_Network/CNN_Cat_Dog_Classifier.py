from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense

#Initialize the CNN
classifier = Sequential()

#S1 Convolution
classifier.add(Convolution2D(32, 3, 3, input_shape=(64,64,3), activation="relu"))  

#S2 Pooling
classifier.add(MaxPooling2D(pool_size = (2,2)))

#S3 Flattening
classifier.add(Flatten())

#S4 Full Connection
classifier.add(Dense(output_dim = 128, activation = 'relu'))
classifier.add(Dense(output_dim = 1, activation = 'sigmoid'))

#Compile
classifier.compile(optimizer = 'adam', loss='binary_crossentropy', metrics=['accuracy'])

#Fitting Part
from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale = 1./255)

training_set = train_datagen.flow_from_directory('dataset/cat_dog_dataset/training_set',
                                                 target_size = (64, 64),
                                                 batch_size = 32,
                                                 class_mode = 'binary')

test_set = test_datagen.flow_from_directory('dataset/cat_dog_dataset/test_set',
                                            target_size = (64, 64),
                                            batch_size = 32,
                                            class_mode = 'binary')

classifier.fit_generator(training_set,
                         samples_per_epoch = 8000,
                         nb_epoch = 10,
                         validation_data = test_set,
                         nb_val_samples = 2000)


