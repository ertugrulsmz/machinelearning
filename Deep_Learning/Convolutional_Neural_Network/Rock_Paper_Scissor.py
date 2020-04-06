from keras_preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras import layers    

training_datagen = ImageDataGenerator(
      rescale = 1./255,
	  rotation_range=45,
      width_shift_range=0.2,
      height_shift_range=0.2,
      shear_range=0.2,
      zoom_range=0.2,
      horizontal_flip=True,
      fill_mode='nearest')

validation_datagen = ImageDataGenerator(rescale = 1./255)

train_generator = training_datagen.flow_from_directory(
	"../dataset/rock_paper_scissors/training_set",
	target_size=(150,150),
	class_mode='categorical'
)

validation_generator = validation_datagen.flow_from_directory(
	"../dataset/rock_paper_scissors/test_set",
	target_size=(150,150),
	class_mode='categorical'
)

rpsmodel = Sequential([
    layers.Conv2D(64,(3,3), activation='relu', input_shape=(150,150,3)),
    layers.MaxPooling2D(2,2),
    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D(2,2),
    layers.Conv2D(128, (3,3), activation='relu'),
    layers.MaxPooling2D(2,2),
    layers.Conv2D(128, (3,3), activation='relu'),
    layers.MaxPooling2D(2,2),
    layers.Flatten(),
    layers.Dropout(0.5),
    layers.Dense(512,activation='relu'),
    layers.Dense(3,activation='softmax')
    ])

rpsmodel.compile(loss = 'categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
history = rpsmodel.fit(train_generator, epochs=25, validation_data = validation_generator, verbose = 1)



#plot loss accuracy of training and test 
import matplotlib.pyplot as plt
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(acc))

plt.plot(epochs, acc, 'r', label='Training accuracy')
plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.legend(loc=0)
plt.figure()


plt.show()







