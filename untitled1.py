import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers.convolutional import Convolution2D
from keras.layers.convolutional import MaxPooling2D
from IPython.display import display
from PIL import Image


from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator()

train_generator = train_datagen.flow_from_directory(
        '/Users/CK/data/train',
        target_size=(224,224),
        batch_size=10,
        class_mode ='categorical'
        )
validation_datagen = ImageDataGenerator()

validation_generator = validation_datagen.flow_from_directory(
        '/Users/CK/data/validation',
        target_size=(224,224),
        batch_size = 10,
        class_mode = 'categorical'
        )



model = Sequential()
model.add(Convolution2D(12,(3, 3), input_shape=(224, 224,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Convolution2D(2, (3, 3), border_mode='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Convolution2D(3, (2, 2), border_mode='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(5, activation='relu'))
model.add(Dense(10, activation='softmax'))
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit_generator(
        train_generator,
        samples_per_epoch=46827,
        nb_epoch=10,
        validation_data=validation_generator,
        nb_val_samples=46827)