import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator

#preprocesing test set
train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

training_set = train_datagen.flow_from_directory(
    'training_set', 
    target_size=(64,64),
    batch_size=32,
    class_mode='binary'
    )
#preprocesing training set

test_datagen = ImageDataGenerator(rescale=1./255)
test_set = train_datagen.flow_from_directory(
    'test_set', 
    target_size=(64,64),
    batch_size=32,
    class_mode='binary'
    )

cnn = tf.keras.models.Sequential()
cnn.add(tf.keras.layers.Conv2D(filters=32,kernel_size=3, activation='relu', input_shape=[64,64,3]))

#pooling 
cnn.add(tf.keras.layers.MaxPooling2D())
#adding second layer 
cnn.add(tf.keras.layers.Conv2D(filters=32,kernel_size=3, activation='relu'))
cnn.add(tf.keras.layers.MaxPooling2D())
#flatening 
cnn.add(tf.keras.layers.Flatten())
#layer connecting 
cnn.add(tf.keras.layers.Dense(units=128, activation="relu"))
#output layer
cnn.add(tf.keras.layers.Dense(units=1, activation="sigmoid"))
cnn.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
#traning cnn 
cnn.fit(training_set,
        steps_per_epoch = 8000,
        epochs = 25,
        validation_data = test_set,
        validation_steps = 2000)

import numpy as np
from keras.preprocessing import image
test_image = image.load_img("single_prediction/cat_or_dog_1.jpg", target_size=(64,64))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis=0)

result = cnn.predict(test_image)#1 is dog 0 is catto
print(result[0][0])