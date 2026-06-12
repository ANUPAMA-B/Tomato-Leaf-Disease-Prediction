import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# Importing the Libraries
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import Model
from keras.applications.vgg16 import VGG16
from tensorflow.keras.layers import Input,Lambda,Dense,Flatten 

from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator,load_img

#from keras.applications.vgg16 import VGG
from glob import glob
# Re-Size the Image
Image_Size = [224,224]

train_path = r"tomato\New Plant Diseases Dataset(Augmented)\New Plant Diseases Dataset(Augmented)\train"

valid_path = r"tomato\New Plant Diseases Dataset(Augmented)\New Plant Diseases Dataset(Augmented)\valid"
model_save_path = r"model.h5"

vgg16 = VGG16(input_shape= Image_Size + [3], weights= "imagenet",include_top= False)

for layer in vgg16.layers:
  layer.trainable = False

folders = glob("tomato/New Plant Diseases Dataset(Augmented)/New Plant Diseases Dataset(Augmented)/valid/*")
print(folders)

x = Flatten()(vgg16.output)

len(folders)
prediction = Dense(len(folders),activation = "softmax")(x)

model = Model(inputs = vgg16.input,outputs = prediction)

model.summary()

# tell the model what cost and optimization method to use
model.compile(
  loss='categorical_crossentropy',
  optimizer='adam',
  metrics=['accuracy']
)


# Data Preprocesing

train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale = 1./255)


# Make sure you provide the same target size as initialied for the image size
training_set = train_datagen.flow_from_directory("tomato/New Plant Diseases Dataset(Augmented)/New Plant Diseases Dataset(Augmented)/train",
                                                 target_size = (224, 224),
                                                 batch_size = 32,
                                                 class_mode = 'categorical')

test_set = test_datagen.flow_from_directory("tomato/New Plant Diseases Dataset(Augmented)/New Plant Diseases Dataset(Augmented)/valid",
                                            target_size = (224,224),
                                            batch_size = 32,
                                            class_mode = "categorical")


# fit the model
# Run the cell. It will take some time to execute
model = Model(inputs = vgg16.input,outputs = prediction)
r = model.fit_generator(
  training_set,
  validation_data=test_set,
  epochs=2,
  steps_per_epoch=len(training_set),
  validation_steps=len(test_set)
)

model.save(model_save_path)
print(f"Model saved at {model_save_path}")
