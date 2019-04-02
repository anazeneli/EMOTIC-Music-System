from __future__ import absolute_import, division, print_function
import os
import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# # Directory with our training cat pictures
# train_faces_dir = os.path.join(train_dir, 'faces')
# print ('Total training face images:', len(os.listdir(train_faces_dir)))
#
# # Directory with our validation faces pictures
# validation_faces_dir = os.path.join(validation_dir, 'cats')
# print ('Total validation face images:', len(os.listdir(validation_faces_dir)))

# image_size = 224 # All images will be resized to 160x160
# batch_size = 52
#
# # Rescale all images by 1./255 and apply image augmentation
# train_datagen = keras.preprocessing.image.ImageDataGenerator(
#                 rescale=1./255)
#
# validation_datagen = keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
#
# # Flow training images in batches of 20 using train_datagen generator
# train_generator = train_datagen.flow_from_directory(
#                 train_dir,  # Source directory for the training images
#                 target_size=(image_size, image_size),
#                 batch_size=batch_size,
#                 # Since we use binary_crossentropy loss, we need binary labels
#                 class_mode='binary')
#
# # Flow validation images in batches of 20 using test_datagen generator
# validation_generator = validation_datagen.flow_from_directory(
#                 validation_dir, # Source directory for the validation images
#                 target_size=(image_size, image_size),
#                 batch_size=batch_size,
#                 class_mode='binary')
# IMG_SHAPE = (image_size, image_size, 3)
#
# # Create the base model from the pre-trained model ResNet50
# base_model = tf.keras.applications.ResNet50(input_shape=IMG_SHAPE,
#                                                include_top=False,
#                                                weights='imagenet')
#
# base_model.trainable = False
# # Let's take a look at the base model architecture
# base_model.summary()
# model = tf.keras.Sequential([
#   base_model,
#   keras.layers.GlobalAveragePooling2D(),
#   keras.layers.Dense(1, activation='sigmoid')
# ])
# model.compile(optimizer=tf.keras.optimizers.RMSprop(lr=0.0001),
#               loss='binary_crossentropy',
#               metrics=['accuracy'])
#
# model.summary()
# len(model.trainable_variables)
# epochs = 10
# steps_per_epoch = train_generator.n // batch_size
# validation_steps = validation_generator.n // batch_size
#
# history = model.fit_generator(train_generator,
#                               steps_per_epoch = steps_per_epoch,
#                               epochs=epochs,
#                               workers=4,
#                               validation_data=validation_generator,
#                               validation_steps=validation_steps)
#
# base_model.trainable = True
# # Let's take a look to see how many layers are in the base model
# print("Number of layers in the base model: ", len(base_model.layers))
#
# # Fine tune from this layer onwards
# fine_tune_at = 100
#
# # Freeze all the layers before the `fine_tune_at` layer
# for layer in base_model.layers[:fine_tune_at]:
#   layer.trainable =  False
#
# model.compile(loss='binary_crossentropy',
#               optimizer = tf.keras.optimizers.RMSprop(lr=2e-5),
#               metrics=['accuracy'])
#
# model.summary()
#
# len(model.trainable_variables)
#
# history_fine = model.fit_generator(train_generator,
#                                    steps_per_epoch = steps_per_epoch,
#                                    epochs=epochs,
#                                    workers=4,
#                                    validation_data=validation_generator,
#                                    validation_steps=validation_steps)
#


# 16 convolutional layers with 1-D kernels
# effectively modeling 8 layers using 2-D kernels
# outputs features from activation map of last convolutional layer
# to preserve the localization of different parts of image

class PersonFeatures():
    def __init__(self, bodies):
        self.bodies = bodies
        # TODO: Look into annotation of bounding boxes

        # TODO: Use bounding box coordinates as dataset for person features
        #  create 16 layer convolutional neural network with  d1-D kernel
        #  output features of final layer