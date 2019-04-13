from keras.preprocessing import image
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input
from keras.utils.data_utils import get_file
import numpy as np

# pretrain using Places365 dataset: https://github.com/CSAILVision/places365

# 16 convolutional layers with 1-D kernels
# effectively modeling 8 layers using 2-D kernels

# outputs features from activation map of last convolutional layer
# to preserve the localization of different parts of image
import torch
class GlobalFeatures():
    def __init__(self, images):
        self.images = images

        # return self.images


    def vgg_places():

        # weights path to places365 dataset for location segmentation
        weights = 'https://github.com/GKalliatakis/Keras-VGG16-places365/releases/download/v1.0/vgg16-hybrid1365_weights_tf_dim_ordering_tf_kernels_notop.h5'
        model = VGG16()
        weights_path = get_file('vgg16-hybrid1365_weights_tf_dim_ordering_tf_kernels_notop.h5', weights, cache_subdir='models')

        # load pre-trained model minus top for feature extraction
        model = VGG16(weights=weights_path, include_top=False)

        img_path = '12.jpg'
        # load image setting the image size to 224 x 224
        img = image.load_img(img_path, target_size=(224, 224))
        # convert image to numpy array
        x = image.img_to_array(img)
        # the image is now in an array of shape (3, 224, 224)
        # need to expand it to (1, 3, 224, 224) as preprocess takes in a list
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        # extract the features
        features = model.predict(x)[0]
        # convert from Numpy to a list of values
        features_arr = np.char.mod('%f', features)

        # TODO: reduce dimensionality
