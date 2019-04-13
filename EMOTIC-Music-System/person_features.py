from keras_vggface.vggface import VGGFace
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
from keras.utils.data_utils import get_file
import numpy as np

class PersonFeatures():
    def __init__(self, bodies):
        self.bodies = bodies
        # TODO: Look into annotation of bounding boxes

        # TODO: Use bounding box coordinates as dataset for person features
        #  Look into transfer learning/fine-tuning for ImageNet
        #  create 16 layer convolutional neural network with  d1-D kernel
        #  output features of final layer

    def vgg_model():
        face_model = VGGFace(model='resnet50', include_top=False)

