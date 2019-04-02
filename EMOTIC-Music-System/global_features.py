# pretrain using Places datasets to get a good understanding
# of scene recognition

# 16 convolutional layers with 1-D kernels
# effectively modeling 8 layers using 2-D kernels

# outputs features from activation map of last convolutional layer
# to preserve the localization of different parts of image

class GlobalFeatures():
    def __init__(self, images):
        self.images = images

        return self.images


        # TODO: Build base convolutional neural network on PLACES database
        #  as specified in EMOT Recognition
        #  create 16 layer convolutional neural network with  d1-D kernel
        #  output features of final layer