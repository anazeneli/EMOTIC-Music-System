from keras_vggface.vggface import VGGFace
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
from keras.utils.data_utils import get_file
from PIL import Image
import numpy as np
import scipy.io


class PersonFeatures():
    def __init__(self, annotations):
        # pull out body boxes
        self.bodies = annotations

        # crop images
        cropped_images = []
        # load in training images body boxes
        for k, v in self.bodies.items():
            for b in v[:, 0]:
                # open image for cropping
                img = Image.open("EMOTIC/emotic/" + test_image)
                #         img.show()
                # crop the image
                img_crop = img.crop(box=b)
                #         img_crop.show()
                cropped_images.append(b)

    #         self.features = vgg_model(cropped_images)

    def vgg_model(images):
        face_model = VGGFace(model='resnet50', include_top=False)

        # preprocess images
        # TODO: preprocess images
        features = []
        for i in images:
            # load image setting the image size to 224 x 224
            img = image.load_img(img_path, target_size=(224, 224))

            # convert image to numpy array
            x = image.img_to_array(img)
            # the image is now in an array of shape (3, 224, 224)
            # need to expand it to (1, 3, 224, 224) as preprocess takes in a list
            x = np.expand_dims(x, axis=0)
            x = preprocess_input(x)
            # extract the features
            feats = model.predict(x)[0]
            # convert from Numpy to a list of values
            features.append(np.char.mod('%f', feats))

            # TODO: reduce feature dimensionality

        return features


# p = PersonFeatures(emotic_annotations)
# print (p.features)