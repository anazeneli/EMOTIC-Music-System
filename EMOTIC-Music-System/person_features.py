from keras_vggface.vggface import VGGFace
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
from keras.utils.data_utils import get_file
import numpy as np
import scipy.io
import cv2


class PersonFeatures():
    def __init__(self, train, test):
        print("LETS DO THIS ")
        # pull out body boxes
        self.train = train
        self.test = test

        X_train = self.extract_bodies(self.train)
        X_test = self.extract_bodies(self.test)

        # Compute a PCA
        n_components = 100
        print("PCA")
        pca = PCA(n_components=n_components, whiten=True).fit(X_train)

        # apply PCA transformation
        X_train_pca = pca.transform(X_train)
        X_test_pca = pca.transform(X_test)

        self.train_feats = self.extract_features(X_train_pca)
        self.test_feats = self.extract_features(X_test_pca)

    def extract_bodies(self, images):
        print("BODIES")

        dir_path = "EMOTIC/emotic/"
        #         test_image = 'framesdb/images/frame_ghkq7yp0itqz0kn7.jpg'

        # crop images
        cropped_images = []
        # load in training images body boxes
        for k, v in images.items():
            for b in v[:, 0]:
                # read image for cropping
                img = cv2.imread(dir_path + k)
                #                 cv2.imshow("image", img)
                cropped_img = img[b[1]:b[3], b[0]:b[2]]
                #                 cv2.imshow("cropped", cropped_img)
                #                 cv2.waitKey(0)

                cropped_images.append(cropped_img)

        return cropped_images

    def extract_features(self, images):
        print("FEATURES")

        face_model = VGGFace(model='resnet50', include_top=False)

        # list of pre-processed images
        imgs = []
        for i in images:
            # load image setting the image size to 224 x 224
            img = image.load_img(img_path, target_size=(224, 224))

            # convert image to numpy array
            x = image.img_to_array(img)
            # the image is now in an array of shape (3, 224, 224)
            # need to expand it to (1, 3, 224, 224) as preprocess takes in a list
            x = np.expand_dims(x, axis=0)
            # ensure proper input format for model
            x = preprocess_input(x)

            # extract the features
            imgs.append(x)

        imgs = np.concatenate(imgs, axis=0)
        img_feats = face_model.predict(imgs)

        return img_feats


