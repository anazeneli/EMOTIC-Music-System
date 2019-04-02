import tensorflow as tf
import numpy as np
import scipy.io
from person_features import PersonFeatures
from global_features import GlobalFeatures
from fusion_module import FusionModule

test_image = "EMOTIC/emotic/emodb_small/images/0apauiwt5nd0id0qx2.jpg"


# send full image to global_feature_extractor

# load in annotations
annotations = scipy.io.loadmat('EMOTIC/annotations/Annotations.mat')

# segment the image into the region of the image comprising of
# the person to person_feature_extractor
# body box coordinates are pre-labeled
bodies = []
for i in annotations['train']:
    for j in i['person']:
        bodies.append([i['filename'], j['body_bbox']])

# pass bounding box parameters to person to handle
person_features = PersonFeatures(bodies)
global_features = GlobalFeatures(annotations['train']['filename'])
fusion = GlobalFeatures(annotations['train']['filename'])
