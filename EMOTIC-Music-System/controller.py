import tensorflow as tf
import numpy as np
import scipy.io
from person_features import PersonFeatures
# from global_features import GlobalFeatures
# from fusion_module import FusionModule

test_image = "EMOTIC/emotic/emodb_small/images/0n5sa6o2zfrg3shtoo.jpg"

# send full image to global_feature_extractor

# load in emotic annotations
# Annotations = {train, val, test} 
# filename, folder, image_size, original_database, person 
# person: body_bbox, annotations_categories, annotations_continuous, gender, age
emotic = scipy.io.loadmat('EMOTIC/annotations/Annotations.mat')

# create a dictionary of annotations structures
# hashed by file name
# length of value is number of persons annotated
# [['person'],['gender'],['annotations_categories'],['annotations_continuous'],['gender'],['age']]
emotic_annotations = {}
test_image = "mscoco/images/COCO_train2014_000000140322.jpg"

print("Creating body boxes")
for i in range(len(emotic['train']['filename'][0])):
    file = emotic['train']['filename'][0][i][0]
    # if more than one person is the focus of the image
    for j in range(len(emotic['train']['person'][0][i][0])):
        file_path = str(emotic['train']['folder'][0][i][0]) + "/" + file

        #         if file_path == test_image:
        #             print(np.array(emotic['train']['person'][0][i][0][j]['body_bbox'][0]).tolist())
        #             print(np.array(np.array(np.array(emotic['train']['person'][0][i][0][j]['annotations_categories']).tolist()).tolist()).flatten().tolist())
        #             print(np.array(np.array(emotic['train']['person'][0][i][0][j]['annotations_continuous']).tolist()).flatten().tolist())
        #             print(np.array(emotic['train']['person'][0][i][0][j]['gender']).tolist())
        #             print(np.array(emotic['train']['person'][0][i][0][j]['age']).tolist())

        if file_path in emotic_annotations:
            emotic_annotations[file_path] = np.append(emotic_annotations[file_path],
                                                      # tuples for bounding box coordinates for Image.crop
                                                      [[tuple(np.array(
                                                          emotic['train']['person'][0][i][0][j]['body_bbox'][
                                                              0]).tolist()),
                                                        np.array(np.array(np.array(
                                                            emotic['train']['person'][0][i][0][j][
                                                                'annotations_categories']).tolist()).tolist()).flatten().tolist(),
                                                        np.array(np.array(emotic['train']['person'][0][i][0][j][
                                                                              'annotations_continuous']).tolist()).flatten().tolist(),
                                                        np.array(
                                                            emotic['train']['person'][0][i][0][j]['gender']).tolist(),
                                                        np.array(
                                                            emotic['train']['person'][0][i][0][j]['age']).tolist()]],
                                                      axis=0)

        else:
            emotic_annotations[file_path] = np.array(
                [[tuple(np.array(emotic['train']['person'][0][i][0][j]['body_bbox'][0]).tolist()),
                  np.array(np.array(np.array(emotic['train']['person'][0][i][0][j][
                                                 'annotations_categories']).tolist()).tolist()).flatten().tolist(),
                  np.array(np.array(
                      emotic['train']['person'][0][i][0][j]['annotations_continuous']).tolist()).flatten().tolist(),
                  np.array(emotic['train']['person'][0][i][0][j]['gender']).tolist(),
                  np.array(emotic['train']['person'][0][i][0][j]['age']).tolist()]])
    
# pass bounding box parameters to person to handle
person_features = PersonFeatures(emotic_annotations)
# global_features = GlobalFeatures(annotations['train']['filename'])
# fusion = GlobalFeatures(annotations['train']['filename'])




