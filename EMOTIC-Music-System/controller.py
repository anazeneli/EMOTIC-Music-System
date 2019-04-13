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
# hashed by fiale name 
# length of value is number of persons annotated 
# [['person'],['gender'],['annotations_categories'],['annotations_continuous'],['gender'],['age']]
emotic_annotations = {}

print ("Creating body boxes")
for i in range(len(emotic['train']['filename'][0])): 
    file = emotic['train']['filename'][0][i][0]
    
    # if more than one person is the focus of the image 
    for j in range(len(emotic['train']['person'][0][i][0])):      
        if file in emotic_annotations: 
            emotic_annotations[file].append([np.array(emotic['train']['person'][0][i][0][j]['body_bbox'][0]).tolist(),
                                             np.array(np.array(np.array(emotic['train']['person'][0][i][0][j]['annotations_categories']).tolist()).tolist()).flatten().tolist(),      
                                             np.array(np.array(emotic['train']['person'][0][i][0][j]['annotations_continuous']).tolist()).flatten().tolist(),
                                             np.array(emotic['train']['person'][0][i][0][j]['gender']).tolist(),
                                             np.array(emotic['train']['person'][0][i][0][j]['age']).tolist()])            
        else: 
            emotic_annotations[file]= [[np.array(emotic['train']['person'][0][i][0][j]['body_bbox'][0]).tolist(),
                                        np.array(np.array(np.array(emotic['train']['person'][0][i][0][j]['annotations_categories']).tolist()).tolist()).flatten().tolist(),                             
                                        np.array(np.array(emotic['train']['person'][0][i][0][j]['annotations_continuous']).tolist()).flatten().tolist(),
                                        np.array(emotic['train']['person'][0][i][0][j]['gender']).tolist(),
                                        np.array(emotic['train']['person'][0][i][0][j]['age']).tolist()]]    

# for k,v in emotic_annotations.items(): 
#     print(k, v)
    
# pass bounding box parameters to person to handle
person_features = PersonFeatures(emotic_annotations)
# global_features = GlobalFeatures(annotations['train']['filename'])
# fusion = GlobalFeatures(annotations['train']['filename'])




