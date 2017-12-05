import cv2
import numpy as np
import xml.etree.ElementTree as ET
import sys
import os.path as path
import json
from scipy.spatial import distance
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.models import model_from_json

# ensure the needed number of command line arguments, quick and dirty way
assert len(sys.argv)==3

# read paths to image and its markdown
image_file=path.abspath(sys.argv[1])
xml_file=path.abspath(sys.argv[2])

# extract the file name without extension
name_stub = path.basename(image_file).split('.')[0]

# if arguments are not existing files, exit
if not (path.isfile(image_file) and path.isfile(xml_file)):
    print("some of files don't exist")
    sys.exit()
    
    
# a keras sequential NN model
model = Sequential()

# load the model structure we are going to use
with open('model.json', 'r') as f:
    model = model_from_json(f.read())

# load pretrained weights    
model.load_weights('first_try.h5')

# a function to arrange coordinates in the right order
def preprocess_coords(xs, ys):
    distances = []
    # calculate all side lengths of the quadrilateral
    for i in range(4):
        distances.append(distance.euclidean(np.float32([xs[i], ys[i]]), np.float32([xs[(i+1)%4], ys[(i+1)%4]])))
    # which one is the longest?
    starting_point=np.argmax(np.array(distances))
    # rearrange coordinates cyclically, so that longest side goes first
    new_xs = xs[starting_point:]+xs[:starting_point]
    new_ys = ys[starting_point:]+ys[:starting_point]
    return new_xs, new_ys

# parse XML markdown
tree = ET.parse(xml_file)
root = tree.getroot()

# read the incoming image as greyscale
image = cv2.imread(image_file, 0)
# a stub for extracting spaces
pts2 = np.float32([[0,60],[0,0],[40,0],[40,60]])
slots = [] # list of preprocessed slot images
ids = [] # list of slot ids
coord=[] # list of slot coordinates

# iterate over all parking spaces
for space in root.findall('space'):
    # extract an id
    space_id = space.get('id')
    ids.append(space_id)
    # extract contour coordinates
    contour = space.find('contour')
    xs = []
    ys = []
    for point in contour.findall('point'):
        xs.append(point.get('x'))
        ys.append(point.get('y'))
    # ensure contour is a quadrilateral. This assertion failed once.
    assert len(xs)==4
    assert len(ys)==4
    # preprocess and save coordinates
    xs, ys = preprocess_coords(xs, ys)
    xs = np.float32(xs)
    ys = np.float32(ys)
    coords=np.vstack((xs, ys)).T
    coord.append(coords)
    # get a matrix for perspective transformation 
    M = cv2.getPerspectiveTransform(coords,pts2)
    # transform a quadrilateral into a solid rectangle
    dst = cv2.warpPerspective(image,M,(40,60))
    slots.append(np.reshape(dst, (40, 60, 1)))

# predict whether each slot is occupied or empty
pred = model.predict(np.array(slots), 16, 1)

pred.ravel().shape, len(ids)
# a dictionary for results of prediction
answer = {}

# construct a JSON entity with results
pred = pred.ravel().tolist()
for i in range(len(ids)):
    answer[ids[i]]='Occupied' if pred[i] else 'Empty'
    # draw bounding quadrilaterals 
    pts = np.array(coord[i], np.int32).reshape((-1, 1, 2))
    color = (0, 255, 0) if pred[i] else (255, 0, 0)
    cv2.polylines(image,[pts],True,color)
 
# save answer to JSON    
with open('%s.json' % (name_stub), 'w') as f:
    f.write(json.dumps(answer))
# save labeled image        
cv2.imwrite('%s_marked.jpg'%(name_stub), image)
