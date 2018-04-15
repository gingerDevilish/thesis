import cv2
import numpy as np
import xml.etree.ElementTree as ET
import sys
import os.path as path
from scipy.spatial import distance

assert len(sys.argv)==3

image_file=path.abspath(sys.argv[1])
xml_file=path.abspath(sys.argv[2])

name_stub = path.basename(image_file).split('.')[0]

if not (path.isfile(image_file) and path.isfile(xml_file)):
    print("some of files don't exist")
    sys.exit()

def preprocess_coords(xs, ys):
    distances = []
    for i in range(4):
        distances.append(
            distance.euclidean(
                np.float32([xs[i], ys[i]]), 
                np.float32([xs[(i+1)%4], ys[(i+1)%4]])))
    starting_point=np.argmax(np.array(distances))
    new_xs = xs[starting_point:]+xs[:starting_point]
    new_ys = ys[starting_point:]+ys[:starting_point]
    return new_xs, new_ys


tree = ET.parse(xml_file)
root = tree.getroot()


image = cv2.imread(image_file, 0)
pts2 = np.float32([[0,60],[0,0],[40,0],[40,60]])


for space in root.findall('space'):
    space_id = space.get('id')
    contour = space.find('contour')
    xs = []
    ys = []
    for point in contour.findall('point'):
        xs.append(point.get('x'))
        ys.append(point.get('y'))
    assert len(xs)==4
    assert len(ys)==4
    xs, ys = preprocess_coords(xs, ys)
    xs = np.float32(xs)
    ys = np.float32(ys)
    coords=np.vstack((xs, ys)).T
    M = cv2.getPerspectiveTransform(coords,pts2)
    dst = cv2.warpPerspective(image,M,(40,60))
    cv2.imwrite("%s_%s.jpg" % (name_stub, space_id), dst)

