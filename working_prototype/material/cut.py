import cv2
import numpy as np
import xml.etree.ElementTree as ET
import sys
import os
import os.path as path
from scipy.spatial import distance
from tqdm import tqdm


def preprocess_coords(xs, ys):
    distances = []
    for i in range(4):
        distances.append(distance.euclidean(np.float32([xs[i], ys[i]]),
                         np.float32([xs[(i + 1) % 4], ys[(i + 1)
                         % 4]])))
    starting_point = np.argmax(np.array(distances))
    new_xs = xs[starting_point:] + xs[:starting_point]
    new_ys = ys[starting_point:] + ys[:starting_point]
    return (new_xs, new_ys)


epath = '/media/kostin_001/5FDF9B9F5A024FA7/thesis/PKLot/segments/empty'
opath = '/media/kostin_001/5FDF9B9F5A024FA7/thesis/PKLot/segments/occupied'
image_file = ""
xml_file = ""

for (d, dirs, files) in tqdm(os.walk('/media/kostin_001/5FDF9B9F5A024FA7/thesis/PKLot/PKLot')):
    for x in files:
        if x.endswith('.xml'):
            name = x.split('.')[0]
            image_file = name + '.jpg'
            xml_file = name + '.xml'
            
            tree = ET.parse(os.path.join(d, xml_file))
            root = tree.getroot()

            image = cv2.imread(os.path.join(d,image_file), 0)
            pts2 = np.float32([[0, 60], [0, 0], [40, 0], [40, 60]])

            for space in root.findall('space'):
                space_id = space.get('id')
                space_occupied=space.get('occupied')
                contour = space.find('contour')
                xs = []
                ys = []
                for point in contour.findall('point'):
                    xs.append(point.get('x'))
                    ys.append(point.get('y'))
                assert len(xs) == 4
                assert len(ys) == 4
                (xs, ys) = preprocess_coords(xs, ys)
                xs = np.float32(xs)
                ys = np.float32(ys)
                coords = np.vstack((xs, ys)).T
                M = cv2.getPerspectiveTransform(coords, pts2)
                dst = cv2.warpPerspective(image, M, (40, 60))
                if space_occupied=="1":
                    cv2.imwrite(os.path.join(opath, '%s_%s.jpg' % (path.basename(name), space_id)), dst)
                else:
                    cv2.imwrite(os.path.join(epath, '%s_%s.jpg' % (path.basename(name), space_id)), dst)

			