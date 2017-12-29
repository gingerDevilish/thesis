import cv2
import numpy as np
import sys
import os.path as path
import json
from scipy.spatial import distance
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.models import model_from_json
from skimage.measure import compare_ssim
import argparse
import imutils

class PicLabeler:
    def __init__(self, image, config, changes):
        self.image = image
        self.slots = config
        self.changes = changes
        with open('model.json', 'r') as f:
            self.model = model_from_json(f.read())
        self.model.load_weights('first_try.h5')
        self.mask = np.zeros(len(self.slots))
        
    def run(self):
        self.pts2 = np.float32([[0,60],[0,0],[40,0],[40,60]])
        slots = [] # list of preprocessed slot images
        ids = [] # list of slot ids
        coord=[] # list of slot coordinates
        for index, space in enumerate(self.slots):
            ids.append(index+1)
            slot, icoord = self.process_slot(space)
            slots.append(slot)
            coord.append(icoord)
        
        # TODO
        # for each parking space (enumerate as we need index) (coord)
        # for each change
        # calculate IoU (need IoU function)
        # IoU needs two arrays of coordinates
        # if IoU > 0.1 assign mask value 1 and skip further

        #retrieve only slots and ids that have mask value 1.

        return self.predict(slots, ids)
        # !!! ensure outside: answer saved to JSON, labeled image saved
            # draw bounding quadrilaterals 
            #pts = np.array(coord[i], np.int32).reshape((-1, 1, 2))
            #color = (0, 255, 0) if pred[i] else (255, 0, 0)
            #cv2.polylines(image,[pts],True,color)
         
        ##SAVE_RESULT
        ## save answer to JSON    
        #with open('%s.json' % (name_stub), 'w') as f:
        #    f.write(json.dumps(answer))
        ## save labeled image        
        #cv2.imwrite('%s_marked.jpg'%(name_stub), image)

    def iou(self, fig1, fig2):
        max_x = max(fig1[:,0].tolist()+fig2[:,0].tolist())
        max_y = max(fig1[:,1].tolist()+fig2[:,1].tolist())
        canvas = np.zeros((max_x+1, max_y+1), dtype=np.uint8)
        shape1 = np.copy(canvas)
        cv2.fillConvexPoly(shape1, fig1, 255)
        shape2 = np.copy(canvas)
        cv2.fillConvexPoly(shape2, fig2, 255)
        intersect = cv2.bitwise_and(shape1, shape2)
        union = cv2.bitwise_or(shape1, shape2)
        iou = float(cv2.countNonZero(intersect))/cv.countNonZero(union)
        return iou
        
    def preprocess_coords(self, xs, ys):
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
        
    def predict(self, slots, ids):
        pred = self.model.predict(np.array(slots), 16, 1)
        # a dictionary for results of prediction
        answer = {}
          
        # construct a JSON entity with results
        pred = pred.ravel().tolist()
        for i in range(len(ids)):
            answer[ids[i]]='Occupied' if pred[i] else 'Empty'
        return answer
        
    def process_slot(self, space):
        xs = []
        ys = []
        for point in space:
            xs.append(point[0])
            ys.append(point[1])
        # ensure contour is a quadrilateral. This assertion failed once.
        assert len(xs)==4
        assert len(ys)==4
        # preprocess and save coordinates
        xs, ys = self.preprocess_coords(xs, ys)
        xs = np.float32(xs)
        ys = np.float32(ys)
        coords=np.vstack((xs, ys)).T
        # get a matrix for perspective transformation 
        M = cv2.getPerspectiveTransform(coords, self.pts2)
        # transform a quadrilateral into a solid rectangle
        dst = cv2.warpPerspective(self.image, M, (40,60))
        return np.reshape(dst, (40, 60, 1)), coords


## load the two input images
imageA = cv2.imread('2.jpg', 0)
imageB = cv2.imread('4.jpg', 0)
config=json.load(open('conf.json'))
old_answer=json.load(open('old.json'))

# convert the images to grayscale
grayA=cv2.medianBlur(imageA,5)
grayB=cv2.medianBlur(imageB,5)
grayA = cv2.GaussianBlur(grayA,(7,7),0)
grayB = cv2.GaussianBlur(grayB,(7,7),0)

rect=[]
# compute the Structural Similarity Index (SSIM) between the two
# images, ensuring that the difference image is returned
(score, diff) = compare_ssim(grayA, grayB, gaussian_weights=True, full=True)
diff = (diff * 255).astype("uint8")
print("SSIM: {}".format(score))

# threshold the difference image, followed by finding contours to
# obtain the regions of the two input images that differ
thresh = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = cnts[0] if imutils.is_cv2() else cnts[1]

# loop over the contours
for c in cnts:
	# compute the bounding box of the contour and then draw the
	# bounding box on both input images to represent where the two
	# images differ
	(x, y, w, h) = cv2.boundingRect(c)
	if w>10 and h>10:
	    rect.append(((x, y),(x + w, y + h)))
	    #if you want see difference between two images uncomment two lines below
	    #cv2.rectangle(imageA, (x, y), (x + w, y + h), (0, 0, 255), 2)
	    #cv2.rectangle(imageB, (x, y), (x + w, y + h), (0, 0, 255), 2)

# show the output images
#cv2.imwrite("Original.jpg", imageA)
#cv2.imwrite("Modified.jpg", imageB)
#cv2.imwrite("Diff.jpg", diff)
#cv2.imwrite("Thresh.jpg", thresh)
#with open('diff_coords.json', 'w') as f:
#    f.write(json.dumps(rect))
    
labeler = PicLabeler(imageB, config, rect)
answer = labeler.run()
new_answer = {}
font = cv2.FONT_HERSHEY_SIMPLEX

#rebuild answer:
#new answer is: answer[i] if existent, old_answer[i] otherwise
#for each answer, draw a labeled rectangle using coords
for k, v in old_answer.items():
    new_answer[int(k)] = answer.get(int(k), v)
    pts = np.array(config[int(k)-1], np.int32).reshape((-1, 1, 2))
    color = (0, 255, 0) if new_answer[int(k)]=='Occupied' else (255, 0, 0)
    cv2.polylines(imageB, [pts], True, color)
    cv2.putText(imageB, k, tuple(pts[0][0]), font, 0.4, color, 1, cv2.LINE_AA)
    
with open('new_pred.json', 'w') as f:
    f.write(json.dumps(new_answer))
cv2.imwrite('imageB_marked.jpg', imageB)
