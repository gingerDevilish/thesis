from skimage.measure import compare_ssim
from skimage.draw import polygon
from scipy.spatial import distance
import numpy as np
import imutils
import cv2


class PicLabeler:
    def __init__(self, model, config):
        self.model = model
        self.slots = config

    def blurAndGray(self):
        self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        self.changes = cv2.cvtColor(self.changes, cv2.COLOR_BGR2GRAY)

        self.image_copy = cv2.medianBlur(self.image, 7)
        self.changes_copy = cv2.medianBlur(self.changes, 7)
        self.image_copy = cv2.GaussianBlur(self.image, (13, 13), 0)
        self.changes_copy = cv2.GaussianBlur(self.changes, (13, 13), 0)

    def findSSIM(self):
        (_, diff) = compare_ssim(self.image_copy,
                                 self.changes_copy,
                                 gaussian_weights=True,
                                 full=True)
        diff = (diff * 255).astype("uint8")

        thresh = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
        self.cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        self.cnts = self.cnts[0] if imutils.is_cv2() else self.cnts[1]
        self.canvasM = np.zeros((self.height + 1, self.width + 1), dtype=np.uint8)

        self.rect = []
        # loop over the contours
        for c in self.cnts:
            # compute the bounding box of the contour and then draw the
            # bounding box on both input images to represent where the two
            # images differ
            (x, y, w, h) = cv2.boundingRect(c)
            if w > 20 and h > 20:
                cv2.rectangle(self.canvasM, (x, y), (x + w, y + h), 255, -1)
                self.rect.append(((x, y), (x + w, y + h)))

        self.mergeArea()

    def mergeArea(self):

        self.cer = []
        self.cnt = cv2.findContours(self.canvasM, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for a in self.cnt[1]:
            (x, y, w, h) = cv2.boundingRect(a)
            self.cer.append(((x, y), (x + w, y + h)))
            cv2.rectangle(self.canvasM, (x, y), (x + w, y + h), 255, -1)
            # cv2.rectangle(imageA, (x, y), (x + w, y + h), (0, 0, 255), 2)
            # cv2.rectangle(imageB, (x, y), (x + w, y + h), (0, 0, 255), 2)

    def run(self, image, changes):

        self.image = image
        self.changes = changes
        self.blurAndGray()

        self.height, self.width = self.image.shape

        self.findSSIM()
        self.pts2 = np.float32([[0, 60], [0, 0], [40, 0], [40, 60]])
        slots = []  # list of preprocessed slot images
        ids = []  # list of slot ids
        coord = []  # list of slot coordinates, do we really need this?
        for index, space in enumerate(self.slots):
            for change in self.cer:
                points = [list(change[0]),
                          [change[1][0], change[0][1]],
                          list(change[1]),
                          [change[0][0], change[1][1]]]
                if self.iou(np.asarray(space), np.asarray(points)) > 0.03:
                    slot, icoord = self.process_slot(space)
                    ids.append(index + 1)  # it's better to put index here and index+1 outside
                    slots.append(slot)
                    coord.append(icoord)
        #           # print("Slot  %d was changed" % (index+1))
                    break

        #for index, space in enumerate(self.slots):
        #    slot, icoord = self.process_slot(space)
        #    ids.append(index + 1)  # it's better to put index here and index+1 outside
        #    slots.append(slot)

        return self.predict(slots, ids)

    def iou(self, fig1, fig2):
        max_x = max(fig1[:, 0].tolist() + fig2[:, 0].tolist())
        max_y = max(fig1[:, 1].tolist() + fig2[:, 1].tolist())
        canvas = np.zeros((max_x + 1, max_y + 1), dtype=np.uint8)
        shape1 = np.copy(canvas)
        shape1[polygon(fig1[:, 0],
                       fig1[:, 1])] = 1
        shape2 = np.copy(canvas)
        shape2[polygon(fig2[:, 0], fig2[:, 1])] = 1  # same as in previous case
        intersect = cv2.countNonZero(cv2.bitwise_and(shape1, shape2))
        union = cv2.countNonZero(cv2.bitwise_or(shape1, shape2))
        iou = 0
        # assert union!=0
        if intersect > 0 and union > 0:
            iou = float(intersect) / union
        return iou

    def preprocess_coords(self, xs, ys):
        distances = []
        # calculate all side lengths of the quadrilateral
        for i in range(4):
            distances.append(
                distance.euclidean(np.float32([xs[i], ys[i]]),
                                   np.float32([xs[(i + 1) % 4], ys[(i + 1) % 4]])))
        # which one is the longest?
        starting_point = np.argmax(np.array(distances))
        # rearrange coordinates cyclically, so that longest side goes first
        new_xs = xs[starting_point:] + xs[:starting_point]
        new_ys = ys[starting_point:] + ys[:starting_point]
        return new_xs, new_ys

    def predict(self, slots, ids):
        answer = {}
        if not slots:
            print("answer empty")
            return answer
        pred = self.model.predict(np.array(slots), 16, 1)

        # construct a JSON entity with results
        pred = pred.ravel().tolist()
        for i, one_id in enumerate(ids):
            answer[one_id] = 'Occupied' if pred[i] else 'Empty'
        return answer

    def process_slot(self, space):
        xs = []
        ys = []
        for point in space:
            xs.append(point[0])
            ys.append(point[1])
        # ensure contour is a quadrilateral. This assertion failed once.
        assert len(xs) == 4
        assert len(ys) == 4
        # preprocess and save coordinates
        xs, ys = self.preprocess_coords(xs, ys)
        xs = np.float32(xs)
        ys = np.float32(ys)
        coords = np.vstack((xs, ys)).T
        # get a matrix for perspective transformation
        M = cv2.getPerspectiveTransform(coords, self.pts2)
        # transform a quadrilateral into a solid rectangle
        dst = cv2.warpPerspective(self.image, M, (40, 60))
        return np.reshape(dst, (40, 60, 1)), coords
