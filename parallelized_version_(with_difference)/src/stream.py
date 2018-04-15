import json

from multiprocessing import Queue
from pathlib import Path

import cv2
import numpy as np


RED_COLOR = (0, 0, 255)
GREEN_COLOR = (0, 255, 0)
FONT = cv2.FONT_HERSHEY_SIMPLEX

def read_image(capture):
    ok, image = capture.read()
    if not ok:
        return None

    return image


def run_stream(video_path: Path, config_file: Path, images_queue: Queue, predictions_queue: Queue):

    with config_file.open() as file:
        config = json.load(file)

    capture = cv2.VideoCapture(str(video_path))

    image_prev = read_image(capture)
    image_cur = read_image(capture)

    images_queue.put((image_prev, image_cur))

    prediction = dict(zip(range(1, len(config)+1), ["Occupied"]*len(config)))

    while capture.isOpened():

        # Capture frame-by-frame
        image_cur = read_image(capture)
        if image_cur is None:
            break


        if images_queue.empty():
            try:
                images_queue.put_nowait((image_prev, image_cur))
            except:
                pass
        else:
            try:
                images_queue.get_nowait()
                images_queue.put_nowait((image_prev, image_cur))
            except:
                pass

        if not predictions_queue.empty():
            prediction_update = predictions_queue.get()

            prediction = {**prediction, **prediction_update}
            image_prev = image_cur

        for index, status in prediction.items():

            color = RED_COLOR if status == 'Occupied' else GREEN_COLOR
            coordinates = np.array(config[int(index) - 1], np.int32).reshape((-1, 1, 2))

            cv2.polylines(image_cur, [coordinates], True, color)
            cv2.putText(image_cur,
                        str(index),
                        tuple(coordinates[0][0]),
                        FONT,
                        0.4,
                        color,
                        1, cv2.LINE_AA)

        cv2.imshow('Movie', image_cur)
        cv2.waitKey(1)


    capture.release()
    cv2.destroyAllWindows()
