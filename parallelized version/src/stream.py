import cv2
import json
import numpy as np

from multiprocessing import Queue
from pathlib import Path

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

    imageA = read_image(capture)
    imageB = read_image(capture)

    images_queue.put((imageA, imageB))

    start_predictions = Path("old.json")

    with start_predictions.open() as file:
        prediction = json.load(file)

    prediction = {int(k): v for (k, v) in prediction.items()}

    while(capture.isOpened()):

        # Capture frame-by-frame
        imageB = read_image(capture)
        if imageB is None:
            break;


        if images_queue.empty():
            try:
                images_queue.put_nowait((imageA, imageB))
            except:
                pass
        else:
            try:
                images_queue.get_nowait()
                images_queue.put_nowait((imageA, imageB))
            except:
                pass

        if not predictions_queue.empty():
            prediction_update = predictions_queue.get()

            prediction = {**prediction, **prediction_update}
            imageA =  imageB

        for index, status in prediction.items():

            color = RED_COLOR if status == 'Occupied' else GREEN_COLOR
            coordinates = np.array(config[int(index) - 1], np.int32).reshape((-1, 1, 2))

            cv2.polylines(imageB, [coordinates], True, color)
            cv2.putText(imageB, str(index), tuple(coordinates[0][0]), FONT, 0.4, color, 1, cv2.LINE_AA)

        cv2.imshow('Movie', imageB)
        cv2.waitKey(1)


    capture.release()
    cv2.destroyAllWindows()
