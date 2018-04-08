import json
from multiprocessing import Queue
from pathlib import Path
from time import sleep

from keras.models import load_model

from src.labeler import PicLabeler

def run_predictor(model_file: Path, config_file: Path, images_queue: Queue, predictions_queue: Queue):
    # TODO check paths before

    model = load_model(str(model_file))

    with config_file.open() as file:
        config = json.load(file)

    labeler = PicLabeler(model, config)

    while True:
        #imageA, imageB = None, None
        imageA, imageB = images_queue.get()

        if imageA is None or imageB is None:
            sleep(1)
            continue

        result = labeler.run(imageA, imageB)
        predictions_queue.put(result)
