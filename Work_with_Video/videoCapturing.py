import cv2
import json
import datetime
import numpy as np
from labeler import PicLabeler
from keras.models import load_model


def workHard(pathToMovie, initpic, configPath, initanswer):
    cap = cv2.VideoCapture(pathToMovie)
    
    imageA = cv2.imread(initpic)
    with open(configPath, 'r') as cp:
        config  = json.load(cp)
    with open(initanswer, 'r') as ia:
        old_answer = json.load(ia)

    model = load_model("model")
    
    labeler = PicLabeler(model, config)

    frameRate = cap.get(5)
    #print(frameRate)
    while(cap.isOpened()):

        frameId = cap.get(1)

        # Capture frame-by-frame
        ret, imageB = cap.read()
        if not ret:
            print("No frames. Nothing to show")
            break

        if (int(frameId) % int(frameRate)):
            continue
            
        # Our operations on the frame come here
        answer = labeler.run(imageA, imageB)
        #if not len(answer):
        #    continue
        new_answer = {}
        font = cv2.FONT_HERSHEY_SIMPLEX

        #rebuild answer:
        #new answer is: answer[i] if existent, old_answer[i] otherwise
        #for each answer, draw a labeled rectangle using coords
        for k, v in old_answer.items():
            new_answer[int(k)] = answer.get(int(k), v)
            pts = np.array(config[int(k)-1], np.int32).reshape((-1, 1, 2))
            #color = (0, 255, 0) if old_answer[k]=='Occupied' else (255, 0, 0)
            #cv2.polylines(imageA, [pts], True, color)
            #cv2.putText(imageA, str(k), tuple(pts[0][0]), font, 0.4, color, 1, cv2.LINE_AA)
            color = (0, 0, 255) if new_answer[int(k)]=='Occupied' else (0, 255, 0)
            cv2.polylines(imageB, [pts], True, color)
            cv2.putText(imageB, str(k), tuple(pts[0][0]), font, 0.4, color, 1, cv2.LINE_AA)
        old_answer=new_answer
        imageA=imageB
        cv2.imshow('Movie', imageB)
        cv2.waitKey(1)
        
        dateFormat="%Y-%m-%d %H:%M:%S"
        
        with open('log/%s.json'%(datetime.datetime.now().strftime(dateFormat)), 'w') as f:
            f.write(json.dumps(new_answer))
        #print("iteration")
    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()
