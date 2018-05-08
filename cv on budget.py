import cv2
import numpy as np
from keras.applications import imagenet_utils, VGG19

model = VGG19(weights='imagenet') # load weights

''' < 20 lines of code starts here '''

cam = cv2.VideoCapture(0) # Open webcam 

while True:
    ret, frame = cam.read()
    k = cv2.waitKey(1)

    if k%256 == 27 or ret == False:
        break
    
    frame_pred = cv2.resize(frame, (224, 224))
    frame_pred = cv2.cvtColor(frame_pred, cv2.COLOR_BGR2RGB).astype(np.float32)
    frame_pred = frame_pred.reshape((1, ) + frame_pred.shape)
    frame_pred = imagenet_utils.preprocess_input(frame_pred)
    predictions = model.predict(frame_pred)
    (label_id, label, score) = imagenet_utils.decode_predictions(predictions)[0][0]

    cv2.putText(frame, "%s with Probability: %.2f" % (label, score), (25, 25), cv2.FONT_HERSHEY_DUPLEX, 0.9, (0, 0, 250), 2)
    cv2.imshow('Computer Vision on a Budget', frame)

cam.release()
cv2.destroyAllWindows()