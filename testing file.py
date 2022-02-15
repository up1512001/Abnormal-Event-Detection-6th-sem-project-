import tensorflow as tf
import cv2
import numpy as np
from keras.models import load_model
# import argparse
# from PIL import Image
import imutils as im
def mean_squared_loss(x1,x2):
    difference=x1-x2
    a,b,c,d,e=difference.shape
    n_samples=a*b*c*d*e
    sq_difference=difference**2
    Sum=sq_difference.sum()
    distance=np.sqrt(Sum)
    mean_distance=distance/n_samples
    return mean_distance
model=load_model('saved_model 20-01-22.h5')
cap = cv2.VideoCapture('train/24.mp4')
# cap = cv2.VideoCapture(0)
# print(cap.isOpened())
while cap.isOpened():
    imagedump=[]
    ret,frame=cap.read()
    for i in range(10):
        ret,frame=cap.read()
        if ret == False:
            break
        image = im.resize(frame, width=1000, height=1000, inter=cv2.INTER_AREA)
        frame = cv2.resize(frame, (227, 227), interpolation=cv2.INTER_AREA)
        gray = 0.2989 * frame[:, :, 0] + 0.5870 * frame[:, :, 1] + 0.1140 * frame[:, :, 2]
        gray = (gray - gray.mean()) / gray.std()
        gray = np.clip(gray, 0, 1)
        imagedump.append(gray)
    imagedump=np.array(imagedump)
    imagedump.resize(227,227,10)
    imagedump=np.expand_dims(imagedump,axis=0)
    imagedump=np.expand_dims(imagedump,axis=4)
    output=model.predict(imagedump)
    loss=mean_squared_loss(imagedump,output)
    if ret == False:
        print("video end")
        break
    if cv2.waitKey(10) & 0xFF==ord('q'):
        break
    print(loss)
    if loss>0.00064:
        print('Abnormal Event Detected')
        cv2.putText(image,"Abnormal Event",(100,80),cv2.FONT_HERSHEY_SIMPLEX,2,(250,24,255),4)
    cv2.imshow("video",image)
cap.release()
cv2.destroyAllWindows()