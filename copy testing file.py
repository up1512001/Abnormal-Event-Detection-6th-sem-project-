import tensorflow as tf
import cv2
import numpy as np
from keras.models import load_model
# import argparse
# from PIL import Image
import imutils as im

# regarding csv file writing
import pandas as pd
import csv
import datetime
import os


# mean square error
def mean_squared_loss(x1,x2):
    difference=x1-x2
    a,b,c,d,e=difference.shape
    n_samples=a*b*c*d*e
    sq_difference=difference**2
    Sum=sq_difference.sum()
    distance=np.sqrt(Sum)
    mean_distance=distance/n_samples
    return mean_distance

def help(video):

    # csv file handling
    fields = ['date','time','video source','abnormal or not']
    file_name = 'event detection details.csv'
    # df = pd.read_csv(file_name)
    # print(df.empty)

    if os.stat(file_name).st_size == 0:
        with open(file_name, 'w') as csvfile:
            csvwriter = csv.writer(csvfile)
            csvwriter.writerow(fields)

    model=load_model('saved_model 20-01-22.h5')
    cap = cv2.VideoCapture(video)
    print(video)
    # cap = cv2.VideoCapture(0)
    print(cap.isOpened())
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
        # print(loss)
        current_date = datetime.date.today()
        # print(current_date)
        current_time = datetime.datetime.now()
        current_time = current_time.strftime('%H:%M:%S')
        # print(current_time)
        if loss>0.00064:
            rows = [[f'{current_date}', f'{current_time}', f'{video}', 'Yes']]
            # regarding csv file writing
            with open(file_name, 'a') as csvfile:
                csvwriter = csv.writer(csvfile)
                # csvwriter.writerow(fields)
                csvwriter.writerows(rows)


            print('Abnormal Event Detected')
            cv2.putText(image,"Abnormal Event",(100,80),cv2.FONT_HERSHEY_SIMPLEX,2,(250,24,255),4)
        cv2.imshow("video",image)
    cap.release()
    cv2.destroyAllWindows()





if __name__ == '__main__':
    train_video_path = 'train/'
    videos = os.listdir(train_video_path)
    print(videos)
    for video in videos:
        if video.endswith('.mp4') or video.endswith('.avi'):
            help(train_video_path+video)