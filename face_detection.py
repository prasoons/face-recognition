# face_detection.py/*

import cv2, time
import numpy as np
import os, platform
import pickle
from PIL import Image

# base directory identification
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# image directory identification
image_dir = os.path.join(BASE_DIR,'images')
# cascade and cascade data directory identification
cascades_dir = os.path.join(BASE_DIR,'cascades')
cascades_dir_data = os.path.join(cascades_dir,'data')

# function to drow boundary over face and eyes DRAW_BOUNDARY
def draw_boundary(frameImg,faceclassifer,eyesClassifer,scaleFactor,minNeighbour,color,text,count):
    global name
    global cnt
    gray_img = cv2.cvtColor(frameImg, cv2.COLOR_BGR2GRAY)
    faces = faceclassifer.detectMultiScale(gray_img,scaleFactor,minNeighbour)
    cords = []
    index = count   # to take care of image count for unknown face
    folder =1

    # for loop to detect face and eyes within the region of interest
    for(x,y,w,h) in faces:
        # Detect face within the region of interest and Draw a blue box around face.
        cv2.rectangle(frameImg,(x,y),(x+w,y+h),color,1) # drawing of rectangle
        roi_gray = gray_img[y:y + h, x:x + w]  #reason of interest
        roi_color = frame[y:y + h, x:x + w]
        # getting confidence level
        id_, conf = recognizer.predict(roi_gray)  #
        # define font and colors
        font = cv2.FONT_HERSHEY_SIMPLEX;
        color = (255, 255, 0)
        storke = 1
        # assignment of label id
        name = labels[id_]

        # If loop to check the confidence level with in a range from 55 to 93
        if conf >= 55 and conf<=93:
            #putting label and confidence value at top and bottom of rectangle
            cv2.putText(frameImg, name, (x, y), font, 1, color, storke, cv2.LINE_AA)
            cv2.putText(frameImg, str(conf), (x+w, y+h), font, 1, (255,0,255), storke, cv2.LINE_AA)

            # For each face, detect eyes within the region of interest.
            eyes = eyesClassifer.detectMultiScale(roi_gray)
            for (ex, ey, ew, eh) in eyes:
                # Draw a green box around each eye
                cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)
        else:
            # putting 'New Face' label for unknown face
            cv2.putText(frameImg, text, (x, y), font, 1, color, storke, cv2.LINE_AA)
            # For each face, detect eyes within the region of interest.
            eyes = eyesClassifer.detectMultiScale(roi_gray)
            for (ex, ey, ew, eh) in eyes:
                cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)
            name = "New Face"
            # calling of saveImage function to save the unknown face images
            saveImage(frameImg,index)

    return cords,frameImg

# detect function to call draw_boundary function
def detect(frameImg, faceClassifer,eyesClassifer,count):
    color = {"blue":(255,0,0),"green":(0,0,255),"red":(0,255,0)}
    crods,img = draw_boundary(frameImg,faceClassifer,eyesClassifer,1.5,10,color['blue'],"New Face",count)
    return img

# function to save 30 images of unknown face
def saveImage(imageSave,index):
    imgDir = image_dir
    folderName = '\\newface'
    dirname = imgDir + folderName
    # checking of existence of 'newface' directory else create a new directory as 'newface'
    if not os.path.exists(dirname):
        os.mkdir(dirname)
        #print('directory creation')
    else:
        img_name = str(index) + '.png'  # name of image start with index value
        # saving of images in 'newface'
        cv2.imwrite(os.path.join(dirname, img_name), imageSave)
        #print(index)


# Launching of face detection program

# Loading of pre-trained classifiers. These can be found in opencv/data/haarcascades

#face_cascade = cv2.CascadeClassifier('lbpcascade_frontalface.xml')
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades+"haarcascade_frontalface_default.xml")
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades+"haarcascade_eye.xml")
#smile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades+"haarcascade_smile.xml")

# creation of Local Binary Pattern Histogram (LBPH) Face recognizer
recognizer = cv2.face.LBPHFaceRecognizer_create()
# reading of trained file
recognizer.read("trainer.yml")


name ="New Face"
cnt = 1
labels = {}
# reading ids and labels from 'labels.pickle', created during training of data
with open("labels.pickle",'rb') as f:
    org_labels = pickle.load(f)
    labels = {v:k for k,v in org_labels.items()}

# launching of live camera, opening of webcam device
cap = cv2.VideoCapture(0)
imgCount = 1

#reading of frames from live webcam in while loop
while True:
    check, frame = cap.read()  # Read an frame from the webcam.
    # calling of detect function
    frame = detect(frame,face_cascade,eye_cascade,imgCount)
    cv2.imshow('img',frame)
    if name == "New Face":
        imgCount = imgCount + 1

    # Close the script when q is pressed.
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    elif (imgCount > 30 and name == "New Face"):
        print((imgCount > 30 and name == "New Face"))
        break

# release all windows
cap.release()
cv2.destroyAllWindows()

