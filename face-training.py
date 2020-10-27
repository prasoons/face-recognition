# face-training.py/*

import os
import datetime
from PIL import Image
import numpy as np
import cv2
import pickle

# base directory identification
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# image directory identification
image_dir = os.path.join(BASE_DIR,'images')

# declaration of variables
current_id = 0
label_ids = {}
x_train = []
y_labels = []
start_time=0

# Loading of pre-trained classifiers. These can be found in opencv/data/haarcascades
#face_cascade = cv2.CascadeClassifier('lbpcascade_frontalface.xml')
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades+"haarcascade_frontalface_default.xml")
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades+"haarcascade_eye.xml")

# creation of Local Binary Pattern Histogram (LBPH) Face recognizer
recognizer = cv2.face.LBPHFaceRecognizer_create()
start_time = datetime.datetime.now()

# FOR loop to find out the all labelled directories
for root, dirs, files in os.walk(image_dir):
    for file in files:
        # checking images extension, should be .png or .jpg or .jpeg
        if file.lower().endswith('png') or file.lower().endswith('jpg') or file.lower().endswith('jpeg'):
            path = os.path.join(root,file)
            label = os.path.basename(os.path.dirname(path)).replace(' ','-').lower()
            #print(path, label)
            # assignment of  of labels and id
            if not label in label_ids:
                label_ids[label] = current_id
                current_id +=1
            id_ = label_ids[label]
         #   print(label_ids)
            # conversion in gray scale
            pil_image = Image.open(path).convert("L")  #grayscale
            size = (550,550)   # resize image for training
            final_image = pil_image.resize(size,Image.ANTIALIAS)
            # creation of image array
            image_array =  np.array(final_image,"uint8")
          #detection of face
            faces = face_cascade.detectMultiScale(image_array,scaleFactor=1.5,minNeighbors=10)
            for(x,y,w,h) in faces:
                roi = image_array[y:y+h,x:x+w]
                x_train.append(roi)
                y_labels.append(id_)

# checking time difference in training
end_time = datetime.datetime.now()
time_diff = (end_time - start_time)
execution_time = time_diff.total_seconds() * 1000

# storing all label id into pickle 'labels.pickle'
with open("labels.pickle",'wb') as f:
    pickle.dump(label_ids,f)

# training of all labels and saving into 'trainer.yml'
recognizer.train(x_train,np.array(y_labels))
recognizer.save("trainer.yml")