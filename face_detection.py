import cv2
import tensorflow as tf
import numpy as np
import tf_keras
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense, BatchNormalization
from tf_keras.models import model_from_json

#face dataset
face_cascade = cv2.CascadeClassifier('dataset/haarcascade_frontalface_default.xml')

#eye dataset
#eye_cascade = cv2.CascadeClassifier('dataset/haarcascade_eye.xml')

#initializes the camera, 0 is the webcam, the default, 1 would be any external camera
cap = cv2.VideoCapture(0)

#Build Mini CNN to recognize emotioins based off mini grif of face
def mini_CNN():
   model = Sequential()
   #Block 1, grayscale
   model.add(Conv2D(32, (3,3), activation='relu', input_shape=(48,48,1)))
   model.add(BatchNormalization())
   model.add(MaxPooling2D(pool_size=(2,2)))
   model.add(Dropout(0.25))

   #Block 2: Deepening the Features 
   model.add(Conv2D(64, (3,3), activation='relu'))
   model.add(BatchNormalization())
   model.add(MaxPooling2D(pool_size=(2,2)))
   model.add(Dropout(0.25))

   #Decision Layer: Flattening the final classification
   model.add(Flatten())
   model.add(Dense(128, activation='relu'))
   model.add(BatchNormalization())
   model.add(Dropout(0.5))
   #Output
   model.add(Dense(7, activation='softmax'))

   return model 
def overlay_img(frame,overlay, x,y):
    if overlay is None: return frame
    overlay = cv2.resize(overlay, (150,150))
    h,w = overlay.shape[:2]
    if y+h < frame.shape[0] and x+w < frame.shape[1]:
        frame[y:y+h, x:x+w] = overlay
    return frame

json_file = open('dataset/model.json','r')
loaded_model_json = json_file.read()
json_file.close()

model = model_from_json(loaded_model_json)

model.load_weights('dataset/weights.h5')


emotion_dict = {0:"Angry", 1:"Disgusted", 2:"Fearful", 3:"YAY!! You want to join AI Club", 4:"Sad", 5:"Suprised", 6:"Neutral"}

frames = 0
last_emo = "Detecting..."
#AI Club logo pre-loaded
ai_club = cv2.imread('photos/logo.png')
maxindex = -1
while 1:
    #ret is whether the frame is being read(True/False) / ldie
    ret, img = cap.read()

    #Saftey Check to see if frame exists before processing
    if not ret or img is None:
       print("Waiting for camera...")
       continue
    frames += 1
    #converts the image to greyscale
    #Needs to be in greyscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    #Detects faces of different sizes in the input image
    #Slices a window, to find faces based of of (1.3, shrink size)/ 5(voting system to detct face)
    faces = face_cascade.detectMultiScale(gray,1.1,6)
    #Outputs NumPy Array that has (x,y,w,h) of each face 
 
    

    for (x,y,w,h) in faces:
        #Drawing a rectangle, (255,255,0) = rgb color / 2 =thickness
       
        cv2.rectangle(img,(x,y),(x+w, y+h),(255,255,0),2)
        #Region of Interest, creates a frame of just the face to look for the eyes
        roi_gray = gray[y:y+h, x:x+w]
        roi_color= img[y:y+h, x:x+w]
        if frames % 5 == 0:
            #Standardize lighting/contrast
            roi_gray = cv2.equalizeHist(roi_gray)

            #Resizing the frame to 48x48
            cropped_img = cv2.resize(roi_gray, (48,48))
            #Normalize the pixels (0-255) -> (0,1), model expects (1,48,48,1)
            image_pixels = np.expand_dims(np.expand_dims(cropped_img,-1), 0) #/ 255.0
            #expand_dims add the (48,48) -> (48,48,1) with grayscale channel -> (1,48,48,1) with the batch size

            #Perdiction, verbose just slows down image processing
            prediction = model.predict(image_pixels, verbose=0)
            print(f"before argmax ={prediction }")
            maxindex = int(np.argmax(prediction)) #finds the highest probability
            print(f"emotion # = {maxindex} and emotion = {emotion_dict[maxindex]}")
            last_emo = emotion_dict[maxindex]
            
        #Displaying text based off findings
        if maxindex ==3:
            img = overlay_img(img, ai_club,x-150,y-85)
            
        cv2.putText(img, last_emo, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255), 2)


        #detect eye of different sizes
        #eyes = eye_cascade.detectMultiScale(roi_gray)
        #Detect image in gray, display in color. thats why we make two
        #for (ex, ey,ew,eh) in eyes:
         #cv2.rectangle(roi_color ,(ex,ey),(ex+ew, ey+eh),(0,127,255),2)
    cv2.imshow('AI Club Emotion Detector',img)

    #uses hexadecial, because 0xff = only returns last 8 bits of the operation
    k = cv2.waitKey(30) & 0xff
    if frames > 1000:
       frames = 0

    if k == ord('q'):
        break
#closes the window
cap.release()
#De-Allocate any associated memory usage

