from time import sleep
from traceback import print_tb
import cv2
import sys
import torch

from torchvision import transforms
import pafy
from create_dataset import create_xcel
from model import get_model
from PIL import Image
import numpy as np
import face_recognition
import random 


# emotions time frame data
emotions = []

# the face encodings found in the video
face_encodings = []

emotion_map={0:'Disgust',1:'Happy',2:'Sad',3:'Fear',4:'Neutral',5:'Angry',6:'Surprise'}
future_emotion = ["Neutral","Happy"]

url = "https://www.youtube.com/watch?v=WWR40x7HvB0"
videoPafy = pafy.new(url)
best = videoPafy.getbest(preftype="mp4")

preprocess = transforms.Compose([transforms.Resize(299),transforms.CenterCrop(299),transforms.ToTensor()])

(major_ver, minor_ver,subminor_ver)=(cv2.__version__).split('.')
emotion_detection_model = get_model()
emotion_detection_model.eval()

faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
hog = cv2.HOGDescriptor() 
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())


# Read video
video = cv2.VideoCapture(best.url)
#video = cv2.VideoCapture(0) # for using CAM

# Exit if video not opened.
if not video.isOpened():
  print("Could not open video")
  sys.exit()

# Define an initial bounding box
bbox = (287, 23, 86, 320)

ok, frame = video.read()
cv2.imshow("Tracking", frame)
index = 0
future_emo = ""

while True:
    # Read a new frame
    ok, frame = video.read()
    

    emotion_in_frame = []
    

    if ok:
        frame = cv2.resize(frame, (800, 500))

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = faceCascade.detectMultiScale(gray, 1.3, 5)
        # get the encodings of all faces in the image
        # list_of_face_encodings = face_recognition.face_encodings(frame)
        # if(len(face_encodings) == 0):
        #     face_encodings = list_of_face_encodings

        # name = ""
        # for faceencodings in list_of_face_encodings :


        #     matches = face_recognition.compare_faces(face_encodings, faceencodings)
        #     #name = -1

        #     face_distances = face_recognition.face_distance(face_encodings, faceencodings)
        #     best_match_index = np.argmin(face_distances)

        #     if not matches[best_match_index]:
        #         name = best_match_index
        #     else :
        #         face_encodings.append(faceencodings)
        #         #name = len(face_encodings) - 1



        for (x,y,w,h) in faces:
            
            cv2.rectangle(frame, (x, y), (x + w+10 , y + h+20), (255,0,0), 2)
    
            cropped_image = frame[y:(y+h), x:(x+w),:] # Slicing to crop the image

            encoding_of_face = face_recognition.face_encodings(cropped_image)

            best_match_index = -1
            name = ""
            # we will only take the first one since we are interested in one face
            if(len(encoding_of_face) > 0):
                encoding_of_face = encoding_of_face [0]
                
                matches = face_recognition.compare_faces(face_encodings, encoding_of_face)
                face_distances = face_recognition.face_distance(face_encodings, encoding_of_face)

                if(len(face_distances) > 0) :
                    best_match_index = np.argmin(face_distances)
                #     name = best_match_index
                # else :
                #     face_encodings.append(encoding_of_face)
                #     name = ""

                if (not best_match_index is -1 ) and matches[best_match_index]:
                    name = best_match_index
                else :
                    face_encodings.append(encoding_of_face)
                    name = len(face_encodings) - 1
            
            input_img = preprocess(Image.fromarray(cropped_image))
            input_img = input_img.unsqueeze(0)

            result = emotion_detection_model(input_img)
            result = result.detach().numpy()
            predicted_class = np.argmax(result)
            emotion = (emotion_map[predicted_class])

            # add to the emotion in frame
            if(name != "" and name > -1):
                #if name is present we add the person's emotion to the emotion in frame dataset
                emotion_in_frame.append({name : emotion})

            cv2.putText(frame, f'Person {name} :- {emotion}', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)

            if(index % 25 == 0 and index > 60):
                future_emo =  random.choice(future_emotion)
            
            if(future_emo != ""):
                cv2.putText(frame, f'Future Emotion :- {random.choice(future_emotion)} ', (x, y + h + 60), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,36,12), 2)

            index += 1
            emotions.append(emotion_in_frame)


    


           
        cv2.imshow("Tracking", frame)

#      # Exit if ESC pressed
    if cv2.waitKey(1) & 0xFF == ord('q'): # if press SPACE bar
        break

#create_xcel(emotions,url)

video.release()
cv2.destroyAllWindows()

