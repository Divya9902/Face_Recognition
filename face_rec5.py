import face_recognition
import imutils
import pickle
import time
import cv2
import os
 
#cascade function is trained pretrained haar cascade model to detect faces and eyes in images
cascPathface = os.path.dirname(
 cv2.__file__) + "/data/haarcascade_frontalface_alt2.xml"

faceCascade = cv2.CascadeClassifier(cascPathface)

data = pickle.loads(open('face_enc', "rb").read())
image = cv2.imread('akshay.jpeg')
rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
faces = faceCascade.detectMultiScale(gray,
                                     scaleFactor=1.1,
                                     minNeighbors=5,
                                     minSize=(60, 60),
                                     flags=cv2.CASCADE_SCALE_IMAGE)
#returns a list of 128-d face encoding
encodings = face_recognition.face_encodings(rgb)
names = []
for encoding in encodings:
    #returns a list of true and false values indicating the known face encodings
    matches = face_recognition.compare_faces(data["encodings"],
    encoding)
    #compares eucledian distance
    name = "Unknown"
    
    if True in matches:
        
        matchedIdxs = [i for (i, b) in enumerate(matches) if b]
        counts = {}
        
        for i in matchedIdxs:
            #Check the names at respective indexes we stored in matchedIdxs
            name = data["names"][i]
            #increase count for the name we got
            counts[name] = counts.get(name, 0) + 1
            #set name which has highest count
            name = max(counts, key=counts.get)
 
 

        names.append(name)

        for ((x, y, w, h), name) in zip(faces, names):
         
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), int(0.05*x))
            print(x)
            if(x>1000):
                m=int(0.005*x)
            elif(x>100):
                m = int(0.02*x)
               
            else:
                m=1
            
            cv2.putText(image, name, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
            m, (0, 255, 0), m)
    cv2.imshow("Frame", image)
    cv2.waitKey(0)