from keras.models import load_model  # TensorFlow is required for Keras to work
import cv2  # Install opencv-python
import numpy as np
#vid=cv2.VideoCapture("http://192.168.1.215:8080/video") for ip camera
vid=cv2.VideoCapture("group.mp4")
facemodel=cv2.CascadeClassifier("face.xml") 
maskmodel = load_model("mask.h5", compile=False) # Load the model
i=1
while True:
    flag,frame=vid.read()
    if flag:
        faces=facemodel.detectMultiScale(frame)
        for (x,y,l,w) in faces:
            face_img=frame[y:y+w,x:x+l] #cropping the face
            face_img = cv2.resize(face_img, (224, 224), interpolation=cv2.INTER_AREA) #resizing the face
            face_img = np.asarray(face_img, dtype=np.float32).reshape(1, 224, 224, 3) #converting image in numpy array float and reshaping it
            face_img = (face_img / 127.5) - 1 #Normalize the image array
            pred=maskmodel.predict(face_img)[0][0]
            if(pred>0.9):
                path="data/"+str(i)+".png"
                i=i+1
                cv2.imwrite(path,frame[y:y+w,x:x+l])
                cv2.rectangle(frame,(x,y),(x+l,y+w),(0,0,255),3)
            else:
                cv2.rectangle(frame,(x,y),(x+l,y+w),(0,255,0),3)
                
        cv2.namedWindow("vaishvi",cv2.WINDOW_NORMAL)
        cv2.imshow("vaishvi",frame)
        k=cv2.waitKey(30) # unicode of x
        if(k==ord("x")):
            break
    else:
        break
vid.release()
cv2.destroyAllWindows() 
