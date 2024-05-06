from keras.models import load_model  # TensorFlow is required for Keras to work
import cv2  # Install opencv-python
import numpy as np
import streamlit as st
st.set_page_config(page_title="Mask Detection System", page_icon="https://cdn-icons-png.flaticon.com/512/3120/3120350.png")
st.title("FACE MASK DETECTION SYSTEM")
choice=st.sidebar.selectbox("Menu", ("HOME", "IP CAMERA", "CAMERA", "SHOW FACES"))
if "i" not in st.session_state:
    st.session_state["i"]=1
if(choice=="HOME"):
    st.image("https://media.licdn.com/dms/image/C4D12AQHLRlFL5xUPxA/article-cover_image-shrink_600_2000/0/1601195309873?e=2147483647&v=beta&t=C4EBGtyioccsfoWu6BBBq5bPNe4O58Ureb-gblPFv6E")
    st.text("This project is developed by Vaishvi Gupta.")
elif(choice=="IP CAMERA"):
    url=st.text_input("Enter IP CAMERA URL")
    window=st.empty()
    btn=st.button("Start Detection")
    if btn:
        vid=cv2.VideoCapture(url)
        facemodel=cv2.CascadeClassifier("face.xml") 
        maskmodel = load_model("mask.h5", compile=False) # Load the model
        
        btn2=st.button("Stop Detection")
        if btn2:
            vid.release()
            st.experimental_rerun()
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
                        path="data/"+str(st.session_state["i"])+".png"
                        st.session_state["i"]=st.session_state["i"]+1
                        cv2.imwrite(path,frame[y:y+w,x:x+l])
                        cv2.rectangle(frame,(x,y),(x+l,y+w),(0,0,255),3)
                    else:
                        cv2.rectangle(frame,(x,y),(x+l,y+w),(0,255,0),3)
                window.image(frame,channels="BGR")

elif(choice=="CAMERA"):
    url=st.selectbox("Choose 0 for Primary and 1 for Secondary",("None",0,1))
    window=st.empty()
    btn=st.button("Start Detection")
    if btn:
        vid=cv2.VideoCapture(url)
        facemodel=cv2.CascadeClassifier("face.xml") 
        maskmodel = load_model("mask.h5", compile=False) # Load the model
        st.session_state["i"]=1
        btn2=st.button("Stop Detection")
        if btn2:
            vid.release()
            st.experimental_rerun()
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
                        path="data/"+str(st.session_state["i"])+".png"
                        st.session_state["i"]=st.session_state["i"]+1
                        cv2.imwrite(path,frame[y:y+w,x:x+l])
                        cv2.rectangle(frame,(x,y),(x+l,y+w),(0,0,255),3)
                    else:
                        cv2.rectangle(frame,(x,y),(x+l,y+w),(0,255,0),3)
                window.image(frame,channels="BGR")
elif(choice=="SHOW FACES"):
    if "n" not in st.session_state:
        st.session_state["n"]=1
    btn4=st.button("Previous Image")
    if btn4:
        if st.session_state["n"]>1:
            st.session_state["n"]-=1

    btn3=st.button("Next Image")
    if btn3:
        if st.session_state["n"]<st.session_state["i"]:
            st.session_state["n"]+=1
            
    path="data/"+str(st.session_state["n"])+".png"
    st.image(path)
