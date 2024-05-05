# VisageIQ
**Mask Detection App**

This project uses machine learning classification using OpenCV and Tensorflow to detect facemasks on people.The method involves drawing a bounding box around the face to indicate mask presence. If a person's face is not covered with a mask, the system can detect this and save the individual's photo prompting them to take necessary precautions.

**METHODOLOGY**

Dataset Collection: The dataset was collected from Kaggle Repository and was split into training and testing data after its analysis.

Training a model to detect face masks: A default OpenCV module was used to obtain faces followed by training a Keras model to identify face mask.

Detecting the person not wearing a mask: A open CV model was trained to detect the faces of the people who are not wearing masks and saving their photos.

