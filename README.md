# VisageIQ
**Mask Detection App**

This project uses machine learning classification using OpenCV and Tensorflow to detect facemasks on people.The method involves drawing a bounding box around the face to indicate mask presence. If a person's face is not covered with a mask, the system can detect this and save the individual's photo prompting them to take necessary precautions.

**METHODOLOGY**

Dataset Collection: The dataset was collected from Kaggle Repository and was split into training and testing data after its analysis.

Training a model to detect face masks: A default OpenCV module was used to obtain faces followed by training a Keras model to identify face mask.

Detecting the person not wearing a mask: A open CV model was trained to detect the faces of the people who are not wearing masks and saving their photos.


**RESULT AND ANALYSIS**

The model is trained, validated and tested upon two datasets. Corresponding to dataset 1, the method attains accuracy up to 95.77%. This optimized accuracy mitigates the cost of error. Dataset 2 is more versatile than dataset 1 as it has multiple faces in the frame and different types of masks having different colors as well. Therefore, the model attains an accuracy of 94.58% on dataset 2. The system can efficiently detect partially occluded faces either with a mask or hair or hand. It considers the occlusion degree of four regions – nose, mouth, chin and eye to differentiate between annotated mask or face covered by hand. Therefore, a mask covering the face fully including nose and chin will only be treated as “with mask” by the model.

The main challenges faced by the method mainly comprise of varying angles and lack of clarity. Indistinct moving faces in the video stream make it more difficult. However, following the trajectories of several frames of the video helps to create a better decision – “with mask” or “without mask”.

**CONCLUSION**

Using basic ML tools and simplified techniques the project has achieved reasonably high accuracy. It can be used for a variety of applications. Wearing a mask may be obligatory in the near future, considering the Covid-19 crisis. Many public service providers will ask the customers to wear masks correctly to avail of their services. The deployed model will contribute immensely to the public health care system. In future it can be extended to detect if a person is wearing the mask properly or not. The model can be further improved to detect if the mask is virus prone or not i.e. the type of the mask is surgical, N95 or not.
