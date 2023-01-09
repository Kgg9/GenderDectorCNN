import cv2
import numpy as np
import torch
import torchvision.transforms as T
import os
from GenderModel import ConvNetwork
import cvlib as cv


def prediction(image):
    imgArray =  image

    transfrom = T.Compose([T.ToPILImage(),T.Resize((100,100)),T.ToTensor()])
    img_t = transfrom(imgArray).unsqueeze(0)

    model = ConvNetwork()
    model.load_state_dict(torch.load("modelM.pth"))
    model.eval()

    with torch.no_grad():
        predict = model(img_t)
        probability = torch.nn.functional.softmax(predict, dim=1)

        conf, classes = torch.max(probability, 1)
        return (conf, classes)



cap = cv2.VideoCapture(0)

# detects Face
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

while cap.isOpened():

    # reads from webcam
    status, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Detect the faces
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=7,
        minSize=(100, 100),
        flags=cv2.CASCADE_SCALE_IMAGE
    )

    classess = ["Man", "Woman"]

    # Draw the rectangle around each face
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 255, 255), 2)
        faceCrop = np.copy(img[y:(y+h),x:(x+w)])


        #Making and Displaying the Label

        confidence, gender = prediction(faceCrop)
        label = classess[gender.item()]
        print(confidence)

        cv2.putText(img, label, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7,(0, 255, 0), 2 )


    # Display
    cv2.imshow('img', img)

    key = cv2.waitKey(1)
    if key == 27:
        break