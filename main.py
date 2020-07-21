from cv2 import cv2
import numpy as np
from keras.models import load_model
import matplotlib.pyplot as plt
from PIL import Image

cam_id=0
cap = cv2.VideoCapture(cam_id)
cap.set(3,640)
cap.set(4,480)
cap.set(10,50)

myColors=[[166,145,128,179,255,255],[97,190,202,114,255,255]]#[[red],[blue]]

myColorValues=[[34,34,242],[242,208,34]] #BGR format

myPoints=[]  #[x,y,colorId]

model=load_model("Mymodel.h5")

def findColor(img,myColor,myColorValues):
    imgHSV=cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
    count=0
    newPoints=[]
    for color in myColors:
        lower=np.array(color[0:3])
        upper=np.array(color[3:6])
        mask=cv2.inRange(imgHSV,lower,upper)
        x,y=getContours(mask)
        cv2.circle(imgResult,(x,y),15,(0,0,0),cv2.FILLED)
        if x!=0 and y!=0:
            newPoints.append([x,y,count])
        count+=1
        #cv2.imshow(str(color[0]),mask)
    return newPoints


def Draw(myPoints,myColorValues):
    #for point in myPoints:
        #cv2.circle(imgResult,(point[0],point[1]),8,myColorValues[point[2]],cv2.FILLED)
    for i in range(1,len(myPoints)):
        if myPoints[i - 1] is None or myPoints[i] is None:
            continue
        cv2.line(imgResult,(myPoints[i-1][0],myPoints[i-1][1]),(myPoints[i][0],myPoints[i][1]),(0,0,0),15)

def getContours(img):
    contours,heirarchy=cv2.findContours(img,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    x,y,w,h=0,0,0,0
    for cnt in contours:
        area=cv2.contourArea(cnt)
        #cv2.drawContours(imgResult,cnt,-1,(255,0,0),3)
        peri=cv2.arcLength(cnt,True)
        approx=cv2.approxPolyDP(cnt,0.02*peri,True)
        objCorners=len(approx)
        x,y,w,h=cv2.boundingRect(approx)
    return x+w//2,y


def Predict(imgResult):
    cropped=imgResult[0:340,300:640]
    cv2.imwrite("cropped.jpg",cropped)
    col=Image.open("cropped.jpg")
    gray=col.convert('L')
    bw=gray.point(lambda x:0 if x<100 else 255,'1')
    bw.save("bw_image.jpg")
    image=cv2.imread("bw_image.jpg",cv2.IMREAD_GRAYSCALE)
    image=cv2.bitwise_not(image)
    image_new=cv2.resize(image,(28,28))
    image = image_new.reshape(-1,28, 28, 1)
    image = np.array(image, dtype='float32')
    image /= 255
    pred_array = model.predict(image)
    #print(pred_array)
    pred = np.argmax(sum(pred_array))
    print('Predicting: {} with Accuracy: {}%'.format(pred,int(pred_array[0][pred]*100)))
    accuracy=int(pred_array[0][pred]*100)
    return (pred,accuracy)

pred,accuracy=0,0

while True:
    success, img=cap.read()
    flipHorizontal=cv2.flip(img,1)
    imgResult=flipHorizontal.copy()
    cv2.rectangle(imgResult,(300,0),(640,340),(255,255,255),cv2.FILLED)
    newPoints=findColor(flipHorizontal,myColors,myColorValues)
    if len(myPoints)!=0:
        Draw(myPoints,myColorValues)
    k=cv2.waitKey(1)
    if k==27:
        break
    elif k==ord('d'):
        if len(newPoints)!=0:
            for newP in newPoints:
                myPoints.append(newP)
        else:
            myPoints.append(None)
    elif k==ord('r'):
        myPoints=[]
    elif k==ord('p'):
        pred,accuracy=Predict(imgResult)
    cv2.rectangle(imgResult,(300,380),(640,450),(255,255,255),cv2.FILLED)
    cv2.putText(imgResult,'Predicting: {} with Accuracy: {}%'.format(pred,accuracy),
    (320,420),cv2.FONT_HERSHEY_COMPLEX,0.5,(0,0,0))
    cv2.imshow("MnistVision",imgResult)