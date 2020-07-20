import cv2
import numpy as np
import concurrent.futures

cam_id=0
cap = cv2.VideoCapture(cam_id)
cap.set(3,640)
cap.set(4,480)
cap.set(10,50)

myColors=[[166,145,128,179,255,255],[97,190,202,114,255,255]]#[[red],[blue]]

myColorValues=[[34,34,242],[242,208,34]] #BGR format

myPoints=[]  #[x,y,colorId]

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
        if(myPoints[i-1][2]==myPoints[i][2]):
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

while True:
    success, img=cap.read()
    flipHorizontal=cv2.flip(img,1)
    imgResult=flipHorizontal.copy()
    cv2.rectangle(imgResult,(300,0),(640,340),(255,255,255),cv2.FILLED)
    newPoints=findColor(flipHorizontal,myColors,myColorValues)
    k=cv2.waitKey(1)
    if k==27:
        break
    elif k==ord('d'):
        if len(newPoints)!=0:
            for newP in newPoints:
                myPoints.append(newP)
        else:
            myPoints.append(None)
    if len(myPoints)!=0:
        Draw(myPoints,myColorValues)
    cv2.imshow("Output",imgResult)
