import cv2
import time
from module import HandsAI
import numpy as np
from pynput.mouse import Controller,Button
from win32api import GetSystemMetrics
ctrl_mouse = Controller()

width = GetSystemMetrics(0)
height = GetSystemMetrics(1)

cam = cv2.VideoCapture(0)
h= 480
w= 640
cam.set(3,w)
cam.set(4,h)
BOUND = 100
detector = HandsAI(max=1)
smoothening = 5
plocX, plocY = 0,0
clocX , clocY=  0,0
curr = 0
prev = 0
fingerup = False
while True:

    bool , img = cam.read()
    img = detector.detect_hands(img)
    lm ,bbox = detector.positions(img)
    if len(lm) != 0:
        x1 , y1 = lm[8][1:]
        x2, y2 = lm[12][1:]
        fingers = detector.fingersUp()
        if fingers[1] == 1 and fingers[2] == 0:
            cv2.rectangle(img,(BOUND,BOUND),(w-BOUND,h-BOUND),(255,0,255),2)
            fingerup = False
            x3 = np.interp(x1,(BOUND,w-BOUND),(0,width))
            y3 = np.interp(y1,(BOUND,h-BOUND),(0,height))

            clocX = plocX + (x3-plocX)/smoothening
            clocY = plocY + (y3 - plocY)/smoothening
            ctrl_mouse.position = (width-clocX,clocY)
            plocY,plocX = clocY,clocX
        cv2.circle(img ,(x1,y1),15,(255,0,255),cv2.FILLED)

        if fingers[1] == 1 and fingers[2] ==1:
            if not fingerup:
                leng , img , info = detector.finddis(8,12,img)
                if leng < 45:
                    ctrl_mouse.press(Button.left)
                    ctrl_mouse.release(Button.left)
                    fingerup = True

    curr = time.time()
    fps = 1/(curr-prev)
    prev = curr
    cv2.putText(img,str(int(fps)),(30,40),cv2.FONT_HERSHEY_PLAIN,3,(0,100,255),3)
    cv2.imshow("Image",img)
    cv2.waitKey(1)

    