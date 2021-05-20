import cv2
import time
from cv2 import data
import mediapipe as mp
import math

class HandsAI:

    def __init__(self,
    mode = False,
    max = 2,
    de_conf = 0.5,
    tr_conf = 0.5):

        self.mode = mode
        self.max = max
        self.de = de_conf
        self.tr = tr_conf

        self.xhands = mp.solutions.hands
        self.hands = self.xhands.Hands(self.mode,self.max,self.de,self.tr)
        self.draw = mp.solutions.drawing_utils
        self.tipIds = [4, 8, 12, 16, 20]
    def detect_hands(self,img,bool=True):

        newimg = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        self.res = self.hands.process(newimg)
        if self.res.multi_hand_landmarks:
            for landmarks in self.res.multi_hand_landmarks:
                if bool: 
                    self.draw.draw_landmarks(img,landmarks,self.xhands.HAND_CONNECTIONS)

        return img
    
    def positions(self,img,num=0,draw = True):
        bbox = []
        self.lms = []
        xval = []
        yval = []

        if self.res.multi_hand_landmarks:
            handnum = self.res.multi_hand_landmarks[num]
            for index, mark in enumerate(handnum.landmark):
                h,w,c = img.shape
                cx,cy = int(mark.x * w) , int(mark.y * h) 
                self.lms.append([index,cx,cy])
                xval.append(cx)
                yval.append(cy)
                if index == 8:
                    if draw:
                        cv2.circle(img,(cx,cy),15,(255,0,255),cv2.FILLED)
            bbox = min(xval),min(yval),max(xval),max(yval)

            if draw:
                cv2.rectangle(img,(bbox[0]-20,bbox[1]-20),(bbox[2]+20,bbox[3]+20),(0,255,0),2)
        return self.lms, bbox

    def fingersUp(self):
        fingers = []
    
        if self.lms[4][1] < self.lms[3][1]:
            fingers.append(1)
        else:
            fingers.append(0)

        for id in range(1, 5):

            if self.lms[self.tipIds[id]][2] < self.lms[self.tipIds[id] - 2][2]:
                fingers.append(1)
            else:
                fingers.append(0)

        return fingers

    def finddis(self,x1,x2,img,draw = True):
        px1,y1 = self.lms[x1][1],self.lms[x1][2]
        px2,y2 = self.lms[x2][1],self.lms[x2][2]

        cx , cy = (px1+px2)//2, (y1+y2)//2
        length = math.hypot(px2-px1,y2-y1)
        return length,img,[px1,px2,y1,y2,cx,cy]
def main():
    cam = cv2.VideoCapture(0)
    prev = 0
    curr = 0
    ai = HandsAI()

    while True:
        bool, img = cam.read()
        img = ai.detect_hands(img)
        li,bbox = ai.positions(img)
        if len(li) != 0:
            fingers = ai.fingersUp()
            print(fingers)
        curr = time.time()
        fps = 1/(curr-prev)
        prev = curr
        cv2.putText(img,str(int(fps)),(30,40),cv2.FONT_HERSHEY_PLAIN,3,(0,100,255),3)
        
        cv2.imshow("IMAGE",img)
        cv2.waitKey(1)

if __name__ == "__main__":
    main()