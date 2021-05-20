import cv2
import time
import mediapipe as mp


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

    def detect_hands(self,img,bool=True):

        newimg = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        self.res = self.hands.process(newimg)
        if self.res.multi_hand_landmarks:
            for landmarks in self.res.multi_hand_landmarks:
                if bool: 
                    self.draw.draw_landmarks(img,landmarks,self.xhands.HAND_CONNECTIONS)

        return img
    
    def postions(self,img,num=0,draw = True):

        lms = []
        if self.res.multi_hand_landmarks:
            handnum = self.res.multi_hand_landmarks[num]
            for index, mark in enumerate(handnum.landmark):
                h,w,c = img.shape
                cx,cy = int(mark.x * w) , int(mark.y * h) 
                lms.append([index,cx,cy])
                if index == 8:
                    if draw:
                        cv2.circle(img,(cx,cy),15,(255,0,255),cv2.FILLED)
        return lms

def main():
    cam = cv2.VideoCapture(0)
    prev = 0
    curr = 0
    ai = HandsAI()

    while True:
        bool, img = cam.read()
        img = ai.detect_hands(img)
        li = ai.postions(img)
        if len(li) != 0:
            print(li[4])
        curr = time.time()
        fps = 1/(curr-prev)
        prev = curr
        cv2.putText(img,str(int(fps)),(30,40),cv2.FONT_HERSHEY_PLAIN,3,(0,100,255),3)
        
        cv2.imshow("IMAGE",img)
        cv2.waitKey(1)

if __name__ == "__main__":
    main()