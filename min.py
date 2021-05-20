import cv2
import time
import mediapipe as mp
cam = cv2.VideoCapture(0)

xhands = mp.solutions.hands
hands = xhands.Hands()
draw = mp.solutions.drawing_utils

prev = 0
curr = 0
while True:
    
    bool, img = cam.read()
    newimg = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    res = hands.process(newimg)
    if res.multi_hand_landmarks:
        for landmarks in res.multi_hand_landmarks:
            for index, mark in enumerate(landmarks.landmark):
                h,w,c = img.shape
                cx,cy = int(mark.x * w) , int(mark.y * h) 
                if index == 8:
                    cv2.circle(img,(cx,cy),15,(255,0,255),cv2.FILLED)
            draw.draw_landmarks(img,landmarks,xhands.HAND_CONNECTIONS)

    curr = time.time()
    fps = 1/(curr-prev)
    prev = curr
    cv2.putText(img,str(int(fps)),(30,40),cv2.FONT_HERSHEY_PLAIN,3,(0,100,255),3)
    
    cv2.imshow("IMAGE",img)
    cv2.waitKey(1)