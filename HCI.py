# Matr No. 30001275 
# HCI Makeup assignment


import cv2
import pyautogui
import dlib
import numpy as np
import matplotlib.pyplot as plt
from math import hypot
import time

predictor = dlib.shape_predictor("/Users/yuvrajsingh/Documents/HCI/shape_predictor_68_face_landmarks.dat")
weights =  "/Users/yuvrajsingh/Documents/HCI/res10_300x300_ssd_iter_140000_fp16.caffemodel"
architecture = "/Users/yuvrajsingh/Documents/HCI/deploy.prototxt.txt"
combi = cv2.dnn.readNetFromCaffe(architecture,weights)

capture = cv2.VideoCapture(0)

prevtime = 0
newtime = 0

pyautogui.PAUSE = 0

def facedetector(image, face_threshold =0.5):
    h, w = image.shape[:2]
    blob = cv2.dnn.blobFromImage(image, 1.0,(300,300), (105.0, 115.0, 120.0))
    combi.setInput(blob)
    faces = combi.forward()
    score = faces[:,:,:,2]
    i = np.argmax(score)
    face = faces[0,0,i]
    face_confidence = face[2]

    if face_confidence > face_threshold:
        box = face[3:7] * np.array([w, h, w, h])
        (x1, y1, x2, y2) = box.astype("int")
        annotated_frame = cv2.rectangle(image.copy(),(x1, y1),(x2, y2),(0, 255, 0),3)
        output = (annotated_frame, (x1, y1, x2, y2), True, face_confidence)
    else:
        output = (image,(),False, 0)
    return output

def landmarksdetector(box, image):
    gray_scale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    (x1, y1, x2, y2) = box
    shape = predictor(gray_scale, dlib.rectangle(x1, y1, x2, y2))
    landmarks = shape_numpy(shape)

    for (x, y) in landmarks:
        annotated_image = cv2.circle(image, (x, y),1, (255, 255, 0), -1)
    return annotated_image, landmarks
    
def shape_numpy(shape): 
    landmark = np.zeros((68, 2), dtype="int")

    for i in range(0, 68): 
        landmark[i] = (shape.part(i).x, shape.part(i).y)    
    return landmark

def eyebrows(landmark, eyebrows_threshold = 0.38): 
    EBR1 = hypot(landmark[23][0] - landmark[47][0], landmark[23][1] - landmark[47][1])
    EBR2 = hypot(landmark[24][0] - landmark[46][0], landmark[24][1] - landmark[46][1])
    EBR3 = hypot(landmark[27][0] - landmark[16][0], landmark[27][1] - landmark[16][1])
    EBL1 = hypot(landmark[19][0] - landmark[41][0], landmark[19][1] - landmark[41][1])
    EBL2 = hypot(landmark[20][0] - landmark[40][0], landmark[20][1] - landmark[20][1])
    EBL3 = hypot(landmark[27][0] - landmark[0][0], landmark[27][1] - landmark[0][1])

    eyebrows_ratio = ((((EBR1 + EBR2))/(2.0 * EBR3)) + (((EBL1 + EBL2)) / (2.0 * EBL3)))/2

    if eyebrows_ratio > eyebrows_threshold:
        return True, eyebrows_ratio
    else:
        return False, eyebrows_ratio

def eyes(landmark, eye_threshold = 0.26): 
    P1 = hypot(landmark[43][0] - landmark[47][0], landmark[43][1] - landmark[47][1])
    Q1 = hypot(landmark[44][0] - landmark[46][0], landmark[44][1] - landmark[46][1])
    R1 = hypot(landmark[42][0] - landmark[45][0], landmark[42][1] - landmark[45][1])
    P2 = hypot(landmark[37][0] - landmark[41][0], landmark[37][1] - landmark[41][1])
    Q2 = hypot(landmark[38][0] - landmark[40][0], landmark[38][1] - landmark[40][1])
    R2 = hypot(landmark[36][0] - landmark[39][0], landmark[36][1] - landmark[39][1]) 

    eye_ratio = ((((P1 + Q1))/(2.0 * R1)) + (((P2 + Q2)) / (2.0 * R2)))/2

    if eye_ratio > eye_threshold:
        return True, eye_ratio
    else:
        return False, eye_ratio

def mouth(landmark, mouth_threshold = 0.7): 
    M1 = hypot(landmark[50][0] - landmark[58][0], landmark[50][1] - landmark[58][1])
    M2 = hypot(landmark[52][0] - landmark[56][0], landmark[52][1] - landmark[56][1])
    M3 = hypot(landmark[48][0] - landmark[54][0], landmark[48][1] - landmark[54][1])

    mouth_ratio = ((M1 + M2)*1.2) / (2.0 * M3) 

    if mouth_ratio > mouth_threshold:
        return True, mouth_ratio
    else:
        return False, mouth_ratio
    
cv2.namedWindow('Face to Keyboard command', cv2.WINDOW_NORMAL)

while(True):
    rectify, frame = capture.read()
    if not rectify:
        break

    frame = cv2.flip( frame, 1 )
    face_image, boxcoords, status, config = facedetector(frame)

    newtime = time.time()
    fps = (1/(newtime-prevtime))
    prevtime = newtime
    fps = int(fps)
    fps = str(fps)
    cv2.putText(frame, fps, (20, 100),cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 3, cv2.LINE_AA)
   
    if status:  
        landmark, landmarks = landmarksdetector(boxcoords, frame)

        eyebrows_status,_ = eyebrows(landmarks, eyebrows_threshold = 0.35)
        if eyebrows_status:     
            pyautogui.keyDown('up')
        else:
            pyautogui.keyUp('up')

        mouth_status,_ = mouth(landmarks, mouth_threshold = 0.7)
        if mouth_status:     
            pyautogui.keyDown('n')
        else:
            pyautogui.keyUp('n')
      
        eyes_status,_ = eyes(landmarks, eye_threshold = 0.26)
        if eyes_status:     
            pyautogui.keyUp('space')
        else:
            pyautogui.keyDown('space')
        
        cv2.putText(frame,'Mouth is Open->> {}'.format(mouth_status),
                    (22, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 200),2)
        cv2.putText(frame,'Are Eyes Open->> {}'.format(eyes_status),
                    (20, 47), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 100, 0),2)
        cv2.putText(frame,'Are Eyebrows UP->> {}'.format(eyebrows_status),
                    (20, 69), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 0, 255),2)
    cv2.imshow('Face to Keyboard command',frame)
   

    if cv2.waitKey(1) & 0xFF == ord('x'):
        break

capture.release()
cv2.destroyAllWindows()