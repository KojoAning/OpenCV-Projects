# import OpenCV modules

import cv2 as cv 



haarcascade_eye = cv.CascadeClassifier("haarcascade_eye_tree_eyeglasses.xml")
capture = cv.VideoCapture('Brazil.mp4')  #capture frames from a video

while True:
    ret,frame = capture.read()
    gray = cv.cvtColor(frame,cv.COLOR_BGR2GRAY)  #convert frames to gray colour
    eye_detection = haarcascade_eye.detectMultiScale(frame,1.1,4) # eye detection usinng haarcascade


    # Rectangle boundary around face
    for (x,y,w,h) in eye_detection:
        frame = cv.rectangle(frame,(x,y),(x+w,y+h),(255,255,0),thickness=2)
    # Display video
    cv.imshow('live',frame)

    if cv.waitKey(1) == ord('e'):   #press e on the keyboard to exit
        break
capture.release()
cv.destroyAllWindows() 