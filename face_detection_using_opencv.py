# import OpenCV modules

import cv2 as cv 


haarcascade_face = cv.CascadeClassifier("haarcascade_frontalface_default.xml")   #specify haarcascade frontal face classifier
capture = cv.VideoCapture('Brazil.mp4')  #capture frames from a video

while True:
    ret,frame = capture.read()
    gray = cv.cvtColor(frame,cv.COLOR_BGR2GRAY)  #convert frames to gray colour
    face_detection=haarcascade_face.detectMultiScale(frame,1.1,4)


    # Rectangle boundary around face
    for (x,y,w,h) in face_detection:
        frame=cv.rectangle(frame,(x,y),(x+w,y+h),(255,255,255),thickness= 1)
    # Display video
    cv.imshow('live',frame)

    if cv.waitKey(1) == ord('e'):   #press e on the keyboard to exit
        break
capture.release()
cv.destroyAllWindows() 