# import OpenCV modules
import cv2 as cv 


haarcascade_face = cv.CascadeClassifier('haarcascade_frontalface_default.xml')   #specify haarcascade frontal face classifier
haarcascade_eye = cv.CascadeClassifier('haarcascade_eye_tree_eyeglasses.xml')
capture = cv.VideoCapture('Brazil.mp4')  #capture frames from a video

while True:
    ret,frame = capture.read()  #read frames from the video
    gray = cv.cvtColor(frame,cv.COLOR_BGR2GRAY)  #convert frames to gray colour
    face_detection = haarcascade_face.detectMultiScale(frame,1.1,4) #use haarcascade to detect faces
    eye_detection = haarcascade_eye.detectMultiScale(frame,1.1,4) #use haarcascade to detect eyes

    # Rectangle boundary around face
    for (x,y,w,h) in face_detection:
        frame=cv.rectangle(frame,(x,y),(x+w,y+h),(255,255,255),thickness= 3)

    for (x,y,w,h) in eye_detection:
        frame=cv.rectangle(frame,(x,y),(x+w,y+h),(255,255,255),thickness= 1)
    
    # DisplayS video
    cv.imshow('live',frame)

    if cv.waitKey(1) == ord('e'):  #press key 'e' on the keybord to exit video wjile playing
        break


capture.release()
cv.destroyAllWindows() 