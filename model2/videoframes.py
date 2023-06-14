import cv2
import cv2
cap= cv2.VideoCapture('/home/vishnu/Documents/ANMR_GCID/model/crop.mp4')
i=0
while(cap.isOpened()):
    ret, frame = cap.read()
    if ret == False:
        break
    cv2.imwrite('/home/vishnu/Documents/ANMR_GCID/model/video/kang'+str(i)+'.jpg',frame)
    i+=1
 
cap.release()
cv2.destroyAllWindows()