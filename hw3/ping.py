import cv2
import numpy as np

def OnMouseAction(event,x,y,flags,param):
    if event == cv2.EVENT_LBUTTONDOWN:
        print(x,y)

img_front = cv2.imread('./input/crosswalk_front.jpg')
cv2.namedWindow('image')
cv2.setMouseCallback('image',OnMouseAction)  
cv2.imshow('image',img_front)
cv2.waitKey(50000)
cv2.destroyAllWindows()