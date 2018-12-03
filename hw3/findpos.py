import cv2
import numpy as np

#创建回调函数
def OnMouseAction(event,x,y,flags,param):

    if event == cv2.EVENT_LBUTTONDOWN:
        print("左键点击")
        print(x,y)
    elif event==cv2.EVENT_RBUTTONDOWN :
        print("右键点击")
    elif flags==cv2.EVENT_FLAG_LBUTTON:
        print("左鍵拖曳")
    elif event==cv2.EVENT_MBUTTONDOWN :
        print("中键点击")

'''
创建回调函数的函数setMouseCallback()；
下面把回调函数与OpenCV窗口绑定在一起
'''
img = np.zeros((500,500,3),np.uint8)
cv2.namedWindow('image')
cv2.setMouseCallback('image',OnMouseAction)     
cv2.imshow('image',img)
cv2.waitKey(30000)
cv2.destroyAllWindows()