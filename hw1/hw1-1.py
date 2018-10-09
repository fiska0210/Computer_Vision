import numpy as np
import cv2

def conventional_rgb2gray(bgr):
    convent = np.array([0.114,0.587,0.299]).transpose()  #cv2 read by "BGR" 
    print(convent)
    img_y = np.dot(bgr,convent)
    #print(img_y)
    cv2.imwrite('test1.png', img_y)
    cv2.imshow("Result:", cv2.imread('test1.png'))
    cv2.waitKey(0)
    cv2.destroyAllWindows()










if __name__ == '__main__' : 
    img = []
    img = cv2.imread('testdata/1a.png')
    print(img)
    conventional_rgb2gray(img)
    
