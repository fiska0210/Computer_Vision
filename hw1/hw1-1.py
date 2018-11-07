import numpy as np
import cv2
import os
import sys

def conventional_rgb2gray(bgr,name):
    #convent = np.array([0.114,0.587,0.299]).transpose()  #cv2 read by "BGR" 
    convent = np.array([0,0,1]).transpose()
    print(convent)
    img_y = np.dot(bgr,convent)
    print(img_y)
    cv2.imshow("Result:", img_y)
    filename = name + '_y.png'
    # cv2.imwrite(filename, img_y)
    # cv2.imshow("Result:", cv2.imread(filename))
    cv2.imwrite('ans/0a_y2_0010.png', img_y)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def create_floder(floder):
    sigma_s = [1,2,3]
    sigma_r = [0.05,0.1,0.2]
    floder = floder + '/'
    for i in range(3) :
        for j in range(3):
            subfloder = 'r-' + str(sigma_r[i]*255) + 's-' + str(sigma_s[j])
            os.makedirs(floder + subfloder)

if __name__ == '__main__' : 
    img = []
    name = sys.argv[1]
    img = cv2.imread('testdata/' + name + '.png')
    print(img)
    conventional_rgb2gray(img, name)
    # floder = name + '_guide'
    # print(floder)
    # create_floder(floder)
    
    
