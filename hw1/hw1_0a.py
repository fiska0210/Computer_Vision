import numpy as np
import cv2
import math
import os
import sys

def filter_spatial(window_size, sigma_s):
    if window_size == 7 :
        i = [-3,-2,-1,0,1,2,3]
    elif window_size == 13 :
        i = [-6,-5,-4,-3,-2,-1,0,1,2,3,4,5,6]
    elif window_size == 19 :
        i = [-9,-8,-7,-6,-5,-4,-3,-2,-1,0,1,2,3,4,5,6,7,8,9]
    i = np.tile(i, (window_size, 1))
    j = i.transpose()
    s = np.exp(-(np.square(i) + np.square(j)) / (2 * (sigma_s ** 2)))
    ss = s.flatten()
    return ss

def filter_range(window_size, window_data, sigma_r): #img*img 
    if window_size == 7 :
        center = window_data[3,3]
    elif window_size == 13 : 
        center = window_data[6,6]
    elif window_size == 19 :
        center = window_data[9,9]

    T = np.zeros((window_size, window_size))
    for i in range(window_size) :
        for j in range(window_size) :
            b = float(window_data[i,j,0]) - float(center[0])
            g = float(window_data[i,j,1]) - float(center[1])
            r = float(window_data[i,j,2]) - float(center[2])
            T[i, j] = np.square(b) + np.square(g) + np.square(r)

    ran = np.exp(- T / (2 * (sigma_r ** 2)))
    return ran.flatten()

def JBfilter_range(window_size, window_data, sigma_r) : #guide
    window_data = window_data.astype(float)
    if window_size == 7 :
        center = window_data[3,3]
    elif window_size == 13 : 
        center = window_data[6,6]
    elif window_size == 19 :
        center = window_data[9,9]

    T = (window_data - center) ** 2   #single-channel
    ran = np.exp(- T / (2 * (sigma_r ** 2)))
    return ran.flatten()




def bilateral_filter(img, window_size, sigma_r, sigma_s) : 
    height = img.shape[0]
    width = img.shape[1]
    ans = np.zeros((height, width, 3))

    if window_size == 7 :
        color_img = np.pad(img, ((3,3), (3,3), (0,0)), 'symmetric')
    elif window_size == 13 :
        color_img = np.pad(img, ((6,6), (6,6), (0,0)), 'symmetric')
    elif window_size == 19 :
        color_img = np.pad(img, ((9,9), (9,9), (0,0)), 'symmetric')

    spatial_kernel = filter_spatial(window_size, sigma_s)
    
    for i in range(height) :
        for j in range(width) :
            window_data = color_img[i: i + window_size, j: j + window_size ]
            range_kernel = filter_range(window_size, window_data, sigma_r)
            weight = np.multiply(spatial_kernel, range_kernel)
            Iq = color_img[i: i + window_size, j: j + window_size ].reshape(-1, 3) # sum
            sumerize = np.average(Iq, weights = weight, axis = 0)

            ans[i, j] = sumerize
    return ans

def joint_bilateral_filter(img, guidance, window_size, sigma_r, sigma_s) : 
    height = img.shape[0]
    width = img.shape[1]
    ans = np.zeros((height, width, 3))
    if window_size == 7 :
        color_img = np.pad(img, ((3,3), (3,3), (0,0)), 'symmetric')
        guide_img = np.pad(guidance, ((3,3), (3,3)), 'symmetric') #single-channel
    elif window_size == 13 :
        color_img = np.pad(img, ((6,6), (6,6), (0,0)), 'symmetric')
        guide_img = np.pad(guidance, ((6,6), (6,6)), 'symmetric') #single-channel
    elif window_size == 19 :
        color_img = np.pad(img, ((9,9), (9,9), (0,0)), 'symmetric')
        guide_img = np.pad(guidance, ((9,9), (9,9)), 'symmetric') #single-channel

    spatial_kernel = filter_spatial(window_size, sigma_s)
    for i in range(height) :
        for j in range(width) :
            window_data = guide_img[i: i + window_size, j: j + window_size ]
            range_kernel = JBfilter_range(window_size, window_data, sigma_r)
            weight = np.multiply(spatial_kernel, range_kernel)
            Iq = color_img[i: i + window_size, j: j + window_size ].reshape(-1, 3) # sum
            sumerize = np.average(Iq, weights = weight, axis = 0)
            ans[i, j] = sumerize
    return ans



def conversion_filter(img, window_size, sigma_r, sigma_s, name ):
    foldername =  '0c_guide/'
    # fn2 = '0a_sub/'
    # folerJBF = '0a_JBF/'
    subfoldername = 'r-' + str(sigma_r) + 's-' + str(sigma_s) + '/'
    count = 0
    weight_list = []
    Y_gui = []
    cost_dic = {}

    for i in range(0,11,1) : 
        for j in range(0,11,1) :
            if i + j <= 10 :
                wb = i / 10
                wg = j / 10
                wr = abs(round((1 - wb - wg),1))
                count = count + 1
                weight_list.append([wb,wg,wr])
            else : 
                continue
    weight = np.array(weight_list)

    # BF_ans = bilateral_filter(img, window_size, sigma_r, sigma_s)
    # cv2.imwrite('BF_ans_0a.png', BF_ans)
    # print(BF_ans)

    for i in range(weight.shape[0]) :
        convent = (weight[i]).transpose()
        Y_gui = np.dot(img,convent)
        # JBF_ans = joint_bilateral_filter(img, Y_gui, window_size, sigma_r, sigma_s)
        # print(JBF_ans)
        # fileJBF = 'JBF_' +str(i) + '.png'
        #cv2.imwrite(folerJBF + fileJBF, JBF_ans)

        # delta = np.sum(np.absolute(BF_ans - JBF_ans))
        # print("Delta = "  + str(delta))
        #filename = 'number_' + str(i) + '.png'
        #cv2.imwrite(fn2 + filename, delta)
        # print("write number_" + str(i) + "done!")
        filename2 = '0c,b' + str(weight[i][0]) + 'g' + str(weight[i][1]) +'r' + str(weight[i][2]) + '.png'
        # subfoldername = 'r-' + str(sigma_r) + 's-' + str(sigma_s) + '/'
        cv2.imwrite(foldername + subfoldername + filename2 , Y_gui)
        # print(i)

        # Wbgr = (weight[i][0], weight[i][1], weight[i][2])
        # cost_dic[Wbgr] = delta
        # print('b' + str(weight[i][0]) + 'g' + str(weight[i][1]) +'r' + str(weight[i][2]))
        # print(delta)
        # print(cost_dic[Wbgr])

    return cost_dic
    # print(cost_dic.keys())

    # player.append(cost_dic.keys())
    # print(player)
    # print(len(player))


    # for i in range(player) :


def vote(img,name):

    sigma_s = [1, 2, 3]
    sigma_r = [12.75, 25.5, 51.0]
    times = 0
    cost_dic = {}
    vote_dic = {}
    player = []

    for kt in range(len(sigma_r)) :
        for rng in range(len(sigma_s)):
            r = 3 * int(sigma_s[kt])
            print(sigma_s[kt])
            print(sigma_r[rng])
            window_size = 2*r + 1
            print(window_size) 
            player = []
            cost_dic = conversion_filter(img, window_size, sigma_r[kt], sigma_s[rng], name)
        
            # for i in range(0,11,1) : 
            #     for j in range(0,11,1) :
            #         if i + j <= 10 :
            #             b = i / 10
            #             g = j / 10
            #             r = abs(round((1 - b - g),1))
                       
            #             player.append([b,g,r])
            #         else : 
            #             continue
            # print(len(player))
            # print(player)

            # for i in range(len(player)) :
            #     x = player[i][0]
            #     y = player[i][1]
            #     z = player[i][2]
            #     key = (x,y,z)
            #     value = cost_dic[(x,y,z)]
            #     if times == 0:
            #         vote_dic[key] = 0

            #     if x+0.1 > 1 or y-0.1 < 0:
            #         v1 = 999999999
            #     else:
            #         a = round(x+0.1,1)
            #         b = round(y-0.1,1)
            #         v1 = cost_dic[(a,b,z)]

            #     if x-0.1 < 0 or y+0.1 > 1 :
            #         v2 = 999999999
            #     else:
            #         a = round(x-0.1,1)
            #         b = round(y+0.1,1)
            #         v2 = cost_dic[(a,b,z)]

            #     if x+0.1 > 1 or z-0.1 < 0:
            #         v3 = 999999999
            #     else:
            #         a = round(x+0.1,1)
            #         c = round(z-0.1,1)
            #         v3 = cost_dic[(a,y,c)]
                
            #     if x-0.1 < 0 or z+0.1 > 1:
            #         v4 = 999999999
            #     else:
            #         a = round(x-0.1,1)
            #         c = round(z+0.1,1)
            #         v4 = cost_dic[(a,y,c)]
                
            #     if y+0.1 > 1 or z-0.1 < 0 :
            #         v5 = 999999999
            #     else:
            #         b = round(y+0.1,1)
            #         c = round(z-0.1,1)
            #         v5 = cost_dic[(x,b,c)]

            #     if y-0.1 < 0 or z+0.1 > 1 :
            #         v6 = 999999999
            #     else :
            #         b = round(y-0.1,1)
            #         c = round(z+0.1,1)
            #         v6 = cost_dic[(x,b,c)]

            #     print("-----------------")
            #     print(v1)
            #     print(v2)
            #     print(v3)
            #     print(v4)
            #     print(v5)
            #     print(v6)
            #     print(value)
            #     print("-----------------")
                
            #     if(value < v1 and value < v2 and value < v3 and value < v4 and value < v5 and value < v6):
            #         # print(vote_dic[key])
            #         vote_dic[key] += 1
            #         print(key)

            # times = times + 1
            # print(times)
            # print(vote_dic)







if __name__ == '__main__' : 
    name = sys.argv[1]
    img = []
    img = cv2.imread('testdata/' + name + '.png')
    #img = cv2.imread('testdata/0a.png')
    #guidance = cv2.imread('0c_y.png')
    vote(img,name)
    




