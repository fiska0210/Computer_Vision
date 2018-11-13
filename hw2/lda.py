import numpy as np
import cv2
import sys
import os
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier

'''
$1: path of whole dataset
$2: path of the first 1 Fisherface
E.g., python3 hw2-3_lda.py ./hw2/hw2-2_data ./output_Fisher.png
'''

def read_data(path) :
    list_all_data = []
    for i in range(40) :
        for j in range(7) :
            filename = path + str(i+1) + '_' + str(j+1) + '.png'
            # print(filename)
            img = cv2.imread(filename, 0) #one channel
            img = img.flatten()
            
            list_all_data.append(img)
            
   
    img_data = np.array(list_all_data, float)
    
    mean = np.mean(img_data, axis = 0) 
    return img_data, mean

def pca(img, N) :
    # img_data (280, 2576)
    # mean (2576, )
    # covariance matrix (2576, 2576)
    covariance = np.cov(img, rowvar = False) 
    eigenvalue_pca, eignevector_pca = np.linalg.eig(covariance)
    eignevector_pca = np.real(eignevector_pca)
    # print(img.shape[0])
    # eig_vec (2576, 279)
    eignevector_pca = eignevector_pca[:, 0 : N]

    return eignevector_pca


def lda(img, eig_vec_pca, N) :
    # pca (2576, 240)     2576 x N-C
    pca = eig_vec_pca[:, 0 : 240]
    
    # x_m (280, 2576)
    x_m = img - np.average(img, axis = 0)
    # W (280, 240)
    W = np.matmul(x_m, pca)
    u = np.average(W, axis = 0) #avg for all data
    S_W = 0
    S_B = 0
    for i in range(0, 280, 7): # C
        # print(i)
        x_i = W[i: i + 7] # slice a group
        u_i = np.average(x_i, axis = 0) # avg for this group
        for j in range(7) : 
            a_j = (x_i[j] - u_i).reshape(-1, 1)
            b_j = (x_i[j] - u_i).reshape(1, -1) #(x-u)T
            S_W = S_W + np.matmul(a_j, b_j) 
        
        a_i = (u_i - u).reshape(-1, 1)
        b_i = (u_i - u).reshape(1, -1)
        S_B = S_B + np.matmul(a_i, b_i)
    
    S_W_1 = np.linalg.inv(S_W)  #Sw inveter
    eigenvalue_lda, eignevector_lda = np.linalg.eig(np.matmul(S_W_1,S_B))
    # eignevector_lda (240, 240)
    eignevector_lda = np.real(eignevector_lda)
    # (240, 39) = (N-C)x(C-1)
    eignevector_lda = eignevector_lda[:, 0 : N] #C-1
    return eignevector_lda

def fishfaces(eig_vec_pca, eig_vec_lda, outputfile ) :
    # dx(C-1) = dx(N-C) X (N-C)x(C-1)
    # eig_vec_pca (2576, 279)
    # eig_vec_lda (240, 39)
    # pca = (2576, 240)
    pca = eig_vec_pca[:, 0 : 240]
    # fishfaces (2576, 39)
    fishfaces = np.matmul(pca, eig_vec_lda)
    for i in range(5):
        fishfaces_min = np.min(fishfaces[:, i])
        fishfaces_max = np.max(fishfaces[:, i])
        fishfaces_re = ((fishfaces[:, i] - fishfaces_min)/(fishfaces_max - fishfaces_min)) * 255
        fishfaces_re = fishfaces_re.reshape(56,46)
        # print(fishfaces_re)
        filename = 'fishfaces' + str(i+1) + '.png'
        cv2.imwrite(filename, fishfaces_re)
        if i == 0 :
            cv2.imwrite(outputfile, fishfaces_re)
        # cv2.imshow("Result:", cv2.imread(filename))
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

def visualize (data, mean, eig_vec_pca, eig_vec_lda, dim) :
    list_test_data = []
    for i in range(40) :
        for j in range(7 ,10) :
            filename = path + str(i+1) + '_' + str(j+1) + '.png'
            # print(filename)
            img = cv2.imread(filename, 0) #one channel
            img = img.flatten()
            
            list_test_data.append(img)
    
    # test_data (120,2576)
    test_data = np.array(list_test_data, float)
    # test_data_mean (2576, )
    test_data_mean = np.mean(test_data, axis = 0) 
    
    pca = eig_vec_pca[: , : 240] #pca
    lda = eig_vec_lda[: , : dim] #lda
    
    
    #x_m (120,576)
    x_m = test_data - mean
    W = np.matmul(x_m, pca)

    y = np.matmul(W, lda)
    
    tsne = TSNE(n_components = 2).fit_transform(y)

    for i in range(0, 120, 3):  #40, 3
        x = tsne[ i : i + 3, 0] 
        y = tsne[ i : i + 3, 1]
        
        plt.scatter(x,y) 
    plt.savefig('LDA_visualize.png')
    print("Sace lda visualize Successful!")


if __name__ == '__main__' :
    path = sys.argv[1] + '/'
    outputfile = sys.argv[2]

    # # path = 'hw2-2_data/' # for all data, N=400
    img, mean = read_data(path)
    eig_vec_pca = pca(img, 279)
    eig_vec_lda = lda(img, eig_vec_pca, 39)
    fishfaces(eig_vec_pca,eig_vec_lda, outputfile)
    # visualize(img, mean, eig_vec_pca, eig_vec_lda, 30)
