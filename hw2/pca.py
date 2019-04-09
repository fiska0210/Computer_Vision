import numpy as np
import cv2
import sys
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
'''
$1: path of whole dataset
$2: path of the input testing image
$3: path of the output testing image reconstruct by
E.g., python3 hw2-2_pca.py ./hw2/hw2-2_data ./hw2/test_image.png ./output_pca.png
'''

def pca_data(path) :
    list_all_data = []
    for i in range(40) :
        for j in range(7) :
            filename = path + str(i+1) + '_' + str(j+1) + '.png'
            # print(filename)
            img = cv2.imread(filename, 0) #one channel
            img = img.flatten()
            
            list_all_data.append(img)
   
    all_data = np.array(list_all_data, float)
    
    all_data_mean = np.mean(all_data, axis = 0) 
    # print("======== mean face=========")
    return all_data, all_data_mean

def eignevector(img) :
    # img (280, 2576)
    # mean (2576, )
    # covariance matrix (2576, 2576)
    covariance = np.cov(img, rowvar = False) 
    eigenvalue, eignevector = np.linalg.eig(covariance)
    eignevector = np.real(eignevector)
    # eig_vec (2576, 279)
    eig_vec_T = eignevector.transpose()
    return eignevector, eig_vec_T


def pca_first_five_eigvec(eig_vec) :
    # eig_vec (2576, 279)
    eignevector = eig_vec[:, 0 : 279]
    eig_vec_T = eignevector.transpose()
    for i in range(5):
        eig_min = np.min(eignevector[:, i])
        eig_max = np.max(eignevector[:, i])
        eig_vec = ((eignevector[:, i] - eig_min)/(eig_max - eig_min)) * 255
        eig_vec = eig_vec.reshape(56,46)
        filename = 'eigenface' + str(i+1) + '.png'
        cv2.imwrite(filename, eig_vec)
        # cv2.imshow("Result:", cv2.imread(filename))
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
    
def pca_restruction (eig_vec, N, testfilename, outputfile):
    ##### reconstruction image 8_6 #####
    img_8_6 = cv2.imread(testfilename, 0) #one channel
    img_8_6 = img_8_6.flatten()
    eig_vec_T = eig_vec[:, 0 : 279].transpose()
    # x_m (2576,)
    x_m = img_8_6 - mean
    # p (279,)
    p = np.matmul(eig_vec_T, x_m)
    p = p.reshape(279, 1)

    pv = []
    
    for i in range(N) : # n = 5,50,150,all(279) 
        pi = p[i][0] * eig_vec_T[i]
        pv.append(pi)
        
    pv = np.array(pv)
    pv_sum = np.sum(pv, axis = 0) + mean
    img_re = pv_sum.reshape(56,46)
    # filename_re = 'n=' + str(N) + '.png'
    # cv2.imwrite(filename_re, img_re)
    cv2.imwrite(outputfile, img_re)
    # cv2.imshow("Result:", cv2.imread(filename_re))
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    mse = np.sum((img_8_6 - pv_sum) ** 2) / 2576
    # print("rmse = " + str(mse))

    ##### reconstruction image 8_6  END! #####


def visualize (data, mean, dim) :
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

    covariance = np.cov(data, rowvar = False) 
    eigenvalue, eignevector = np.linalg.eig(covariance)
    eignevector = np.real(eignevector)
    eig_vec = eignevector[:, 0 : dim]

    y = np.matmul((test_data - mean), eig_vec)

    tsne = TSNE(n_components=2).fit_transform(y)
    for i in range(0, 120, 3):  #40, 3
        x = tsne[ i : i + 3, 0] 
        y = tsne[ i : i + 3, 1]
        
        plt.scatter(x,y) 
    
    # plt.savefig('PCA_visualize.png')
    # print("Sace pcv visualize Successful!")


def meanface(mean) :
    mean_face = mean.reshape(56,46)
    cv2.imwrite('mean_face.png', mean_face)
    # cv2.imshow("Result:", cv2.imread('mean_face.png'))
    # cv2.write('mean face', mean_face)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
        
    

if __name__ == '__main__' :
    path = sys.argv[1] + '/'
    testfilename = sys.argv[2]
    outputfile = sys.argv[3]

    # path = 'hw2-2_data/' # for all data, N=400
    
    data, mean = pca_data(path)
    # print("Read hw2-2_data Successful!")
    eig_vec, eig_vec_T = eignevector(data)
    # plot mean face
    # meanface(mean)
    # print("plot mean face successful!")
    # plot first five eigenfaces
    # pca_first_five_eigvec(eig_vec)
    # print("plot first five eigenface successful!")

    # plot first five eigenfaces 
    # for i in [5, 50, 150, 279]:
    #     pca_restruction(eig_vec, i, testfilename)
    pca_restruction(eig_vec, 279, testfilename, outputfile)
    print("resturcetion successful!")

    

    visualize(data, mean, 100)


    
