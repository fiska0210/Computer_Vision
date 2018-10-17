import os

sigma_s = [1,2,3]
sigma_r = [0.05,0.1,0.2]
floder = '0c_guide/'

for i in range(3) :
    for j in range(3):
        subfloder = 'r-' + str(sigma_r[i]*255) + 's-' + str(sigma_s[j])
        os.makedirs(floder + subfloder)