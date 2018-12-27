import numpy as np
import cv2
import time
from matplotlib import pyplot as plt
from PIL import Image



def computeDisp(Il, Ir, max_disp):
    # Tsukuba shape (288, 384, 3)
    h, w, ch = Il.shape
    labels = np.zeros((h, w), dtype=np.uint8)
    img_left = np.array(Il)
    img_ritht = np.array(Ir)

    # print(h,w,ch)
    # >>> Cost computation
    tic = time.time()
    # TODO: Compute matching cost from Il and Ir
    # for i in range(h) :
    #     for j in range(w) :
            # if (left_img[])
            
    toc = time.time()
    print('* Elapsed time (cost computation): %f sec.' % (toc - tic))

    # >>> Cost aggregation
    tic = time.time()
    # TODO: Refine cost by aggregate nearby costs
    toc = time.time()
    print('* Elapsed time (cost aggregation): %f sec.' % (toc - tic))

    # >>> Disparity optimization
    tic = time.time()
    # TODO: Find optimal disparity based on estimated cost. Usually winner-take-all.
    toc = time.time()
    print('* Elapsed time (disparity optimization): %f sec.' % (toc - tic))

    # >>> Disparity refinement
    tic = time.time()
    # TODO: Do whatever to enhance the disparity map
    # ex: Left-right consistency check + hole filling + weighted median filtering
    toc = time.time()
    print('* Elapsed time (disparity refinement): %f sec.' % (toc - tic))

    return labels


def main():
    print('Tsukuba')
    img_left = cv2.imread('./testdata/tsukuba/im3.png')
    img_right = cv2.imread('./testdata/tsukuba/im4.png')
    max_disp = 15
    scale_factor = 16
    # labels = computeDisp(img_left, img_right, max_disp)
    labels = cv2.boxFilter(img_right, -1, (3,3))
    cv2.imwrite('tsukuba.png', np.uint8(labels * scale_factor))

    # print('Venus')
    # img_left = cv2.imread('./testdata/venus/im2.png')
    # img_right = cv2.imread('./testdata/venus/im6.png')
    # max_disp = 20
    # scale_factor = 8
    # labels = computeDisp(img_left, img_right, max_disp)
    # cv2.imwrite('venus.png', np.uint8(labels * scale_factor))

    # print('Teddy')
    # img_left = cv2.imread('./testdata/teddy/im2.png')
    # img_right = cv2.imread('./testdata/teddy/im6.png')    # max_disp = 60
    # scale_factor = 4
    # labels = computeDisp(img_left, img_right, max_disp)
    # cv2.imwrite('teddy.png', np.uint8(labels * scale_factor))

    # print('Cones')
    # img_left = cv2.imread('./testdata/cones/im2.png')
    # img_right = cv2.imread('./testdata/cones/im6.png')
    # max_disp = 60
    # scale_factor = 4
    # labels = computeDisp(img_left, img_right, max_disp)
    # cv2.imwrite('cones.png', np.uint8(labels * scale_factor))

def stereo_match(left_img, right_img, kernel, max_offset):
    # Load in both images, assumed to be RGBA 8bit per channel images
    # left_img = Image.open(left_img).convert('L')
    left_img = cv2.imread(left_img, cv2.IMREAD_GRAYSCALE)
    left = np.asarray(left_img)
    left = cv2.boxFilter(left, -1, (2,2))

    # right_img = Image.open(right_img).convert('L')
    right_img = cv2.imread(right_img, cv2.IMREAD_GRAYSCALE)
    right = np.asarray(right_img)    
    right = cv2.boxFilter(right, -1, (2,2))

    h, w = left_img.shape # assume that both images are same size   
    print(w, h)
    # Depth (or disparity) map
    depth = np.zeros((w, h), np.uint8)
    depth.shape = h, w
       
    kernel_half =int(kernel / 2)    
    # offset_adjust = 255 / max_offset  # this is used to map depth map output to 0-255 range
      
    for y in range(kernel_half, h - kernel_half):      
        print(".", end="", flush=True)  # let the user know that something is happening (slowly!)
        for x in range(kernel_half, w - kernel_half):
            best_offset = 0
            prev_ssd = 65534
            
            for offset in range(max_offset):               
                ssd = 0
                ssd_temp = 0                            
                
                # v and u are the x,y of our local window search, used to ensure a good 
                # match- going by the squared differences of two pixels alone is insufficient, 
                # we want to go by the squared differences of the neighbouring pixels too
                for v in range(-kernel_half, kernel_half):
                    for u in range(-kernel_half, kernel_half):
                        # iteratively sum the sum of squared differences value for this block
                        # left[] and right[] are arrays of uint8, so converting them to int saves
                        # potential overflow, and executes a lot faster 
                        ssd_temp = int(left[y+v, x+u]) - int(right[y+v, (x+u) - offset])  
                        # ssd += ssd_temp * ssd_temp              
                        ssd += abs(ssd_temp)
                # if this value is smaller than the previous ssd at this block
                # then it's theoretically a closer match. Store this value against
                # this block..
                if ssd < prev_ssd:
                    prev_ssd = ssd
                    best_offset = offset
                            
            # set depth output for this x,y location to the best match
            depth[y, x] = best_offset #* offset_adjust
    for y in range(h):
        for x in range(w):
            if y < kernel_half or y > h - kernel_half :
                if x < kernel_half or x > w - kernel_half :
                    depth[y, x] = depth[kernel_half, kernel_half]
                else :
                    depth[y, x] = depth[kernel_half, x]
            elif x < kernel_half or x > w - kernel_half :
                depth[y, x] = depth[y, kernel_half]

    # Convert to PIL and save it
    # Image.fromarray(depth).save('depth.png')
    
    
    depth =  cv2.medianBlur(depth, 5)

    cv2.imwrite('cones.png', np.uint8(depth * 4))
    print("Successful!\n")

if __name__ == '__main__':
    #main()
    #stereo_match('./testdata/tsukuba/im3.png', './testdata/tsukuba/im4.png', 10, 15) 
    #stereo_match('./testdata/venus/im2.png', './testdata/venus/im6.png', 10, 20) 
    #stereo_match('./testdata/teddy/im2.png', './testdata/teddy/im6.png', 10, 60) 
    stereo_match('./testdata/cones/im2.png', './testdata/cones/im6.png', 10, 60) 
