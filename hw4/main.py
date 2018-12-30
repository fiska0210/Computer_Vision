import numpy as np
import cv2
import time
import cv2.ximgproc

def compute_ssd(left, right, offset):
    print(left.shape)
    height, width = left.shape[0], left.shape[1]
    map = np.zeros((height, width), dtype=np.uint8)
    for h in range (height) :
        print(".", end="", flush=True)
        for w in range(width) :
            ssd = 0
            if h < 2 or w < 2 or h + 2 > height or w + 2 > width :
                cost = np.sum((left[h,w] - right[h,w]) ** 2)
                # cost = (left[h, w] - right[h, w]) ** 2
                map[h, w] = cost
            else :
                for v in range(-2, 2):
                    for u in range(-2, 2):
                        ssd_temp = np.sum(left[h+v, w+u] - right[h+v, (w+u-offset)])
                        # ssd_temp = (left[h+v, w+u] - right[h+v, (w+u-offset)])
                        # ssd.append(ssd_temp ** 2)
                        ssd += ssd_temp ** 2
                map[h, w] = ssd
    print(map.shape)

    return map

def computeDisp(Il, Ir, max_disp):
    # print(Il.shape)
    Il = Il.astype('float32')
    Ir = Ir.astype('float32')
    h, w, ch = Il.shape
    labels = np.zeros((h, w), dtype=np.uint8)
    # >>> Cost computation
    tic = time.time()
    map = np.zeros((h,w,max_disp+1))

    for offset in range(max_disp + 1):
        #sd or ad
        left = Il[:, offset :]
        right = Ir[:, : w - offset]
        cost = np.sum(abs(right - left ), axis=2) #ad
        # cost = np.sum((right - left )**2 , axis=2) #sd
        cost = np.pad(cost, ((0, 0), (offset, 0)), 'edge')
        # cost = (288,384)
        map[:, :, offset] = cost
        #ssd
        # cost = compute_ssd(left, right, offset)
        # cost = np.pad(cost, ((0, 0), (offset, 0)), 'edge')
        # map[:, :, offset] = cost
        
    toc = time.time()
    print('* Elapsed time (cost computation): %f sec.' % (toc - tic))

    # >>> Cost aggregation
    tic = time.time()
    # map = map.astype('float32')
    # map_fliter = cv2.ximgproc.guidedFilter(Il, map, 21, 0.16)
    map_fliter = cv2.blur(map, (8,8))
    toc = time.time()
    print('* Elapsed time (cost aggregation): %f sec.' % (toc - tic))

    # >>> Disparity optimization
    tic = time.time()
    dis = np.argmin(map_fliter,axis=2).astype('uint8') #WTA
    toc = time.time()
    print('* Elapsed time (disparity optimization): %f sec.' % (toc - tic))

    # >>> Disparity refinement
    tic = time.time()
    # use weighted median filter
    result = cv2.ximgproc.weightedMedianFilter(Il.astype('uint8'),dis,25,9,cv2.ximgproc.WMF_JAC)
    result = cv2.medianBlur(result,5)
    toc = time.time()
    print('* Elapsed time (disparity refinement): %f sec.' % (toc - tic))
    result = result.astype('uint8')
    return result


def main():
    print('Tsukuba')
    img_left = cv2.imread('./testdata/tsukuba/im3.png') 
    img_right = cv2.imread('./testdata/tsukuba/im4.png')
    max_disp = 15
    scale_factor = 16
    labels = computeDisp(img_left, img_right, max_disp)
    cv2.imwrite('tsukuba.png', np.uint8(labels * scale_factor))

    print('Venus')
    img_left = cv2.imread('./testdata/venus/im2.png') 
    img_right = cv2.imread('./testdata/venus/im6.png') 
    max_disp = 20
    scale_factor = 8
    labels = computeDisp(img_left, img_right, max_disp)
    cv2.imwrite('venus.png', np.uint8(labels * scale_factor))

    print('Teddy')
    img_left = cv2.imread('./testdata/teddy/im2.png') 
    img_right = cv2.imread('./testdata/teddy/im6.png') 
    max_disp = 60
    scale_factor = 4
    labels = computeDisp(img_left, img_right, max_disp)
    cv2.imwrite('teddy.png', np.uint8(labels * scale_factor))

    print('Cones')
    img_left = cv2.imread('./testdata/cones/im2.png') 
    img_right = cv2.imread('./testdata/cones/im6.png')
    max_disp = 60
    scale_factor = 4
    labels = computeDisp(img_left, img_right, max_disp)
    cv2.imwrite('cones.png', np.uint8(labels * scale_factor))


if __name__ == '__main__':
    main()
