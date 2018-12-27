import numpy as np
import cv2


# u, v are N-by-2 matrices, representing N corresponding points for v = T(u)
# this function should return a 3-by-3 homography matrix
def solve_homography(u, v):
    # u(4, 2) v(4, 2)
    N = u.shape[0]
    if v.shape[0] is not N:
        print('u and v should have the same size')
        return None
    if N < 4:
        print('At least 4 points should be given')
    A = np.zeros((2*N, 9)) # A(8, 9)
    b = np.zeros((2*N, 1))
    H = np.zeros((3, 3))
    h = np.zeros(9)
    # Ah = b, H = h.respape(3, 3)
    for i in range(4) :
        col_1 = 2*i
        col_2 = 2*i + 1
        ux = u[i][0]
        uy = u[i][1]
        vx = v[i][1]
        vy = v[i][0]
        A[:][col_1] = [ux, uy, 1, 0, 0, 0, -(ux*vx), -(uy*vx), -(vx)]
        A[:][col_2] = [ 0, 0, 0, ux, uy, 1, -(ux*vy), -(uy*vy), -(vy)]
    A_T = A.transpose()
    U, sigma, VT = np.linalg.svd(np.matmul(A_T, A))
    h = U[:, 8]
    H = h.reshape(3, 3)
    return H

# corners are 4-by-2 arrays, representing the four image corner (x, y) pairs
def transform(img, canvas, corners): 
    u = np.array([ [0, 0], [0, img.shape[1]], [img.shape[0], 0], [img.shape[0], img.shape[1]] ])
    v = corners
    H = solve_homography(u, v)
    height = img.shape[0]
    width = img.shape[1]
    new = []
    for i in range(height) :
        for j in range(width) :
            z_n = np.matmul(H[2, :], [i, j, 1])
            x_n = np.matmul(H[0, :], [i, j, 1])
            y_n = np.matmul(H[1, :], [i, j, 1])
            new.append([(x_n/z_n), (y_n/z_n), z_n]) # let z = 1
    # pos (121104, 3)
    pos = np.array(new) 
    #reshape to the frame
    row_n = pos[:, 0].reshape(height, width)
    col_n = pos[:, 1].reshape(height, width)
    for i in range(height) :
        for j in range(width) :
            x = int(row_n[i, j])
            y = int(col_n[i, j])
            canvas[x, y] = img[i, j]
    return canvas
	
def transform_cor(img, canvas, canvas_corners, img_corners ) :
    u = np.array(img_corners)
    v = np.array(canvas_corners)
    H = np.linalg.inv(solve_homography(u, v))
    height = canvas.shape[0]
    width = canvas.shape[1]
    new = []
    for i in range(height) :
        for j in range(width) :
            z_n = np.matmul(H[2, :], [i, j, 1])
            x_n = np.matmul(H[0, :], [i, j, 1])
            y_n = np.matmul(H[1, :], [i, j, 1])
            new.append([(x_n/z_n), (y_n/z_n), z_n]) # let z = 1
    # pos (121104, 3)
    pos = np.array(new) 
    #reshape to the frame
    row_n = pos[:, 0].reshape(height, width)
    col_n = pos[:, 1].reshape(height, width)
    for i in range(height) :
        for j in range(width) :
            x = int(row_n[i, j])
            y = int(col_n[i, j])
            canvas[i, j] = img[x, y]
    return canvas

def OnMouseAction(event,x,y,flags,param):
    if event == cv2.EVENT_LBUTTONDOWN:
        print(x,y)

def part1():
    canvas = cv2.imread('./input/times_square.jpg')
    img1 = cv2.imread('./input/wu.jpg')
    img2 = cv2.imread('./input/ding.jpg')
    img3 = cv2.imread('./input/yao.jpg')
    img4 = cv2.imread('./input/kp.jpg')
    img5 = cv2.imread('./input/lee.jpg')

    corners1 = np.array([[818, 352], [884, 352], [818, 407], [885, 408]]) #v1
    corners2 = np.array([[311, 14], [402, 150], [157, 152], [278, 315]])
    corners3 = np.array([[364, 674], [430, 725], [279, 864], [369, 885]])
    corners4 = np.array([[808, 495], [892, 495], [802, 609], [896, 609]])
    corners5 = np.array([[1024, 608], [1118, 593], [1032, 664], [1134, 651]])

    canvas_1 = transform(img1, canvas, corners1)
    canvas_2 = transform(img2, canvas_1, corners2)
    canvas_3 = transform(img3, canvas_2, corners3)
    canvas_4 = transform(img4, canvas_3, corners4)
    canvas_5 = transform(img5, canvas_4, corners5)
    cv2.imwrite('output/part1.png', canvas_5)
    print("part 1 complete!")

def part2():
    img = cv2.imread('./input/screen.jpg')
    canvas = np.zeros((300, 300, 3)) # qr code size
    img_corners = [[369, 1038], [396, 1100], [553, 984], [600, 1038]] 
    canvas_corners = [[0, 0], [300, 0], [0, 300], [300, 300]]
    qr_code = transform_cor(img, canvas, canvas_corners, img_corners)
    cv2.imwrite('output/part2.png', qr_code)
    print("part 2 complete!")

def part3():
    img = cv2.imread('./input/crosswalk_front.jpg')
    canvas = np.zeros((407, 725, 3)) 
    # img_corners = [[150, 140], [150, 580], [296, 11], [291, 724]]
    img_corners = [[147, 109], [141, 582], [296, 12], [291, 724]]
    canvas_corners = [[0, 0], [725, 0], [0, 407], [725, 407]]
    top = transform_cor(img, canvas, canvas_corners, img_corners)
    cv2.imwrite('output/part3.png', top)
    print("part 3 complete!")

def main():
    # Part 1
    part1()
    # Part 2
    part2()
    # Part 3
    part3()

    ### find four corners in screen.jpg
    # img_front = cv2.imread('./input/crosswalk_front.jpg')
    # cv2.namedWindow('image')
    # cv2.setMouseCallback('image',OnMouseAction)  
    # cv2.imshow('image',img_front)
    # cv2.waitKey(50000)
    # cv2.destroyAllWindows()
    # ans = cv2.imread('./output/part3.png')
    # cv2.imshow('image', ans)
    # cv2.waitKey(30000)
    # cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
