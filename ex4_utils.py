import math

import cv2
import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
from scipy.ndimage import filters


def disparitySSD(img_l: np.ndarray, img_r: np.ndarray, disp_range: (int, int), k_size: int) -> np.ndarray:
    """
    img_l: Left image
    img_r: Right image
    range: Minimun and Maximum disparity range. Ex. (10,80)
    k_size: Kernel size for computing the SSD, kernel.shape = (k_size*2+1,k_size*2+1)
    return: Disparity map, disp_map.shape = Left.shape
    """

    img_l = (img_l * 255.0).astype(np.uint8)
    img_r = (img_r * 255.0).astype(np.uint8)
    # L = cv2.cvtColor(img_l, cv2.COLOR_RGB2GRAY)
    # R = cv2.cvtColor(img_r, cv2.COLOR_RGB2GRAY)
    return disp(img_l, img_r, disp_range, k_size, SSD) / 255.0


def SSD(img_l, img_r, r, c, k_size, disp_range):
    """
    :param img_l:
    :param img_r:
    :param r:
    :param c:
    :param k_size:
    :param disp_range:
    :return:
    """

    ymin = 0
    _min = np.inf
    X = img_l[(r - k_size): (r + k_size), (c - k_size): (c + k_size)]

    for col in range(c - disp_range[1] // 2, c + disp_range[1] // 2):
        if np.abs(col - c) < disp_range[0] // 2:
            continue
        if col - k_size < 0:
            continue
        if col + k_size > img_r.shape[1]:
            break

        X_ = img_r[(r - k_size): (r + k_size), (col - k_size): col + k_size]

        _sum = np.sum((X - X_) ** 2)
        if _sum < _min:
            _min = _sum
            ymin = col

    return ymin


def disparityNC(img_l: np.ndarray, img_r: np.ndarray, disp_range: (int, int), k_size: int) -> np.ndarray:
    """
    img_l: Left image
    img_r: Right image
    range: Minimun and Maximum disparity range. Ex. (10,80)
    k_size: Kernel size for computing the SSD, kernel.shape = (k_size*2+1,k_size*2+1)
    return: Disparity map, disp_map.shape = Left.shape
    """

    img_l = (img_l * 255.0).astype(np.uint8)
    img_r = (img_r * 255.0).astype(np.uint8)
    # L = cv2.cvtColor(img_l, cv2.COLOR_RGB2GRAY)
    # R = cv2.cvtColor(img_r, cv2.COLOR_RGB2GRAY)
    return disp(img_l, img_r, disp_range, k_size, NC)


def NC(img_l, img_r, r, c, k_size, disp_range):
    ymin = 0
    _max = 0
    X = img_l[(r - k_size): (r + k_size), (c - k_size): (c + k_size)]

    for col in range(c - disp_range[1] // 2, c + disp_range[1] // 2):
        if col - k_size < 0:
            continue
        if col + k_size > img_r.shape[1]:
            break
        X_ = img_r[(r - k_size): (r + k_size), (col - k_size): col + k_size]
        product = (X - np.mean(X)) * (X_ - np.mean(X_))
        stds = np.std(X) * np.std(X_)
        _sum = np.sum(product / stds)
        _sum /= X.size

        if _sum > _max:
            _max = _sum
            ymin = col

    return ymin


def disp(img_l: np.ndarray, img_r: np.ndarray, disp_range: (int, int), k_size: int, Similarity_Measure):
    disp_map = np.zeros(img_l.shape)
    for r in range(k_size, img_l.shape[0] - k_size):
        for c in range(k_size, img_r.shape[1] - k_size):
            y = Similarity_Measure(img_l, img_r, r, c, k_size, disp_range)
            disp_map[r][c] = abs(y - c) / 255
    return disp_map

# def disparitySSD(img_l: np.ndarray, img_r: np.ndarray, disp_range: (int, int), k_size: int) -> np.ndarray:
#     """
#     img_l: Left image
#     img_r: Right image
#     range: Minimum and Maximum disparity range. Ex. (10,80)
#     k_size: Kernel size for computing the SSD, kernel.shape = (k_size*2+1,k_size*2+1)
#
#     return: Disparity map, disp_map.shape = Left.shape
#     """
#     ker_size = (k_size * 2 + 1)
#     disp_map = np.zeros(img_l.shape)
#
#     for row in range(img_l.shape[0]):
#         for col in range(img_l.shape[1]):
#             shift = 0
#             min_ssd = np.inf
#             for offset in range(disp_range[0], disp_range[1]):
#                 SSD = 0
#                 for x in range(0, ker_size):
#                     for y in range(0, ker_size):
#                         if 0 <= row + x - offset < img_r.shape[0] and 0 <= col + y - offset < img_r.shape[1]:
#                             SSD += ((img_l[row, col]) - (img_r[row + x - offset, col + y - offset])) ** 2
#
#                 if SSD < min_ssd:
#                     min_ssd = SSD
#                     shift = offset
#             disp_map[row][col] = shift
#     # img_l = cv.imread('input/pair0-L.png', 0)
#     # img_r = cv.imread('input/pair0-R.png', 0)
#     # stereo = cv.StereoBM_create(numDisparities=16, blockSize=15)
#     # disparity = stereo.compute(img_l, img_r)
#     # plt.imshow(disparity, 'gray')
#     # plt.show()
#     return disp_map
#
#
# def disparityNC(img_l: np.ndarray, img_r: np.ndarray, disp_range: (int, int), k_size: int) -> np.ndarray:
#     """
#     img_l: Left image
#     img_r: Right image
#     range: The Maximum disparity range. Ex. 80
#     k_size: Kernel size for computing the NormCorolation, kernel.shape = (k_size*2+1,k_size*2+1)
#
#     return: Disparity map, disp_map.shape = Left.shape
#     """
#     ker_size = (k_size * 2 + 1)
#     disp_map = np.zeros(img_l.shape)
#
#     for row in range(img_l.shape[0]):
#         for col in range(img_l.shape[1]):
#             max_corr = -np.inf
#             for offset in range(disp_range[0], disp_range[1]):
#                 corr = 0
#                 l = 0
#                 r = 0
#                 for x in range(0, ker_size):
#                     for y in range(0, ker_size):
#                         if 0 <= row + x - offset < img_r.shape[0] and 0 <= col + y - offset < img_r.shape[1]:
#                             corr += ((img_l[row, col]) * (img_r[row + x - offset, col + y - offset]))
#                             l += ((img_r[row, col]) * (img_r[row + x - offset, col + y - offset]))
#                             r += ((img_l[k_size, k_size]) * (img_l[k_size + x - offset, k_size + y - offset]))
#
#                 if corr > max_corr:
#                     max_corr = corr
#             disp_map[row][col] = corr / math.sqrt(l * r)
#     return disp_map


def computeHomography(src_pnt: np.ndarray, dst_pnt: np.ndarray) -> (np.ndarray, float):
    """
    Finds the homography matrix, M, that transforms points from src_pnt to dst_pnt.
    returns the homography and the error between the transformed points to their
    destination (matched) points. Error = np.sqrt(sum((M.dot(src_pnt)-dst_pnt)**2))

    src_pnt: 4+ keypoints locations (x,y) on the original image. Shape:[4+,2]
    dst_pnt: 4+ keypoints locations (x,y) on the destenation image. Shape:[4+,2]

    return: (Homography matrix shape:[3,3], Homography error)
    """
    # create A
    A = []
    for i in range(len(src_pnt)):
        A.append([src_pnt[i][0], src_pnt[i][1], 1, 0, 0, 0, -dst_pnt[i][0] * src_pnt[i][0], -dst_pnt[i][0] * src_pnt[i][1], -dst_pnt[i][0]])
        A.append([0, 0, 0, src_pnt[i][0], src_pnt[i][1], 1, -dst_pnt[i][1] * src_pnt[i][0], -dst_pnt[i][1] * src_pnt[i][1], -dst_pnt[i][1]])
    A = np.array(A)

    U, S, Vh = np.linalg.svd(np.asarray(A))
    # create the Homography matrix
    M = (Vh[-1, :] / Vh[-1, -1]).reshape(3, 3)

    src = []
    dst = []
    # find error
    for i in range(src_pnt.shape[0]):
        src.append(np.append(src_pnt[i], 1))
        dst.append(np.append(dst_pnt[i], 1))
    src = np.transpose(src)
    dst = np.transpose(dst)
    E = np.sqrt(sum((M.dot(src) / M.dot(src)[-1] - dst)**2))
    return M, E


def warpImag(src_img: np.ndarray, dst_img: np.ndarray) -> None:
    """
    Displays both images, and lets the user mark 4 or more points on each image.
    Then calculates the homography and transforms the source image on to the destination image.
    Then transforms the source image onto the destination image and displays the result.

    src_img: The image that will be ’pasted’ onto the destination image.
    dst_img: The image that the source image will be ’pasted’ on.

    output: None.
    """

    dst_p = []
    src_p = []
    fig1 = plt.figure()

    def onclick_1(event):
        x = event.xdata
        y = event.ydata
        print("Loc: {:.0f},{:.0f}".format(x, y))

        plt.plot(x, y, '*r')
        dst_p.append([x, y])

        if len(dst_p) == 4:
            plt.close()
        plt.show()

    # display image 1
    cid = fig1.canvas.mpl_connect('button_press_event', onclick_1)
    plt.imshow(dst_img)
    plt.show()
    dst_p = np.array(dst_p)

    def onclick_2(event):
        x = event.xdata
        y = event.ydata
        print("Loc: {:.0f},{:.0f}".format(x, y))

        plt.plot(x, y, '*r')
        src_p.append([x, y])

        if len(src_p) == 4:
            plt.close()
        plt.show()

    fig2 = plt.figure()
    # display image 2
    cid = fig2.canvas.mpl_connect('button_press_event', onclick_2)
    plt.gray()
    plt.imshow(src_img)
    plt.show()
    src_p = np.array(src_p)

    M, m = cv2.findHomography(src_p, dst_p)
    warp = np.zeros(dst_img.shape)
    for i in range(src_img.shape[0]):
        for j in range(src_img.shape[1]):
            xy = np.array([j, i, 1]).T
            new_p = M @ xy
            y_ = int(new_p[0] / new_p[-1])
            x_ = int(new_p[1] / new_p[-1])
            if 0 <= x_ < src_img.shape[0] and 0 <= y_ < src_img.shape[1]:
                warp[x_, y_] = src_img[i, j]
            else:
                warp[x_, y_] = src_img[i, j]
    mask = warp == 0
    out = dst_img * mask + warp * (1 - mask)
    plt.imshow(out)
    plt.show()
