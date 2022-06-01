import cv2
import numpy as np
import matplotlib.pyplot as plt


def disparitySSD(img_l: np.ndarray, img_r: np.ndarray, disp_range: (int, int), k_size: int) -> np.ndarray:
    """
    img_l: Left image
    img_r: Right image
    range: Minimum and Maximum disparity range. Ex. (10,80)
    k_size: Kernel size for computing the SSD, kernel.shape = (k_size*2+1,k_size*2+1)

    return: Disparity map, disp_map.shape = Left.shape
    """
    # define the kernel size and the disparity map
    ker_size = (k_size * 2 + 1)
    disp_map = np.zeros(img_l.shape)

    # compute the SSD for each pixel
    for row in range(img_l.shape[0] - ker_size):
        for col in range(img_l.shape[1] - ker_size):
            # define the shift and the minimum SSD
            shift = 0
            min_ssd = np.inf
            # then for each offset in the disparity range find the SSD
            for offset in range(disp_range[0], disp_range[1]):
                if col - offset >= 0 and col - offset + ker_size < img_r.shape[1]:
                    window = img_r[row: row + ker_size, col - offset: col - offset + ker_size]
                    SSD = np.sum((img_l[row: row + ker_size, col: col + ker_size] - window) ** 2)
                # if the SSD is smaller than the minimum SSD, update the minimum SSD and the shift
                if SSD < min_ssd:
                    min_ssd = SSD
                    shift = offset
            # update the disparity map
            disp_map[row][col] = shift
    return disp_map


def disparityNC(img_l: np.ndarray, img_r: np.ndarray, disp_range: (int, int), k_size: int) -> np.ndarray:
    """
    img_l: Left image
    img_r: Right image
    range: The Maximum disparity range. Ex. 80
    k_size: Kernel size for computing the NormCorolation, kernel.shape = (k_size*2+1,k_size*2+1)

    return: Disparity map, disp_map.shape = Left.shape
    """
    # define the kernel size and the disparity map
    ker_size = (k_size * 2 + 1)
    disp_map = np.zeros(img_l.shape)

    # compute the NormCorolation for each pixel
    for row in range(img_l.shape[0] - ker_size):
        for col in range(img_l.shape[1] - ker_size):
            # define the shift and the maximum NormCorolation
            shift = 0
            max_NClr = -np.inf
            # then for each offset in the disparity range find the NormCorolation
            for offset in range(disp_range[0], disp_range[1]):
                NClr = 0
                if col - offset >= 0 and col - offset + ker_size < img_r.shape[1]:
                    window = img_r[row: row + ker_size, col - offset: col - offset + ker_size]
                    Rlr = np.sum((img_l[row: row + ker_size, col: col + ker_size] * window))
                    Rrr = np.sum(window ** 2)
                    Rll = np.sum(img_r[row: row + ker_size, col: col + ker_size] * window)

                    # compute the NormCorolation
                    sqr = np.sqrt(Rrr*Rll)
                    if sqr != 0:
                        NClr = Rlr / sqr
                    else:
                        print("sqr is 0")

                # if the NormCorolation is larger than the maximum NormCorolation,
                # update the maximum NormCorolation and the shift
                if NClr > max_NClr:
                    max_NClr = NClr
                    shift = offset
            # update the disparity map
            disp_map[row][col] = shift
    return disp_map


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

    # get the svd values
    U, S, Vh = np.linalg.svd(np.asarray(A))
    # create the Homography matrix
    M = (Vh[-1, :] / Vh[-1, -1]).reshape(3, 3)

    src = []
    dst = []
    # find error
    for i in range(src_pnt.shape[0]):
        # add each pixel 1
        src.append(np.append(src_pnt[i], 1))
        dst.append(np.append(dst_pnt[i], 1))
    src = np.transpose(src)
    dst = np.transpose(dst)
    # compute the error
    E = np.sqrt(np.sum((M.dot(src) / M.dot(src)[-1] - dst) ** 2))
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

    # create onclick function for the second image
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

    # compute the homography
    M, m = cv2.findHomography(src_p, dst_p)

    # for each pixel in the source image
    proj_src = np.zeros(dst_img.shape)
    for i in range(src_img.shape[0]):
        for j in range(src_img.shape[1]):
            point = np.array([j, i, 1]).T
            # compute the location of the pixel on the destination image
            new_p = M.dot(point)
            y = int(new_p[0] / new_p[-1])
            x = int(new_p[1] / new_p[-1])
            try:
                proj_src[x, y] = src_img[i, j]
            except IndexError:
                continue
    # create mask
    mask = proj_src == 0
    # blend
    canvas = dst_img * mask + (1 - mask) * proj_src
    plt.imshow(canvas)
    plt.show()
