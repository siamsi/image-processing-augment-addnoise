import numpy as np
import cv2 as cv

def aug_addNoise(img_) :
    img = img_/255.0
    x = np.random.randint(0, 10, 1)

    # if (x <= -1.0):    # add noise to image with probability of 0.50
    #     return img_
    # else:

    noise = np.random.normal(loc=0, scale=1, size=img.shape)
    img2 = img * 2
    n2 = np.clip(np.where(img2 <= 1, (img2 * (1 + noise * 0.2)), (1 - img2 + 1) * (1 + noise * 0.2) * -1 + 2) / 2, 0, 1)
    cv.namedWindow("test", cv.WINDOW_NORMAL)
    cv.imshow("test", n2)
    cv.waitKey(0)
    return n2

if __name__ == "__main__" :
    img_name = r'/media/ds/WorkSpace/az_tr_1M/s_az/s1_az_10-AA-830_9.jpg'
    img = cv.imread(img_name)

    # noise =  np.random.normal(loc=0, scale=1, size=img.shape)
    # img2 = img*2
    # n2 = np.clip(np.where(img2 <= 1, (img2*(1 + noise*0.2)), (1-img2+1)*(1 + noise*0.2)*-1 + 2)/2, 0,1)
    # cv.namedWindow("test", cv.WINDOW_NORMAL)
    # cv.imshow("test", n2)
    # cv.waitKey(0)

    im = aug_addNoise(img)
    cv.imshow("test", im)
    cv.waitKey(0)


