import os

from PIL import Image, ImageEnhance
from skimage import exposure as ex
import imageio
import sys
import cv2 as cv
import numpy as np
import imutils

from image_transformer import ImageTransformer

img_shape =  (640, 480)

def aug_normalize(img):
    b,g,r = cv.split(img)
    m_std = cv.meanStdDev(img, 0, 1)

    m  = m_std[0]
    st = m_std[1]

    b_normal = (b-m[0])/st[0]
    g_normal = (g-m[1])/st[1]
    r_normal = (r-m[2])/st[2]

    n_image = cv.merge((b_normal, g_normal, r_normal))
    cv.imshow("normalized", n_image)
    cv.waitKey(0)

def aug_histEq(img):
    if (len(img.shape) == 2):  # gray
        outImg = ex.equalize_hist(img[:, :]) * 255
    elif (len(img.shape) == 3):  # RGB
        outImg = np.zeros((img.shape[0], img.shape[1], 3))
        for channel in range(img.shape[2]):
            outImg[:, :, channel] = ex.equalize_hist(img[:, :, channel]) * 255

    outImg[outImg > 255] = 255
    outImg[outImg < 0] = 0
    return outImg.astype(np.uint8)

def aug_hflip(img):
    flip_image = cv.flip(img, 1)
    return  flip_image

def aug_rotate_center(img, angle=1):
    img_rotate = imutils.rotate_bound(img, angle)
    return img_rotate

def aug_rotate(img, it, ang):
    rotated_img = it.rotate_along_axis(phi=ang, dx=5)
    # cv.imshow("aug_rotate", rotated_img)
    return rotated_img

def aug_brightness(img, enhanceVal = 2.0):
    img_pil = Image.fromarray(img)
    enhancer = ImageEnhance.Brightness(img_pil)

    imgOut_pil = enhancer.enhance(enhanceVal)
    im_brightness = np.asarray(imgOut_pil)
    return im_brightness

def aug_contrast(im,enhanceVal=1):
    img_pil = Image.fromarray(im)
    enhancer = ImageEnhance.Contrast(img_pil)
    imgOut_pil = enhancer.enhance(enhanceVal)
    im_contrast = np.asarray(imgOut_pil)
    return im_contrast

def dataAugment_onImage(img,file_name, image_transformer) :

    #TODO : use for-loop and list as input-output of function to avoid hardcoding and copy-paste of function calls !

    aug = []
    f_img        = aug_hflip(img)
    r_img_15     = aug_rotate(img, image_transformer, 15)
    r_img_15__f  = aug_rotate(f_img, image_transformer, -15)
    b_image_1    = aug_brightness(img, 0.7)
    b_image_2    = aug_brightness(f_img, 0.7)
    c_image_1    = aug_contrast(f_img, 3)

    aug.append(f_img)
    aug.append(r_img_15)
    aug.append(r_img_15__f)
    aug.append(b_image_1)
    aug.append(b_image_2)
    aug.append(c_image_1)

    return aug

def aug_addNoise(img_) :
    img = img_/255.0
    x = np.random.randint(0, 10, 1)

    if (x <= 5):    # add noise to image with probability of 0.50

         return img_ ,False
    else:
        noise = np.random.normal(loc=0, scale=1, size=img.shape)
        img2 = img * 2
        n2 = np.clip(np.where(img2 <= 1, (img2 * (1 + noise * 0.2)), (1 - img2 + 1) * (1 + noise * 0.2) * -1 + 2) / 2, 0, 1)
        return n2, True

if __name__ == "__main__":
    image_folders = r"E:\DataSet\ClassifyByUsage_train\Train\9"
    img_list = []
    for root, dirs, files in os.walk(image_folders):
        for file in files:
            fullname = (os.path.join(root,file))
            fname    = fullname.split('/')[-1].split("\\")[-1].split(".")[0]
            img = cv.imread(fullname)
            it  = ImageTransformer(fullname, (img.shape[0], img.shape[1]))
            img_list = dataAugment_onImage(img, fullname,it)
            cntr = 0
            for im in img_list :
                cntr +=1
                new_name = os.path.dirname(fullname) + "/" + fname + "_a_{}.jpg".format(cntr)
                img_noised,nosie_added = aug_addNoise(im)
                if nosie_added == True:
                    cv.imwrite(new_name, img_noised*255.0)
                else :
                    cv.imwrite(new_name, img_noised)
                print(new_name)
                cv.imshow("new_name", img_noised)
                cv.waitKey(1)

