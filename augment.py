import os

from PIL import Image, ImageEnhance
from skimage import exposure as ex
import imageio
import sys
import cv2 as cv
import numpy as np
import imutils

from image_transformer import ImageTransformer

# img_shape =  (640, 480)

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

    f_img      = aug_hflip(img)

    r_img_15     = aug_rotate(img, image_transformer, 5)
    r_img_30     = aug_rotate(img, image_transformer, 15)
    r_img_45     = aug_rotate(img, image_transformer, 30)

    r_img_15_f   = aug_rotate(f_img, image_transformer, 5)
    r_img_30_f   = aug_rotate(f_img, image_transformer, 15)
    r_img_45_f   = aug_rotate(f_img, image_transformer, 30)    #7


    r_img_15_    = aug_rotate(img, image_transformer, -5)
    r_img_30_    = aug_rotate(img, image_transformer, -15)
    r_img_45_    = aug_rotate(img, image_transformer, -30)
    r_img_15__f  = aug_rotate(f_img, image_transformer, -5)
    r_img_30__f  = aug_rotate(f_img, image_transformer, -15)
    r_img_45__f  = aug_rotate(f_img, image_transformer, -30)     #13


    b_image_1    = aug_brightness(r_img_15, 0.7)
    b_image_2    = aug_brightness(r_img_30, 0.7)
    b_image_3    = aug_brightness(r_img_45, 0.7)
    b_image_4    = aug_brightness(r_img_15_f, 0.7)
    b_image_5    = aug_brightness(r_img_30_f, 0.7)
    b_image_6    = aug_brightness(r_img_45_f, 0.7)
    b_image_7    = aug_brightness(r_img_15_, 0.7)
    b_image_8    = aug_brightness(r_img_30_, 0.7)
    b_image_9    = aug_brightness(r_img_45_, 0.7)
    b_image_10   = aug_brightness(r_img_15__f, 0.7)
    b_image_11   = aug_brightness(r_img_30__f, 0.7)
    b_image_12   = aug_brightness(r_img_45__f, 0.7)     #25

    b_image_1_   = aug_brightness(r_img_15, 1.7)
    b_image_2_   = aug_brightness(r_img_30, 1.7)
    b_image_3_   = aug_brightness(r_img_45, 1.7)
    b_image_4_   = aug_brightness(r_img_15_f, 1.7)
    b_image_5_   = aug_brightness(r_img_30_f, 1.7)
    b_image_6_   = aug_brightness(r_img_45_f, 1.7)
    b_image_7_   = aug_brightness(r_img_15_, 1.7)
    b_image_8_   = aug_brightness(r_img_30_, 1.7)
    b_image_9_   = aug_brightness(r_img_45_, 1.7)
    b_image_10_  = aug_brightness(r_img_15__f, 1.7)
    b_image_11_  = aug_brightness(r_img_30__f, 1.7)
    b_image_12_  = aug_brightness(r_img_45__f, 1.7)     #37

    c_image_1    = aug_contrast(r_img_15, 3)
    c_image_2    = aug_contrast(r_img_30, 3)
    c_image_3    = aug_contrast(r_img_45, 3)
    c_image_4    = aug_contrast(r_img_15_f, 3)
    c_image_5    = aug_contrast(r_img_30_f, 3)
    c_image_6    = aug_contrast(r_img_45_f, 3)
    c_image_7    = aug_contrast(r_img_15_, 3)
    c_image_8    = aug_contrast(r_img_30_, 3)
    c_image_9    = aug_contrast(r_img_45_, 3)
    c_image_10   = aug_contrast(r_img_15__f, 3)
    c_image_11   = aug_contrast(r_img_30__f, 3)
    c_image_12   = aug_contrast(r_img_45__f, 3)      #49

    c_image_1_   = aug_contrast(r_img_15, 2)
    c_image_2_   = aug_contrast(r_img_30, 2)
    c_image_3_   = aug_contrast(r_img_45, 2)
    c_image_4_   = aug_contrast(r_img_15_f, 2)
    c_image_5_   = aug_contrast(r_img_30_f, 2)
    c_image_6_   = aug_contrast(r_img_45_f, 2)
    c_image_7_   = aug_contrast(r_img_15_, 2)
    c_image_8_   = aug_contrast(r_img_30_, 2)
    c_image_9_   = aug_contrast(r_img_45_, 2)
    c_image_10_  = aug_contrast(r_img_15__f, 2)
    c_image_11_  = aug_contrast(r_img_30__f, 2)
    c_image_12_  = aug_contrast(r_img_45__f, 2)   #61

    aug.append(r_img_15_f)
    aug.append(r_img_30_f)
    aug.append(r_img_45_f)
    aug.append(r_img_15_)
    aug.append(r_img_30_)
    aug.append(r_img_45_)
    aug.append(r_img_15__f)
    aug.append(r_img_30__f)
    aug.append(r_img_45__f)
    aug.append(b_image_1)
    aug.append(b_image_2)
    aug.append(b_image_3)
    aug.append(b_image_4)
    aug.append(b_image_5)
    aug.append(b_image_6)
    aug.append(b_image_7)
    aug.append(b_image_8)
    aug.append(b_image_9)
    aug.append(b_image_10)
    aug.append(b_image_11)
    aug.append(b_image_12)
    aug.append(b_image_1_)
    aug.append(b_image_2_)
    aug.append(b_image_3_)
    aug.append(b_image_4_)
    aug.append(b_image_5_)
    aug.append(b_image_6_)
    aug.append(b_image_7_)
    aug.append(b_image_8_)
    aug.append(b_image_9_)
    aug.append(b_image_10_)
    aug.append(b_image_11_)
    aug.append(b_image_12_)
    aug.append(c_image_1)
    aug.append(c_image_2)
    aug.append(c_image_3)
    aug.append(c_image_4)
    aug.append(c_image_5)
    aug.append(c_image_6)
    aug.append(c_image_7)
    aug.append(c_image_8)
    aug.append(c_image_9)
    aug.append(c_image_10)
    aug.append(c_image_11)
    aug.append(c_image_12)
    aug.append(c_image_1_)
    aug.append(c_image_2_)
    aug.append(c_image_3_)
    aug.append(c_image_4_)
    aug.append(c_image_5_)
    aug.append(c_image_6_)
    aug.append(c_image_7_)
    aug.append(c_image_8_)
    aug.append(c_image_9_)
    aug.append(c_image_10_)
    aug.append(c_image_11_)
    aug.append(c_image_12_)

    return aug

def dataAugment_onImage_weak(img,file_name, image_transformer) :

    #TODO : use for-loop and list as input-output of function to avoid hardcoding and copy-paste of function calls !

    aug = []

    f_img      = aug_hflip(img)

    r_img_45     = aug_rotate(img, image_transformer, 30)
    r_img_45_    = aug_rotate(img, image_transformer, -30)

    b_image_9_   = aug_brightness(r_img_45_, 1.7)

    c_image_9    = aug_contrast(r_img_45_, 3)
    c_image_3_   = aug_contrast(r_img_45, 2)
    c_image_9_   = aug_contrast(r_img_45_, 2)

    aug.append(f_img)
    aug.append(r_img_45_)
    aug.append(r_img_45)

    aug.append(b_image_9_)

    aug.append(c_image_9)
    aug.append(c_image_3_)
    aug.append(c_image_9_)

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
    image_folders = r"E:\DataSet\ClassifyByUsage_train\Train\10"
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

