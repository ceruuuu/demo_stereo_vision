import os
import numpy as np
import cv2
from matplotlib import pyplot as plt

def create_test_image(filename, width, height) :
    left_img = np.zeros((width, height, 3), dtype=np.uint8)
    left_filename = filename + "_left.jpg"
    right_img = np.zeros((width, height, 3), dtype=np.uint8)
    right_filename = filename + "_right.jpg"

    for i in range(130,230):
        for j in range(150,250):
            left_img[i][j] = 150

    for i in range(130,250):
        for j in range(200,300):
            right_img[i][j] = 150

    cv2.imwrite(left_filename, left_img)
    cv2.imshow('left', left_img)

    cv2.imwrite(right_filename, right_img)
    cv2.imshow('right', right_img)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

    imgL = cv2.imread(left_filename,0)
    imgR = cv2.imread(right_filename,0)
    stereo = cv2.StereoBM_create(numDisparities=16, blockSize=15)
    disparity = stereo.compute(imgL,imgR)
    plt.imshow(disparity,'gray')
    plt.show()


def crop_ZED_image(filename, width, height):
    image = cv2.imread(filename, cv2.IMREAD_UNCHANGED)

    left_img = image[0:height, 0:width/2].copy()
    left_filename = os.path.splitext(filename) + '_left.jpg'
    right_img = image[0:height, width/2:width].copy()
    right_filename = os.path.splitext(filename) + '_right.jpg'


    cv2.imwrite(left_filename, left_img)
    cv2.imwrite(right_filename, right_img)

def convert_to_16bit_image(filename):
    image = cv2.imread(filename, cv2.IMREAD_UNCHANGED)
    cvt_filename = os.path.splitext(filename) + "_16bit.png"

    image = image.astype('uint16')
    h, w = image.shape
    carr = np.copy(image)

    offset = 10000
    gain = 5000

    for i in range(h):
        for j in range(w):
            carr[i][j] = (carr[i][j] / 255) * gain + offset

    listimg = np.array(carr, dtype='uint16')
    cv2.imwrite(cvt_filename, listimg)


if __name__ == "__main__":
    '''
    용도에 따라 잘 주석처리해서 ...........씁쉬다
    '''

    filename = "./images/test"
    # "./images/test"라고 입력하면 images 폴더에 test_left.jpg, test_right.jpg로 저장됩니당
    create_test_image(filename, 360, 640)


    filename = "./images/zed/bag_1.jpg"
    # images/zed 폴더에 bag_1_left.jpg, bag_1_right.jpg로 저장됩니당
    crop_ZED_image(filename, 4416, 1242)


    filename = "./images/d01-2.jpg"
    # images 폴더에 d01-2_16bit.png로 저장됨니당
    convert_to_16bit_image(filename)