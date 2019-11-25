import numpy as np


def cvt2color(img, color_dict):
    img_h, img_w = img.shape[0], img.shape[1]
    temp = np.zeros((img_h, img_w, 3))

    for x in range(img_w):
        for y in range(img_h):
            trainId = img[y][x]

            temp[y][x] = color_dict[tuple(trainId)]
    temp = temp.astype(np.uint8)
    return temp
