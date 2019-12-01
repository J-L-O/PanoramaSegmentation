import numpy as np
from PIL import Image


def convert_colors(img, color_dict, rgb=True):
    data = np.array(img)

    if rgb:
        dimensions = (data.shape[0], data.shape[1], 3)
    else:
        dimensions = (data.shape[0], data.shape[1])

    temp = np.zeros(dimensions, np.uint8)

    for color in color_dict:
        white_areas = data == color

        if len(data.shape) != 2:
            white_areas = white_areas[:, :, 0]

        temp[white_areas] = color_dict[color]

    converted = Image.fromarray(temp)

    return converted
