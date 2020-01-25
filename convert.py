from os import listdir
from os.path import isfile, join

import click
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


SYNTHIA_Color_to_SYNTHIA_Label = {
    (20, 215, 197): 0,  # void
    (207, 248, 132): 1,  # sky
    (183, 244, 155): 2,  # building
    (144, 71, 111): 3,  # road
    (128, 48, 71): 4,  # sidewalk
    (50, 158, 75): 5,   # fence
    (241, 169, 37): 6,  # vegetation
    (222, 181, 51): 7,   # pole
    (244, 104, 161): 8,  # car
    (31, 133, 226): 9,  # traffic sign
    (204, 47, 7): 10,  # pedestrian
    (170, 252, 0): 11,  # bicycle
    (32, 166, 124): 12,  # lanemarking
    (122, 113, 97): 13,  # void
    (46, 229, 72): 14,  # void
    (250, 163, 41): 15  # traffic light
}

SYNTHIA_Label_to_SYNTHIA_Color = {
    0: (0, 0, 0),  # void
    1: (128, 128, 128),  # sky
    2: (128, 0, 0),  # building
    3: (128, 64, 128),  # road
    4: (0, 0, 192),  # sidewalk
    5: (64, 64, 128),  # fence
    6: (128, 128, 0),  # vegetation
    7: (192, 192, 128),  # pole
    8: (64, 0, 128),  # car
    9: (192, 128, 128),  # traffic sign
    10: (64, 64, 0),  # pedestrian
    11: (0, 128, 192),  # bicycle
    12: (0, 172, 0),  # lanemarking
    13: (0, 0, 0),  # void
    14: (0, 0, 0),  # void
    15: (0, 128, 128)  # traffic light
}

SYNTHIA_Label_to_REDUCED_Label = {
    0: 0,  # void -> void
    1: 1,  # sky -> sky
    2: 2,  # building -> building
    3: 3,  # road -> road
    4: 4,  # sidewalk -> sidewalk
    5: 5,  # fence -> fence
    6: 6,  # vegetation -> vegetation
    7: 7,  # pole -> sign
    8: 8,  # car -> car
    9: 7,  # traffic sign -> sign
    10: 9,  # pedestrian -> person
    11: 9,  # bicycle -> person
    12: 10,  # lanemarking -> lanemarking
    13: 0,  # void -> void
    14: 0,  # void -> void
    15: 7  # traffic light -> sign
}

REDUCED_Label_to_SYNTHIA_Color = {
    0: (0, 0, 0),  # void
    1: (128, 128, 128),  # sky
    2: (128, 0, 0),  # building
    3: (128, 64, 128),  # road
    4: (0, 0, 192),  # sidewalk
    5: (64, 64, 128),  # fence
    6: (128, 128, 0),  # vegetation
    7: (192, 192, 128),  # sign
    8: (64, 0, 128),  # car
    9: (64, 64, 0),  # person
    10: (0, 172, 0),  # lanemarking
}

CITYSCAPES_Label_to_CITYSCAPES_Color = {
    0: (20, 215, 197),  # road
    1: (207, 248, 132),  # sidewalk
    2: (183, 244, 155),  # building
    3: (144, 71, 111),  # wall
    4: (128, 48, 71),  # fence
    5: (50, 158, 75),  # pole
    6: (241, 169, 37),  # traffic light
    7: (222, 181, 51),  # traffic sign
    8: (244, 104, 161),  # vegetation
    9: (31, 133, 226),  # terrain
    10: (204, 47, 7),  # sky
    11: (170, 252, 0),  # person
    12: (32, 166, 124),  # rider
    13: (122, 113, 97),  # car
    14: (46, 229, 72),  # truck
    15: (250, 163, 41),  # bus
    16: (149, 154, 55),  # train
    17: (104, 170, 63),  # motorcycle
    18: (46, 227, 147),  # bicycle
}

CITYSCAPES_Color_to_SYNTHIA_Label = {
        (20, 215, 197): 3,  # road -> road
        (207, 248, 132): 4,  # sidewalk -> sidewalk
        (183, 244, 155): 2,  # building -> building
        (144, 71, 111): 2,  # wall -> building
        (128, 48, 71): 5,  # fence -> fence
        (50, 158, 75): 7,  # pole -> pole
        (241, 169, 37): 15,  # traffic light -> traffic light
        (222, 181, 51): 9,  # traffic sign -> traffic sign
        (244, 104, 161): 6,  # vegetation -> vegetation
        (31, 133, 226): 0,  # terrain -> void
        (204, 47, 7): 1,  # sky -> sky
        (170, 252, 0): 10,  # person -> pedestrian
        (32, 166, 124): 11,  # rider -> bicycle
        (122, 113, 97): 8,  # car -> car
        (46, 229, 72): 8,  # truck -> car
        (250, 163, 41): 8,  # bus -> car
        (149, 154, 55): 8,  # train -> car
        (104, 170, 63): 8,  # motorcycle -> car
        (46, 227, 147): 11,  # bicycle -> bicycle
}

CITYSCAPES_Color_to_SYNTHIA_Color = {
    (20, 215, 197): (128, 64, 128),  # road -> road
    (207, 248, 132): (0, 0, 192),  # sidewalk -> sidewalk
    (183, 244, 155): (128, 0, 0),  # building -> building
    (144, 71, 111): (128, 0, 0),  # wall -> building
    (128, 48, 71): (64, 64, 128),  # fence -> fence
    (50, 158, 75): (192, 192, 128),  # pole -> pole
    (241, 169, 37): (0, 128, 128),  # traffic light -> traffic light
    (222, 181, 51): (192, 128, 128),  # traffic sign -> traffic sign
    (244, 104, 161): (128, 128, 0),  # vegetation -> vegetation
    (31, 133, 226): (0, 0, 0),  # terrain -> void
    (204, 47, 7): (128, 128, 128),  # sky -> sky
    (170, 252, 0): (64, 64, 0),  # person -> pedestrian
    (32, 166, 124): (0, 128, 192),  # rider -> bicycle
    (122, 113, 97): (64, 0, 128),  # car -> car
    (46, 229, 72): (64, 0, 128),  # truck -> car
    (250, 163, 41): (64, 0, 128),  # bus -> car
    (149, 154, 55): (64, 0, 128),  # train -> car
    (104, 170, 63): (64, 0, 128),  # motorcycle -> car
    (46, 227, 147): (0, 128, 192),  # bicycle -> bicycle
}

color_dicts = {'SYNTHIA_Color_to_SYNTHIA_Label': (SYNTHIA_Color_to_SYNTHIA_Label, False),
               'SYNTHIA_Label_to_SYNTHIA_Color': (SYNTHIA_Label_to_SYNTHIA_Color, True),
               'CITYSCAPES_Label_to_CITYSCAPES_Color': (CITYSCAPES_Label_to_CITYSCAPES_Color, True),
               'CITYSCAPES_Color_to_SYNTHIA_Label': (CITYSCAPES_Color_to_SYNTHIA_Label, False),
               'CITYSCAPES_Color_to_SYNTHIA_Color': (CITYSCAPES_Color_to_SYNTHIA_Color, True),
               'SYNTHIA_Label_to_REDUCED_Label': (SYNTHIA_Label_to_REDUCED_Label, False),
               'REDUCED_Label_to_SYNTHIA_Color': (REDUCED_Label_to_SYNTHIA_Color, True)}


@click.command()
@click.option('--input', '-i', default='./SYNTHIA-PANO/PREDICTIONS/seqs02_fall/', help='The folder containing the input images.', type=click.Path(exists=True))
@click.option('--output', '-o', default='./SYNTHIA-PANO/PREDICTIONS_Converted_Labels/seqs02_fall/', help='Where to store the converted images.', type=click.Path(exists=True))
@click.option('--conversion', help='Which conversion shall be be applied.', type=str)
def convert(input, output, conversion):
    files = [f for f in listdir(input) if isfile(join(input, f))]

    for file in files:
        img = Image.open(f'{input}{file}')
        converted = convert_colors(img, color_dicts[conversion][0], color_dicts[conversion][1])
        converted.save(f'{output}{file}')


if __name__ == '__main__':
    convert()
