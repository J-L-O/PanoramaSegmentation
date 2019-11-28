from os import listdir
from os.path import isfile, join
from PIL import Image

from ConvertColors import convert_colors

base_path = './SYNTHIA-PANO/PREDICTIONS/seqs02_fall/'
out_path = './SYNTHIA-PANO/PREDICTIONS_Converted_Color/seqs02_fall/'

color_dict_CITYSCAPES_to_SYNTHIA_color = {
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

files = [f for f in listdir(base_path) if isfile(join(base_path, f))]

for file in files:
    img = Image.open(f'{base_path}{file}')
    converted = convert_colors(img, color_dict_CITYSCAPES_to_SYNTHIA_color)
    converted.save(f'{out_path}{file}')
